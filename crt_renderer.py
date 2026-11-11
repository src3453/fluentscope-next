"""
CRT oscilloscope beam renderer for FluentScope Next (OpenGL 3.3 via moderngl,
running inside the SDL2 window that pygame creates with the OPENGL flag).

Beam model after:
  - Oscilloscope by Hansi Raber (https://github.com/kritzikratzi/Oscilloscope,
    MIT License, (c) 2017 hansi raber): bin/data/shaders/osci.frag, src/OsciMesh.cpp
  - woscope by m1el (https://m1el.github.io/woscope-how/): physically based
    rendering of the electron beam

Every sample-to-sample step of the beam is drawn as a quad around the segment.
The fragment shader evaluates the closed-form integral of a gaussian spot moving
along the segment at constant speed during one time step:

    I(p) = E * sigma / (2 l) * exp(-py^2 / 2 sigma^2)
             * (erf(px / (sqrt2 sigma)) - erf((px - l) / (sqrt2 sigma)))

(p in segment-local coordinates, start at (0, 0), end at (l, 0)). Every step
deposits the same energy E, so slow parts of the trace come out bright and fast
edges dim, and adjacent segments join without seams. The segments are added
into a float framebuffer that keeps the phosphor: every frame it is multiplied
by exp(-dt / persistence) before the new beam is added. A two-level bloom
(half / quarter resolution gaussian blur) gives the glow, and the final pass
maps the light through the display gamma curve with a smooth saturation knee
(overdriven parts bleach toward white) and lays it over the 2D UI surface
drawn by pygame.
"""

import math

import numpy as np

import moderngl

try:
    from scipy.signal import resample_poly
except ImportError:  # linear interpolation fallback
    resample_poly = None


# ==========================
# Shaders
# ==========================

BEAM_VERT = """
#version 330
uniform vec2 uViewport;   // framebuffer size in pixels
uniform float uSigma;     // beam spot std deviation in pixels

in vec4 aSeg;             // x0, y0, x1, y1 (pixels, y down)
in float aEnergy;         // energy deposited during this time step

out vec3 vUVL;            // segment-local x, y and segment length
out float vEnergy;

void main() {
    // 4 vertices per instance (triangle strip): 0,1 around p0 / 2,3 around p1
    int id = gl_VertexID & 3;

    vec2 p0 = aSeg.xy;
    vec2 p1 = aSeg.zw;
    vec2 d = p1 - p0;
    float len = length(d);
    vec2 dir = len > 1e-5 ? d / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);

    float r = uSigma * 4.0; // the gaussian is ~0 beyond 4 sigma
    float side = (id & 1) == 0 ? -1.0 : 1.0;
    bool at_end = id >= 2;

    vec2 p = (at_end ? p1 + dir * r : p0 - dir * r) + nrm * (side * r);

    vUVL = vec3(at_end ? len + r : -r, side * r, len);
    vEnergy = aEnergy;

    vec2 ndc = p / uViewport * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
}
"""

BEAM_FRAG = """
#version 330
#define SQRT2 1.4142135623730951
#define INV_SQRT_TAU 0.3989422804014327
uniform float uSigma;
uniform vec3 uColor;

in vec3 vUVL;
in float vEnergy;
out vec4 fragColor;

// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7); accurate enough that the
// erf difference of very short segments does not turn into noise
float erf_as(float x) {
    float s = sign(x);
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                     - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

void main() {
    float len = vUVL.z;
    vec2 xy = vUVL.xy;
    float s = uSigma;
    float gy = exp(-xy.y * xy.y / (2.0 * s * s));
    float a;

    if (len < 1e-3 * s) {
        // (almost) standing beam: the limit of the integral below
        a = exp(-xy.x * xy.x / (2.0 * s * s)) * gy * INV_SQRT_TAU;
    } else {
        a = (erf_as(xy.x / (SQRT2 * s)) - erf_as((xy.x - len) / (SQRT2 * s)))
            * gy * s / (2.0 * len);
    }

    fragColor = vec4(uColor * (a * vEnergy), 1.0);
}
"""

# full-screen triangle without vertex buffers
FULLSCREEN_VERT = """
#version 330
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FADE_FRAG = """
#version 330
uniform float uDecay;
out vec4 fragColor;
void main() {
    fragColor = vec4(vec3(uDecay), 1.0);
}
"""

# 2x2 box (one bilinear tap per 2x2 block, 4 taps = 4x4 texels): downsample by 2
DOWNSAMPLE_FRAG = """
#version 330
uniform sampler2D uSrc;
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec2 t = 1.0 / vec2(textureSize(uSrc, 0));
    vec3 c = texture(uSrc, vUV + t * vec2(-1.0, -1.0)).rgb
           + texture(uSrc, vUV + t * vec2( 1.0, -1.0)).rgb
           + texture(uSrc, vUV + t * vec2(-1.0,  1.0)).rgb
           + texture(uSrc, vUV + t * vec2( 1.0,  1.0)).rgb;
    fragColor = vec4(c * 0.25, 1.0);
}
"""

# 9-tap gaussian in 5 bilinear fetches
BLUR_FRAG = """
#version 330
uniform sampler2D uSrc;
uniform vec2 uDir; // (1, 0) or (0, 1)
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec2 t = uDir / vec2(textureSize(uSrc, 0));
    vec3 c = texture(uSrc, vUV).rgb * 0.2270270270;
    c += texture(uSrc, vUV + t * 1.3846153846).rgb * 0.3162162162;
    c += texture(uSrc, vUV - t * 1.3846153846).rgb * 0.3162162162;
    c += texture(uSrc, vUV + t * 3.2307692308).rgb * 0.0702702703;
    c += texture(uSrc, vUV - t * 3.2307692308).rgb * 0.0702702703;
    fragColor = vec4(c, 1.0);
}
"""

COMPOSITE_FRAG = """
#version 330
uniform sampler2D uUI;     // pygame surface, top row first
uniform sampler2D uHDR;    // accumulated beam light
uniform sampler2D uBloomNear;
uniform sampler2D uBloomFar;
uniform sampler2D uLines;  // premultiplied plain line layer
uniform sampler2D uOverlay; // pygame SRCALPHA surface (text, menu) on top of everything
uniform int uCRT;
uniform int uLineLayer;
uniform int uUseOverlay;
uniform vec4 uUIRect; // left, top, right, bottom in UI pixels
uniform int uInk;          // light background: the beam darkens like ink
uniform float uExposure;
uniform float uBloomNearStr;
uniform float uBloomFarStr;
uniform float uWhite;      // how much saturated phosphor bleaches to white
uniform float uGamma;
uniform float uKnee;       // saturation hardness (higher = closer to a hard clip at 1)
uniform float uVignette;

in vec2 vUV;
out vec4 fragColor;

// Brightness along the gamma curve, saturating smoothly toward 1:
//   v = L^(1/gamma)                   (display gamma of linear beam light)
//   b = v / (1 + v^knee)^(1/knee)     (~v while dim, -> 1 when overdriven)
float gamma_saturate(float L, out float v) {
    v = pow(max(L, 0.0), 1.0 / uGamma);
    return v / pow(1.0 + pow(v, uKnee), 1.0 / uKnee);
}

vec3 base_color(vec2 uiUV);

void main() {
    vec2 uiUV = vec2(vUV.x, 1.0 - vUV.y);
    vec3 color = base_color(uiUV);
    if (uUseOverlay == 1) {
        vec4 top = texture(uOverlay, uiUV);
        color = mix(color, top.rgb, top.a);
    }
    fragColor = vec4(color, 1.0);
}

vec3 base_color(vec2 uiUV) {
    vec3 ui = texture(uUI, uiUV).rgb;
    vec2 uiPixel = uiUV * vec2(textureSize(uUI, 0));

    if (all(greaterThanEqual(uiPixel, uUIRect.xy)) && all(lessThan(uiPixel, uUIRect.zw))) {
        return ui; // no beam / lines inside this rectangle
    }
    if (uLineLayer == 1) {
        vec4 line = texture(uLines, vUV);
        ui = ui * (1.0 - line.a) + line.rgb;
    }
    if (uCRT == 0) {
        return ui;
    }

    vec3 hdr = texture(uHDR, vUV).rgb
             + texture(uBloomNear, vUV).rgb * uBloomNearStr
             + texture(uBloomFar, vUV).rgb * uBloomFarStr;
    hdr = max(hdr, vec3(0.0)) * uExposure;

    // the curve works on the brightness; the hue (max channel = 1) is kept,
    // so a green trace stays green until it saturates
    float peak = max(hdr.r, max(hdr.g, hdr.b));
    vec3 hue = peak > 1e-6 ? hdr / peak : vec3(0.0);
    float v;
    float b = gamma_saturate(peak, v);

    vec3 color;
    if (uInk == 1) {
        color = ui * (1.0 - hue * b); // absorbed light
    } else {
        // overdriven phosphor bleaches toward white
        vec3 beam = mix(hue, vec3(1.0), uWhite * smoothstep(1.0, 4.0, v)) * b;
        color = 1.0 - (1.0 - ui) * (1.0 - beam); // screen blend

        vec2 d = vUV - 0.5;
        color *= 1.0 - uVignette * dot(d, d) * 2.0;
    }

    return color;
}
"""

# Plain (non-CRT) polyline: every segment is a capsule (rounded ends) rendered
# as an instanced quad. The coverage of each pixel comes from its distance to
# the segment, so joints and ends are round and the width is free (subpixel).
# Coverage is merged with MAX blending, so overlapping capsules at the joints
# do not get darker / brighter like stacked alpha would.
LINE_VERT = """
#version 330
uniform vec2 uViewport;   // framebuffer size in pixels
uniform float uExtent;    // half width + AA fringe, pixels

in vec4 aSeg;             // x0, y0, x1, y1 (pixels, y down)

out vec3 vUVL;            // segment-local x, y and segment length

void main() {
    int id = gl_VertexID & 3;

    vec2 p0 = aSeg.xy;
    vec2 p1 = aSeg.zw;
    vec2 d = p1 - p0;
    float len = length(d);
    vec2 dir = len > 1e-5 ? d / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);

    float r = uExtent;
    float side = (id & 1) == 0 ? -1.0 : 1.0;
    bool at_end = id >= 2;

    vec2 p = (at_end ? p1 + dir * r : p0 - dir * r) + nrm * (side * r);
    vUVL = vec3(at_end ? len + r : -r, side * r, len);

    vec2 ndc = p / uViewport * 2.0 - 1.0;
    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
}
"""

LINE_FRAG = """
#version 330
uniform float uWidth;     // line width in pixels
uniform int uAA;

in vec3 vUVL;
out vec4 fragColor;

void main() {
    float x = vUVL.x;
    float dist = length(vec2(x - clamp(x, 0.0, vUVL.z), vUVL.y));
    float cov;
    if (uAA == 1) {
        // lines thinner than a pixel get a 1px footprint with reduced coverage
        float half_w = max(uWidth, 1.0) * 0.5;
        cov = clamp(half_w + 0.5 - dist, 0.0, 1.0) * min(uWidth, 1.0);
    } else {
        cov = dist <= max(uWidth, 1.0) * 0.5 ? 1.0 : 0.0;
    }
    if (cov <= 0.0) {
        discard;
    }
    fragColor = vec4(cov);
}
"""

# coverage (R) x color -> premultiplied RGBA, blended "over" into the line layer
LINE_RESOLVE_FRAG = """
#version 330
uniform sampler2D uCoverage;
uniform vec3 uColor;
in vec2 vUV;
out vec4 fragColor;
void main() {
    float a = texture(uCoverage, vUV).r;
    fragColor = vec4(uColor * a, a);
}
"""



# ==========================
# Helpers
# ==========================


def upsample(values: np.ndarray, factor: int, margin: int = 0) -> np.ndarray:
    """
    Band-limited (windowed sinc) upsampling so the beam follows a smooth curve
    between the samples instead of straight lines, like the analog output stage.
    The first `margin` samples are only context for the filter. Returns the
    points from input index `margin` up to the last input sample, inclusive:
    (len(values) - margin - 1) * factor + 1 points.
    """

    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if factor <= 1 or n < 2:
        return values[margin:].astype(np.float32)

    last = (n - 1) * factor + 1
    if resample_poly is not None:
        dense = resample_poly(values, factor, 1, padtype="line")[:last]
    else:
        dense = np.interp(np.arange(last) / factor, np.arange(n), values)
    return dense[margin * factor:].astype(np.float32)


# ==========================
# Renderer
# ==========================


class CRTRenderer:
    """
    Usage per frame:
        renderer.upload_ui(surface)
        renderer.upload_overlay(text_surface)           # optional, drawn on top
        if crt:
            renderer.begin_frame(dt)
            renderer.draw_beam(points, energy, color)   # any number of times
        if plain lines:
            renderer.begin_lines()
            renderer.draw_line(points, color)           # any number of times
        renderer.present(crt)
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height

        # tunables (set from the application)
        self.beam_sigma = 1.3       # px
        self.persistence = 0.025    # s, phosphor decay time constant
        self.intensity = 1.0
        self.exposure = 2.5
        self.bloom = True
        self.bloom_near = 0.6
        self.bloom_far = 0.35
        self.white_core = 0.6
        self.gamma = 2.2
        self.knee = 5.0
        self.vignette = 0.3
        self.ink = False            # light background theme

        self.decay = 0.0            # persistence multiplier of the current frame
        self.dt = 0.0

        self.beam_prog = ctx.program(vertex_shader=BEAM_VERT, fragment_shader=BEAM_FRAG)
        self.fade_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=FADE_FRAG)
        self.down_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=DOWNSAMPLE_FRAG)
        self.blur_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=BLUR_FRAG)
        self.comp_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=COMPOSITE_FRAG)

        self.fade_vao = ctx.vertex_array(self.fade_prog, [])
        self.down_vao = ctx.vertex_array(self.down_prog, [])
        self.blur_vao = ctx.vertex_array(self.blur_prog, [])
        self.comp_vao = ctx.vertex_array(self.comp_prog, [])

        self.seg_capacity = 0
        self.seg_buffer = None
        self.beam_vao = None
        self._ensure_capacity(16384)

        self.overlay = False        # upload_overlay() was called this frame

        # plain line renderer (non-CRT)
        self.line_width = 2.0       # px
        self.line_aa = True
        self.line_prog = ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)
        self.line_resolve_prog = ctx.program(
            vertex_shader=FULLSCREEN_VERT, fragment_shader=LINE_RESOLVE_FRAG
        )
        self.line_resolve_vao = ctx.vertex_array(self.line_resolve_prog, [])
        self.line_seg_capacity = 0
        self.line_seg_buffer = None
        self.line_vao = None
        self._ensure_line_capacity(16384)

        self._targets = []
        self._create_targets()
        self.line_layer = False     # begin_lines() was called this frame

        self.clear()

    # ---------- resources ----------

    def _target(self, size, components, dtype="f1", linear=False):
        tex = self.ctx.texture(size, components, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR) if linear else (moderngl.NEAREST, moderngl.NEAREST)
        self._targets.append(tex)
        return tex

    def _framebuffer(self, tex):
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        self._targets.append(fbo)
        return fbo

    def _create_targets(self):
        """Everything sized to the window: phosphor, bloom, UI layers, line layer."""
        size = (self.width, self.height)

        # phosphor: 32-bit float so tiny per-frame deposits with long persistence
        # are not lost to rounding (fall back to half floats if not renderable)
        self.hdr_tex = None
        for dtype in ("f4", "f2"):
            try:
                tex = self.ctx.texture(size, 4, dtype=dtype)
                fbo = self.ctx.framebuffer(color_attachments=[tex])
            except Exception:
                continue
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.hdr_tex, self.hdr_fbo = tex, fbo
            self._targets += [tex, fbo]
            break
        if self.hdr_tex is None:
            raise RuntimeError("no float render target available")

        self.bloom_levels = []
        for div in (2, 4):
            level_size = (max(1, self.width // div), max(1, self.height // div))
            pair = []
            for _ in range(2):
                tex = self._target(level_size, 4, "f2", linear=True)
                tex.repeat_x = tex.repeat_y = False
                pair.append((tex, self._framebuffer(tex)))
            self.bloom_levels.append(pair)

        self.ui_tex = self._target(size, 4)
        self.overlay_tex = self._target(size, 4)
        self.coverage_tex = self._target(size, 1)
        self.coverage_fbo = self._framebuffer(self.coverage_tex)
        self.line_tex = self._target(size, 4)
        self.line_fbo = self._framebuffer(self.line_tex)

    def resize(self, width: int, height: int):
        """Rebuild the window-sized render targets (the phosphor starts empty)."""
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        for obj in reversed(self._targets): # framebuffers before their textures
            obj.release()
        self._targets = []
        self.width, self.height = width, height
        self._create_targets()
        self.clear()

    def _ensure_capacity(self, segments: int):
        if segments <= self.seg_capacity:
            return
        capacity = max(segments, self.seg_capacity * 2)
        if self.beam_vao is not None:
            self.beam_vao.release()
            self.seg_buffer.release()
        self.seg_buffer = self.ctx.buffer(reserve=capacity * 5 * 4, dynamic=True)
        self.beam_vao = self.ctx.vertex_array(
            self.beam_prog, [(self.seg_buffer, "4f 1f /i", "aSeg", "aEnergy")]
        )
        self.seg_capacity = capacity

    def _ensure_line_capacity(self, segments: int):
        if segments <= self.line_seg_capacity:
            return
        capacity = max(segments, self.line_seg_capacity * 2)
        if self.line_vao is not None:
            self.line_vao.release()
            self.line_seg_buffer.release()
        self.line_seg_buffer = self.ctx.buffer(reserve=capacity * 4 * 4, dynamic=True)
        self.line_vao = self.ctx.vertex_array(
            self.line_prog, [(self.line_seg_buffer, "4f /i", "aSeg")]
        )
        self.line_seg_capacity = capacity

    def clear(self):
        """Erase the phosphor (view change etc.)."""
        self.hdr_fbo.clear(0.0, 0.0, 0.0, 0.0)
        for pair in self.bloom_levels:
            for _, fbo in pair:
                fbo.clear(0.0, 0.0, 0.0, 0.0)

    # ---------- UI ----------

    def _upload(self, tex, surface, alpha):
        import pygame

        masks = surface.get_masks()
        if (
            surface.get_bitsize() == 32
            and surface.get_pitch() == self.width * 4
            and masks[:3] == (0xFF0000, 0x00FF00, 0x0000FF)
            and (not alpha or masks[3] == 0xFF000000)
        ):
            # little-endian (A/X)RGB in memory = B, G, R, A/X bytes: no conversion
            tex.swizzle = "BGRA" if alpha else "BGR1"
            tex.write(surface.get_buffer().raw)
        else:
            tex.swizzle = "RGBA" if alpha else "RGB1"
            tex.write(pygame.image.tobytes(surface, "RGBA"))

    def upload_ui(self, surface):
        """Upload a pygame surface (same size as the window) as the 2D layer."""
        self._upload(self.ui_tex, surface, alpha=False)

    def upload_overlay(self, surface):
        """
        Upload a per-pixel alpha pygame surface (same size as the window) that the
        next present() lays over everything, the lines and the beam included.
        """
        self._upload(self.overlay_tex, surface, alpha=True)
        self.overlay = True

    # ---------- beam ----------

    def begin_frame(self, dt: float):
        """Let the phosphor decay for dt seconds."""
        self.dt = max(dt, 0.0)
        tau = max(self.persistence, 1e-4)
        self.decay = math.exp(-self.dt / tau)

        self.hdr_fbo.use()
        self.hdr_fbo.scissor = None
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ZERO, moderngl.SRC_COLOR
        self.fade_prog["uDecay"].value = self.decay
        self.fade_vao.render(moderngl.TRIANGLES, vertices=3)

    def sweep_energy(self) -> float:
        """
        Per-step energy for a trace that is redrawn as a whole every frame
        (triggered sweep). Scaled by (1 - decay) so the steady-state brightness
        does not depend on the frame rate or the persistence setting.
        """
        return self.intensity * (1.0 - self.decay)

    def stream_energy(self, count: int, rate: float, reference: float) -> np.ndarray:
        """
        Per-sample energy for a continuously running beam (XY mode) where only
        the samples that arrived since the last frame are drawn. Older samples
        of the chunk have already decayed for their age. A figure repeating every
        `reference` samples reaches roughly the brightness of a sweep trace.
        """
        tau = max(self.persistence, 1e-4)
        age = (count - 1 - np.arange(count, dtype=np.float64)) / rate
        scale = self.intensity * reference / (rate * tau)
        return (scale * np.exp(-age / tau)).astype(np.float32)

    def draw_beam(self, points, energy, color, scissor=None):
        """
        points: (N, 2) beam positions in window pixels (y down), one per time step.
        energy: scalar or (N - 1,) energy of each step.
        color:  (r, g, b) 0..1.
        scissor: optional (left, top, width, height) in window pixels; the beam is
                 cut off outside (like the edge of the tube face).
        """

        points = np.asarray(points, dtype=np.float32)
        n = len(points) - 1
        if n <= 0:
            return

        segments = np.empty((n, 5), dtype=np.float32)
        segments[:, 0:2] = points[:-1]
        segments[:, 2:4] = points[1:]
        segments[:, 4] = energy

        self._ensure_capacity(n)
        self.seg_buffer.write(segments.tobytes())

        if self.ink:
            color = tuple(1.0 - c for c in color) # absorbed light
        self.beam_prog["uViewport"].value = (self.width, self.height)
        self.beam_prog["uSigma"].value = self.beam_sigma
        self.beam_prog["uColor"].value = tuple(color)

        self.hdr_fbo.use()
        if scissor is not None:
            left, top, w, h = scissor
            self.hdr_fbo.scissor = (int(left), int(self.height - top - h), int(w), int(h))
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.beam_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        self.hdr_fbo.scissor = None

    # ---------- plain lines ----------

    def begin_lines(self):
        """
        Start a new line layer (lines are not kept between frames). The next
        present() lays it over the UI, under the CRT beam.
        """
        self.line_fbo.scissor = None
        self.line_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.line_layer = True

    def draw_line(self, points, color, scissor=None, width=None):
        """
        Polyline with self.line_aa into the line layer (begin_lines() first).
        points: (N, 2) window pixels (y down). color: (r, g, b) 0..1.
        scissor: optional (left, top, width, height) clip rectangle.
        width: pixels, default self.line_width.
        """

        points = np.asarray(points, dtype=np.float32)
        n = len(points) - 1
        if n <= 0:
            return

        segments = np.empty((n, 4), dtype=np.float32)
        segments[:, 0:2] = points[:-1]
        segments[:, 2:4] = points[1:]
        self._ensure_line_capacity(n)
        self.line_seg_buffer.write(segments.tobytes())

        width = max(float(self.line_width if width is None else width), 0.05)
        prog = self.line_prog
        prog["uViewport"].value = (self.width, self.height)
        prog["uWidth"].value = width
        prog["uAA"].value = 1 if self.line_aa else 0
        prog["uExtent"].value = max(width, 1.0) * 0.5 + 1.0

        rect = None
        if scissor is not None:
            left, top, w, h = scissor
            rect = (int(left), int(self.height - top - h), int(w), int(h))

        # 1) coverage of the whole polyline, merged with MAX
        self.coverage_fbo.use()
        self.coverage_fbo.scissor = None
        self.coverage_fbo.clear(0.0)
        self.coverage_fbo.scissor = rect
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.blend_equation = moderngl.MAX
        self.line_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.coverage_fbo.scissor = None

        # 2) tint and lay it over the earlier lines (premultiplied "over")
        self.line_fbo.use()
        self.line_fbo.scissor = rect
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.coverage_tex.use(0)
        self.line_resolve_prog["uCoverage"].value = 0
        self.line_resolve_prog["uColor"].value = tuple(color)
        self.line_resolve_vao.render(moderngl.TRIANGLES, vertices=3)
        self.line_fbo.scissor = None

    # ---------- output ----------

    def _bloom(self):
        self.ctx.disable(moderngl.BLEND)
        src = self.hdr_tex
        for (tex_a, fbo_a), (tex_b, fbo_b) in self.bloom_levels:
            fbo_a.use()
            src.use(0)
            self.down_prog["uSrc"].value = 0
            self.down_vao.render(moderngl.TRIANGLES, vertices=3)

            for direction, (tex_src, fbo_dst) in (((1.0, 0.0), (tex_a, fbo_b)), ((0.0, 1.0), (tex_b, fbo_a))):
                fbo_dst.use()
                tex_src.use(0)
                self.blur_prog["uSrc"].value = 0
                self.blur_prog["uDir"].value = direction
                self.blur_vao.render(moderngl.TRIANGLES, vertices=3)
            src = tex_a

    def present(self, crt: bool, target=None, *, ui_rect=None):
        """
        Composite the UI, the line layer (if begin_lines() was called this frame)
        and the beam (if crt) into `target` (default: the window).
        """

        if crt and self.bloom:
            self._bloom()

        target = target or self.ctx.screen
        if target is self.ctx.screen:
            # the window may have been resized since the context was created
            target.viewport = (0, 0, self.width, self.height)
        target.use()
        self.ctx.disable(moderngl.BLEND)

        self.ui_tex.use(0)
        self.hdr_tex.use(1)
        self.bloom_levels[0][0][0].use(2)
        self.bloom_levels[1][0][0].use(3)
        self.line_tex.use(4)
        self.overlay_tex.use(5)

        prog = self.comp_prog
        prog["uUI"].value = 0
        prog["uHDR"].value = 1
        prog["uBloomNear"].value = 2
        prog["uBloomFar"].value = 3
        prog["uLines"].value = 4
        prog["uCRT"].value = 1 if crt else 0
        prog["uLineLayer"].value = 1 if self.line_layer else 0
        self.line_layer = False
        prog["uOverlay"].value = 5
        prog["uUseOverlay"].value = 1 if self.overlay else 0
        self.overlay = False
        if ui_rect is None:
            prog["uUIRect"].value = (0, 0, 0, 0)
        else:
            x, y, width, height = ui_rect
            prog["uUIRect"].value = (x, y, x + width, y + height)
        prog["uInk"].value = 1 if self.ink else 0
        prog["uExposure"].value = self.exposure
        # glow is light; on a light background (ink) it would only smear the trace
        bloom = (0.15 if self.ink else 1.0) if self.bloom else 0.0
        prog["uBloomNearStr"].value = self.bloom_near * bloom
        prog["uBloomFarStr"].value = self.bloom_far * bloom
        prog["uWhite"].value = self.white_core
        # the gamma lift makes the gaussian tail glow - right for light, not for ink
        prog["uGamma"].value = 1.0 if self.ink else self.gamma
        prog["uKnee"].value = self.knee
        prog["uVignette"].value = self.vignette
        self.comp_vao.render(moderngl.TRIANGLES, vertices=3)
