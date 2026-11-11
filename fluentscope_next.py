import json
import math
import os
import sys
import threading

from window_support import configure_windowing, focus_window, ui_font

configure_windowing()

import numpy as np
import pygame
try:
    if sys.platform != "win32":
        raise ImportError
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio

from audio_devices import capture_devices, resolve_device, print_devices, open_capture, choose_device

# ==========================
# Audio Settings
# ==========================

RATE = 48000
CHANNELS = 2 # 2 = stereo; falls back to mono if the device has only one input channel
FORMAT = pyaudio.paInt16
CHUNK = 512

SCOPE_SAMPLE_COUNTS = (256, 512, 1024, 2048, 4096)
DEFAULT_SCOPE_SAMPLES = 1024

# ==========================
# Spectrum Settings
# ==========================

FFT_SIZE = 2 ** 11 # larger = finer frequency resolution, slower response

SPEC_F_MIN = 20.0
SPEC_F_MAX = RATE / 2

SPEC_DB_MIN = -100.0
SPEC_DB_MAX = 0.0

SPEC_FALL_DB_PER_SEC = 240.0 # how fast the spectrum line falls back after a peak

WATERFALL_ROWS_PER_SEC = 60.0 # scroll speed of the waterfall

# ==========================
# Trigger Settings
# ==========================

TRIGGER_MODES = ["Peak", "FFT", "Hybrid", "Corr", "Legacy", "Diff", "Off"]
# "Peak":   multi-pass max trigger
# "FFT":    autocorrelation + phase lock (Furnace chanOsc style)
# "Hybrid": Peak trigger; when several peaks compete (FM etc.), the FFT phase picks
#           the one that keeps the same position within the period as last frame
# "Corr":   corrscope-style correlation trigger: match against a buffer of past
#           frames + rising-edge finder (https://github.com/corrscope/corrscope)
# "Legacy": old FluentScope idea, simplified: the position that best matches the
#           previous frame (cross-correlation over the whole search region)
# "Diff":   Legacy variant: the position with the smallest plain difference from the
#           previous frame (sum of |a-b|) instead of the correlation peak
# "Off":    no trigger (free-run): always shows the newest samples
TRIGGER_MODE = "Peak"

# Where the fundamental is placed at the screen center, in periods.
# 0.75 = rising zero-crossing of the fundamental (close to the Peak trigger look),
# 0.0  = Furnace's default (trough of the fundamental).
FFT_TRIGGER_ALIGN = 0.75

FFT_TRIGGER_SILENCE = 0.001 # below this amplitude the FFT trigger free-runs

PEAK_TRIGGER_TOLERANCE = 0.04 # peaks within this ratio of the maximum count as candidates

# Hybrid: how fast the remembered in-period position follows the chosen peak (0..1).
# Lower = stickier (resists switching to another peak), higher = follows timbre changes faster.
HYBRID_TRACKING = 0.2

# Corr trigger (names and defaults follow corrscope's CorrelationTriggerConfig / template)
CORR_EDGE_STRENGTH = 1.0      # weight of the rising-edge finder
CORR_BUFFER_STRENGTH = 1.0    # weight of the correlation buffer (past frames)
CORR_RESPONSIVENESS = 0.5     # how fast the buffer follows new frames (0..1)
CORR_BUFFER_FALLOFF = 0.5     # gaussian window std of the buffer, in periods
CORR_TRIGGER_DIAMETER = 0.5   # search width, fraction of the display length
CORR_TRIGGER_RADIUS_PERIODS = 1.5 # max distance from the nominal position, in periods
CORR_SLOPE_WIDTH = 0.25       # edge finder gaussian std, in periods
CORR_RESET_BELOW = 0.3        # reset the buffer if it matches worse than this
CORR_MAX_FREQ = 4000.0        # highest pitch the period estimator accepts
CORR_RECALC_SEMITONES = 1.0   # rebuild the edge finder when pitch moves this much

# Legacy trigger
LEGACY_WINDOW_LEN = 1024      # samples compared against the previous frame
LEGACY_SEARCH_LEN = 1024      # how far back (samples) the new position may be searched
# bias toward newer positions, in units of normalized correlation (1.0 = perfect match).
# old fluentscope's CORR_WEIGHT_FACTOR = 5.0 was roughly 0.07 on this scale
LEGACY_POSITION_WEIGHT = 0.07
# Shift the displayed window so its center sits on the nearest rising zero crossing
# (of the local mean). Display only: matching and the reference fragment still use
# the matched position, so locking behaves exactly as without it.
LEGACY_SUBSAMPLE = True # sub-sample position tracking (stops the slow one-way drift)

LEGACY_ZERO_CROSS = False
LEGACY_ZERO_CROSS_RADIUS = 512 # max shift in samples; no crossing within it = no shift
# Hysteresis: keep using the crossing at the same offset from the matched position as last
# frame (if one is within this many samples of it), instead of switching to whichever
# crossing is nearest. 0 = off (always nearest).
LEGACY_ZERO_CROSS_HYSTERESIS = 16

# Diff trigger (Legacy with minimum difference instead of maximum correlation).
# Window / search length, sub-sample tracking and zero-cross snap use the LEGACY_ settings.
DIFF_SQUARED = False          # False: sum of |a-b| (SAD), True: sum of (a-b)^2 (SSD)
# bias toward newer positions, in units of normalized difference (1.0 = reference's own size)
DIFF_POSITION_WEIGHT = 0.07

# The FFT trigger needs up to 3x the largest Scope time span.
RING_SIZE = max(3 * SCOPE_SAMPLE_COUNTS[-1], FFT_SIZE)

# ==========================
# Stereo Settings
# ==========================

STEREO_MODES = ["Mix", "Overlay", "Split"]
STEREO_MODE = "Mix" # "Mix": (L+R)/2, "Overlay": L and R in different colors, "Split": L on top, R on bottom

# True: trigger once on the L+R mix and show both channels from the same position
#       (keeps the timing / phase relation between L and R visible)
# False: trigger each channel independently in Overlay / Split
STEREO_SHARED_TRIGGER = True

# ==========================
# Lissajous Settings
# ==========================

LISSAJOUS_MODES = ["L/R", "Auto"]
# "L/R":  X = left, Y = right (stereo XY plot)
# "Auto": autocorrelation Lissajous after Bespoke Synth's Lissajous module
#         (Source/Lissajous.cpp): X = s[i], Y = s[i + delay] on the L+R mix.
#         A mono input always uses this mode, like Bespoke does with one channel.
LISSAJOUS_MODE = "L/R"

LISSAJOUS_SWAP = False            # [W] exchange the X and Y axes
LISSAJOUS_INVERTS = ["Off", "X", "Y", "XY"] # [I] flip the polarity of an axis
LISSAJOUS_INVERT = "Off"

LISSAJOUS_POINTS = 2048           # newest samples drawn per frame
LISSAJOUS_HISTORY = 4096          # maximum history for the continuous CRT trace
LISSAJOUS_AUTOCORR_DELAY = 90     # samples; Bespoke's autocorrelationDelay
LISSAJOUS_DELAY_MIN = 1
LISSAJOUS_DELAY_MAX = 1024
LISSAJOUS_GAIN = 0.9              # 1.0 = full scale touches the square's edge
LISSAJOUS_TOP = 64                # leaves room for the info text

# ==========================
# Level Meter Settings
# ==========================

SHOW_LEVEL_METER = True # L/R meter on the right edge (reserves METER_PANEL_WIDTH pixels)
METER_PANEL_WIDTH = 70

METER_DB_MIN = -60.0
METER_WINDOW = 1024 # samples used for RMS / peak (~21ms at 48kHz)
METER_FALL_DB_PER_SEC = 30.0
METER_PEAK_HOLD_SEC = 1.0
METER_WARN_DB = -6.0 # bar turns METER_WARN_COLOR above this
METER_CLIP_LEVEL = 0.999 # linear sample value treated as clipping
METER_CLIP_HOLD_SEC = 1.5

METER_WARN_COLOR = (255, 200, 0)
METER_CLIP_COLOR = (255, 50, 50)

# ==========================
# Display Settings
# ==========================

WIDTH = 1000
HEIGHT = 600
SCOPE_GAIN = 0.9
SCOPE_AGC_MAX_BOOST = 32.0
SCOPE_AGC_SILENCE = 0.0001
SCOPE_AGC_DECAY_TIMES = {"High": 0.12, "Med": 0.35, "Low": 1.0} # seconds
SCOPE_AGC_MODES = ("Off", *SCOPE_AGC_DECAY_TIMES)
SCOPE_AGC_HOLD_TIME = 0.5 # seconds

FPS = 180 # no effect when VSYNC is enabled
VSYNC = False # if True, the display is synchronized with the monitor's refresh rate, otherwise it runs at a fixed FPS

LINE_IS_STEP = True # if True, the waveform is drawn as a step line, otherwise as a smooth line

SPEC_OVERLAY_RATIO = 0.4 # height ratio of the spectrum overlay on the scope view

# ==========================
# CRT Beam Settings
# ==========================
# Shader-based analog oscilloscope look (crt_renderer.py, needs `pip install moderngl`).
# The window is an SDL OpenGL window; the 2D UI is still drawn with pygame onto an
# offscreen surface and composited under the beam.

USE_OPENGL = True # False = plain pygame window (no CRT / GL line option)

# Trace renderer ([C] cycles, needs OpenGL):
#   "CRT"     - analog beam (settings below)
#   "GL Line" - plain lines drawn with OpenGL: free width, optional anti-aliasing
#   "Pygame"  - plain 2D lines drawn by pygame
TRACE_RENDERERS = ["CRT", "GL Line", "Pygame"]
TRACE_RENDERER = "CRT"

LINE_WIDTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0] # GL line width in pixels ([N])
LINE_WIDTH = 2.0
LINE_AA = True # GL line anti-aliasing ([O])

CRT_BEAM_WIDTHS = [0.6, 0.9, 1.3, 1.8, 2.5, 3.5] # beam spot sigma in pixels ([F] focus)
CRT_BEAM_WIDTH = 1.3

CRT_PERSISTENCES = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0] # phosphor decay, seconds ([G])
CRT_PERSISTENCE = 0.025

# Band-limited (sinc) upsampling of the beam path, so the beam moves along a smooth
# curve between samples like the analog output stage does. 1 = straight lines.
CRT_UPSAMPLE = 4
CRT_UPSAMPLE_MARGIN = 16 # context samples for the filter at chunk starts (Lissajous)

CRT_BLOOM = True        # [B] glow around the trace
CRT_BLOOM_NEAR = 0.6    # half resolution blur
CRT_BLOOM_FAR = 0.35    # quarter resolution blur (wide haze)
CRT_VIGNETTE = 0.3      # darken the tube corners (dark themes only)

# Brightness curve. The beam light is accumulated linearly, then displayed as
#   v = (light * EXPOSURE) ^ (1 / GAMMA)
#   brightness = v / (1 + v^KNEE)^(1/KNEE)   -> follows the gamma curve, saturates at 1
CRT_EXPOSURE = 2.5      # overall brightness of the trace ([E] / [Shift+E])
CRT_EXPOSURE_STEP = 1.25
# curve exponent is 1/GAMMA ([V] / [Shift+V]). Higher = dim parts (fast edges, bloom
# halo) come up brighter; 1.0 = linear, the bloom stays faint.
CRT_GAMMAS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0]
CRT_GAMMA = 2.2
CRT_SATURATION_KNEE = 5.0 # higher = harder clip at full brightness, lower = softer roll-off
CRT_WHITE_CORE = 0.6    # how much an overdriven trace bleaches toward white

# Lissajous brightness: a figure that repeats every this many samples is about as
# bright as the scope trace. Faster-repeating figures get brighter, like a real scope.
CRT_LISSAJOUS_REF = 1024

# ==========================
# Color Themes
# ==========================
# colormap: list of (position 0..1, (r, g, b)) used for the waterfall

THEMES = [
    dict(
        name="Phosphor",
        bg=(5, 5, 5), grid=(40, 40, 40), center=(70, 70, 70),
        trace=(0, 255, 0), spectrum=(255, 200, 0), text=(200, 200, 200),
        trace_l=(0, 255, 0), trace_r=(255, 70, 160),
        colormap=[(0.0, (0, 0, 0)), (0.3, (0, 50, 10)), (0.6, (0, 170, 50)),
                  (0.85, (150, 255, 90)), (1.0, (240, 255, 220))],
    ),
    dict(
        name="Amber",
        bg=(8, 5, 0), grid=(48, 32, 10), center=(85, 58, 15),
        trace=(255, 176, 0), spectrum=(255, 90, 40), text=(230, 190, 120),
        trace_l=(255, 176, 0), trace_r=(120, 200, 255),
        colormap=[(0.0, (0, 0, 0)), (0.3, (60, 20, 0)), (0.6, (200, 90, 0)),
                  (0.85, (255, 190, 40)), (1.0, (255, 250, 210))],
    ),
    dict(
        name="Ice",
        bg=(2, 6, 14), grid=(20, 36, 56), center=(40, 64, 96),
        trace=(80, 220, 255), spectrum=(255, 120, 220), text=(170, 210, 240),
        trace_l=(80, 220, 255), trace_r=(255, 150, 80),
        colormap=[(0.0, (0, 0, 8)), (0.3, (10, 30, 100)), (0.6, (20, 120, 220)),
                  (0.85, (100, 230, 255)), (1.0, (240, 255, 255))],
    ),
    dict(
        name="Inferno",
        bg=(0, 0, 4), grid=(40, 20, 45), center=(70, 35, 70),
        trace=(250, 150, 30), spectrum=(190, 90, 255), text=(230, 200, 170),
        trace_l=(250, 150, 30), trace_r=(90, 200, 255),
        colormap=[(0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
                  (0.75, (249, 142, 9)), (1.0, (252, 255, 164))],
    ),
    dict(
        name="Viridis",
        bg=(8, 8, 18), grid=(35, 40, 60), center=(60, 70, 95),
        trace=(94, 201, 98), spectrum=(253, 231, 37), text=(190, 210, 200),
        trace_l=(94, 201, 98), trace_r=(190, 100, 230),
        colormap=[(0.0, (20, 0, 30)), (0.2, (68, 1, 84)), (0.45, (59, 82, 139)),
                  (0.65, (33, 145, 140)), (0.85, (94, 201, 98)), (1.0, (253, 231, 37))],
    ),
    dict(
        name="Mono",
        bg=(0, 0, 0), grid=(35, 35, 35), center=(65, 65, 65),
        trace=(235, 235, 235), spectrum=(140, 140, 140), text=(200, 200, 200),
        trace_l=(235, 235, 235), trace_r=(130, 130, 130),
        colormap=[(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    ),
    dict(
        name="Paper",
        bg=(242, 240, 232), grid=(205, 202, 192), center=(160, 158, 150),
        trace=(25, 25, 25), spectrum=(200, 40, 40), text=(50, 50, 50),
        trace_l=(20, 80, 200), trace_r=(220, 110, 0),
        colormap=[(0.0, (242, 240, 232)), (0.35, (170, 200, 230)), (0.65, (60, 110, 200)),
                  (0.85, (25, 35, 115)), (1.0, (0, 0, 0))],
    ),
]


def build_colormap_lut(stops):
    positions = np.array([p for p, _ in stops], dtype=np.float32)
    colors = np.array([c for _, c in stops], dtype=np.float32)
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.stack(
        [np.interp(t, positions, colors[:, i]) for i in range(3)],
        axis=1,
    )
    return np.rint(lut).astype(np.uint8)


def blend(color_a, color_b, ratio):
    return tuple(int(a * ratio + b * (1 - ratio)) for a, b in zip(color_a, color_b))


for _theme in THEMES:
    _theme["lut"] = build_colormap_lut(_theme["colormap"])
    _theme["fill"] = blend(_theme["spectrum"], _theme["bg"], 0.25)

# ==========================
# Ring Buffer
# ==========================

ring = np.zeros((2, RING_SIZE), dtype=np.float32) # [L, R]
write_pos = 0
samples_written = 0 # total samples received (absolute time base for the Corr trigger)
input_channels = 1 # actual channel count of the opened stream (set at stream open)
lock = threading.Lock()

# ==========================
# Trigger
# ==========================


def trigger_waveform(waveform: np.ndarray):
    start, end = peak_trigger_range(waveform)
    return waveform[start:end]


def peak_trigger_range(waveform: np.ndarray):
    """Multi-pass max trigger. Returns (start, end) indices into `waveform`."""

    sample_count = len(waveform)

    if sample_count == 0:
        return 0, 0

    length = sample_count // 2
    half = length // 2

    trigger = sample_count - half

    search_start = max(0, trigger - half)

    #
    # Pass1
    #

    maximum = np.max(waveform[search_start:trigger + 1])

    #
    # Pass2
    #

    tolerance = PEAK_TRIGGER_TOLERANCE

    best = trigger

    for i in range(trigger, search_start - 1, -1):
        if waveform[i] >= maximum * (1 - tolerance):
            best = i
            break

    trigger = best

    #
    # Pass3
    #

    average = waveform.mean()

    zero_end = max(trigger - half, 0)

    for i in range(trigger, zero_end - 1, -1):

        if waveform[i] <= average:
            trigger = i
            break

    start = max(trigger - half, 0)
    end = min(trigger + half, sample_count)

    return start, end


def fft_trigger_range(samples: np.ndarray, display_len: int):
    """
    Pitch-locked trigger after Furnace's chanOsc (src/gui/chanOsc.cpp):
      1. FFT of the windowed signal (at least 2 * display_len samples)
      2. inverse FFT of the power spectrum = auto-correlation
      3. period = highest auto-correlation peak after its lowest point
      4. DFT of the fundamental over ONE period
      5. lock the display start to that phase

    Returns (start index into `samples`, period_in_samples or None).
    """

    # Keep the default pitch analysis range when zooming in on the waveform.
    analysis_len = max(display_len, DEFAULT_SCOPE_SAMPLES)
    n = analysis_len * 2
    end = len(samples)

    free_run = end - display_len

    if end < analysis_len * 3:
        return free_run, None

    segment = samples[end - n:].astype(np.float64)

    if np.max(np.abs(segment)) <= FFT_TRIGGER_SILENCE:
        return free_run, None

    #
    # 1. windowed FFT (same raised-cosine window as Furnace)
    #

    j = np.arange(n, dtype=np.float64)
    window = 0.55 - 0.45 * np.cos(np.pi * j / (n >> 1))
    spectrum = np.fft.rfft(segment * window)

    #
    # 2. auto-correlation; drop DC and the lowest bin
    #

    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    power[:2] = 0.0
    corr = np.fft.irfft(power, n)

    half = n >> 1
    corr[:half] *= 1.0 - j[:half] / (n << 1)

    #
    # 3. period size
    #

    quarter = n >> 2
    bottom = 3 + int(np.argmin(corr[3:quarter + 1]))

    search = corr[bottom + 1:half]
    if len(search) == 0:
        return free_run, None

    top = bottom + 1 + int(np.argmax(search))
    if corr[top] <= 0.0:
        return free_run, None

    # sub-sample refinement (parabolic) - steadier than Furnace's integer lag
    wave_len = float(top)
    if 0 < top < half - 1:
        a, b, c = corr[top - 1], corr[top], corr[top + 1]
        denominator = a - 2.0 * b + c
        if denominator < 0.0:
            wave_len += 0.5 * (a - c) / denominator

    if wave_len < 2.0 or wave_len > analysis_len:
        return free_run, None

    #
    # 4. DFT of the fundamental over one period
    #

    period = int(math.ceil(wave_len))
    period_start = end - display_len // 2 - period - 1
    k = np.arange(period, dtype=np.float64)
    one = samples[period_start:period_start + period]
    two = k * (-2.0 * np.pi) / wave_len
    re = np.dot(one, np.cos(two))
    im = np.dot(one, np.sin(two))

    #
    # 5. phase lock
    #

    phase = 0.5 + math.atan2(im, re) / (2.0 * math.pi)
    shift = (phase + FFT_TRIGGER_ALIGN) % 1.0

    start = int(round(end - display_len - shift * wave_len))
    start = min(max(start, 0), end - display_len)

    return start, wave_len


def fft_trigger_waveform(samples: np.ndarray, display_len: int):
    start, wave_len = fft_trigger_range(samples, display_len)
    return samples[start:start + display_len], wave_len


def waveform_y(amplitude, top, height, gain):
    return top + height / 2 - amplitude * (gain * height * 0.45)


def update_scope_peak_hold(peak, held_peak, remaining, dt):
    if peak >= max(held_peak, SCOPE_AGC_SILENCE):
        return peak, SCOPE_AGC_HOLD_TIME
    remaining = max(0.0, remaining - dt)
    return (held_peak if remaining > 0.0 else peak), remaining


def update_scope_gain(peak, gain, dt, decay_time):
    target = SCOPE_GAIN
    if peak >= SCOPE_AGC_SILENCE:
        target *= min(1.0 / peak, SCOPE_AGC_MAX_BOOST)
    # Reduce gain immediately for louder signals, but ease into a quieter signal.
    if target <= gain:
        return target
    return gain + (target - gain) * -math.expm1(-dt / decay_time)


def build_waveform_points(
    waveform: np.ndarray,
    width: int,
    height: int,
    gain: float,
    line_is_step: bool,
    top: int = 0,
    round_pixels: bool = True,
):

    waveform_len = len(waveform)
    if waveform_len <= 1:
        return None

    sample_count = min(waveform_len, width)
    sample_indices = np.linspace(0, waveform_len - 1, sample_count, dtype=np.int32)
    sampled_waveform = waveform[sample_indices]

    x = np.linspace(0, width - 1, sample_count, dtype=np.float32)
    y = waveform_y(sampled_waveform, top, height, gain)

    if line_is_step:
        points = np.empty((sample_count * 2, 2), dtype=np.float32)
        points[0::2, 0] = x
        points[0::2, 1] = y
        points[1::2, 0] = x
        points[1::2, 1] = np.roll(y, -1)
        points[-1, 1] = y[-1] # no wrap-around stroke back to the first sample
    else:
        points = np.column_stack((x, y))

    if not round_pixels:
        return points.astype(np.float32, copy=False)
    return np.rint(points).astype(np.int32, copy=False)


def peak_trigger_candidates(waveform: np.ndarray):
    """
    Every trigger point the Peak trigger could have chosen: one per run of samples
    within PEAK_TRIGGER_TOLERANCE of the maximum, each moved back to its
    preceding mean crossing (Pass3). The last one equals peak_trigger_range().
    Returns (candidate trigger indices ascending, half display length).
    """

    sample_count = len(waveform)
    length = sample_count // 2
    half = length // 2
    trigger = sample_count - half
    search_start = max(0, trigger - half)

    window = waveform[search_start:trigger + 1]
    maximum = np.max(window)
    above = window >= maximum * (1 - PEAK_TRIGGER_TOLERANCE)

    # last index of every run above the threshold (Pass2 picks the latest one)
    run_ends = np.flatnonzero(above & ~np.append(above[1:], False)) + search_start

    average = waveform.mean()
    candidates = []
    for peak in run_ends:
        zero_end = max(peak - half, 0)
        below = np.flatnonzero(waveform[zero_end:peak + 1] <= average)
        candidates.append(int(zero_end + below[-1]) if len(below) else int(peak))

    return candidates, half


class HybridTrigger:
    """
    Peak trigger for precision, FFT phase for disambiguation.

    The FFT trigger gives a phase reference that moves with the waveform's period.
    Each Peak candidate gets a position within the period relative to that
    reference; we keep choosing the candidate whose position matches the one
    chosen last frame, so competing near-equal peaks no longer flip.
    """

    def __init__(self):
        self.rel_phase = None # remembered in-period position (0..1)

    def find(self, samples: np.ndarray, display_len: int):
        offset = len(samples) - 2 * display_len
        waveform = samples[offset:]

        candidates, half = peak_trigger_candidates(waveform)
        fft_start, wave_len = fft_trigger_range(samples, display_len)

        if len(candidates) <= 1 or wave_len is None:
            # unambiguous (or no pitch): plain Peak trigger
            start, end = peak_trigger_range(waveform)
            if wave_len is not None and candidates:
                self.rel_phase = self._phase(candidates[-1] - half + offset, fft_start, wave_len)
            return offset + start, offset + end, wave_len

        phases = [self._phase(t - half + offset, fft_start, wave_len) for t in candidates]

        if self.rel_phase is None:
            choice = len(candidates) - 1 # same as Peak on the first frame
        else:
            distances = [abs(self._wrap(p - self.rel_phase)) for p in phases]
            best = min(distances)
            # among equivalent candidates (one period apart) prefer the latest, like Peak
            choice = max(i for i, d in enumerate(distances) if d <= best + 0.02)

        chosen = phases[choice]
        if self.rel_phase is None:
            self.rel_phase = chosen
        else:
            self.rel_phase = (self.rel_phase + HYBRID_TRACKING * self._wrap(chosen - self.rel_phase)) % 1.0

        trigger = candidates[choice]
        start = max(trigger - half, 0)
        end = min(trigger + half, len(waveform))
        return offset + start, offset + end, wave_len

    @staticmethod
    def _wrap(x):
        return (x + 0.5) % 1.0 - 0.5

    @staticmethod
    def _phase(start, reference, wave_len):
        return ((start - reference) / wave_len) % 1.0


hybrid_triggers = {} # one state per signal (mix / left / right)


# ==========================
# Corr trigger (corrscope port)
# ==========================

CORR_MIN_AMPLITUDE = 0.01


def corr_get_period(data: np.ndarray, rate: float, max_freq: float) -> int:
    """corrscope trigger_util.get_period: autocorrelation peak after the first zero crossing."""

    n = len(data)
    if np.max(np.abs(data)) < CORR_MIN_AMPLITUDE:
        return 0

    # linear (zero-padded) autocorrelation, non-negative lags
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(data, size)
    corr = np.fft.irfft(spectrum.real ** 2 + spectrum.imag ** 2, size)[:n]

    min_period = int(round(rate / max_freq))
    negative = np.flatnonzero(corr < 0)
    if len(negative) == 0:
        return 0
    min_x = max(min_period, int(negative[0]))
    if min_x >= n:
        return 0

    peak_x = min_x + int(np.argmax(corr[min_x:]))

    # long periods: compensate the linear autocorrelation's decay
    if peak_x > 0.1 * n:
        divisor = np.maximum(1.0 - 0.9 * np.arange(n) / n, 0.5)
        corr = corr / divisor
        peak_x = min_x + int(np.argmax(corr[min_x:]))

    return peak_x


def corr_normalize(data: np.ndarray):
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    data /= max(peak, CORR_MIN_AMPLITUDE)


def corr_gaussian(n: int, std: float):
    if std <= 0:
        return np.zeros(n, dtype=np.float64)
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    return np.exp(-0.5 * (x / std) ** 2)


def correlate_valid(data: np.ndarray, kernel: np.ndarray):
    """out[i] = sum(data[i:i+K] * kernel), i = 0 .. len(data)-K (FFT based)."""
    n, k = len(data), len(kernel)
    size = 1 << int(math.ceil(math.log2(n + k)))
    out = np.fft.irfft(np.fft.rfft(data, size) * np.conj(np.fft.rfft(kernel, size)), size)
    return out[:n - k + 1]


class CorrelationTrigger:
    """
    Port of corrscope's CorrelationTrigger (without the pitch-tracking buffer rescale).

    Every frame, candidate positions within the search window are scored by the
    correlation of the data with (rising-edge finder + buffer of previous frames).
    The buffer is a gaussian-windowed, normalized running blend of the frames
    that were shown, so the display keeps matching what was shown before.
    """

    def __init__(self, display_len: int):
        self.a = display_len // 2
        self.b = display_len - self.a
        self.kernel_size = display_len
        self.diameter = int(display_len * CORR_TRIGGER_DIAMETER)

        self.buffer = np.zeros(self.kernel_size, dtype=np.float64)
        self.prev_window = np.zeros(self.kernel_size, dtype=np.float64)
        self.prev_period = None
        self.slope_finder = None
        self.prev_trigger_abs = None

    def _calc_slope_finder(self, period):
        n = self.kernel_size
        slope_width = float(np.clip(CORR_SLOPE_WIDTH * period, 1.0, self.a / 3))
        strength = CORR_EDGE_STRENGTH * 2
        finder = np.empty(n, dtype=np.float64)
        finder[:self.a] = -strength / 2
        finder[self.a:] = strength / 2
        return finder * corr_gaussian(n, slope_width)

    def _window_invalid(self, period):
        if self.prev_period is None or self.slope_finder is None:
            return True
        if period == 0 or self.prev_period == 0:
            return period != self.prev_period
        return abs(12 * math.log2(period / self.prev_period)) > CORR_RECALC_SEMITONES

    def find(self, samples: np.ndarray, end_abs=None):
        """Returns (start, end, period or None) into `samples`."""

        a, b, n, diameter = self.a, self.b, self.kernel_size, self.diameter
        end = len(samples)

        # latest search window that still has B samples after it
        trigger_begin = end - b - diameter
        data = samples[trigger_begin - a:end].astype(np.float64) # A + diameter + B

        period = corr_get_period(data - data.mean(), RATE, CORR_MAX_FREQ)

        if self._window_invalid(period):
            self.slope_finder = self._calc_slope_finder(period)
            self.prev_period = period

        corr_enabled = CORR_BUFFER_STRENGTH != 0 and CORR_RESPONSIVENESS != 0
        corr_quality = None

        if corr_enabled:
            corr_quality = correlate_valid(data, self.buffer)

            if CORR_RESET_BELOW > 0:
                peak_idx = int(np.argmax(corr_quality))
                peak_quality = corr_quality[peak_idx]
                data_slice = data[peak_idx:peak_idx + n].copy()
                data_slice -= data_slice.mean()
                corr_normalize(data_slice)
                data_slice *= self.prev_window
                self_quality = float(np.dot(data_slice, data_slice))
                if peak_quality / (self_quality + 0.001) < CORR_RESET_BELOW:
                    self.buffer[:] = 0.0
                    corr_enabled = False

        kernel = self.slope_finder
        if corr_enabled:
            kernel = kernel + self.buffer * CORR_BUFFER_STRENGTH
        corr = correlate_valid(data, kernel)

        if corr_enabled:
            peaks = corr_quality * CORR_BUFFER_STRENGTH
        else:
            peaks = np.zeros_like(corr)
        if CORR_EDGE_STRENGTH:
            edge_score = np.cumsum(data[a - 1:len(data) - b])
            peaks = peaks + edge_score * -CORR_EDGE_STRENGTH

        radius = int(round(period * CORR_TRIGGER_RADIUS_PERIODS)) if period else None
        offset = self._find_peak(corr, peaks, radius)
        trigger = trigger_begin + offset

        # never move backwards in absolute time
        if end_abs is not None:
            trigger_abs = end_abs - end + trigger
            if self.prev_trigger_abs is not None and trigger_abs < self.prev_trigger_abs:
                trigger = min(trigger + (self.prev_trigger_abs - trigger_abs), end - b)
                trigger_abs = end_abs - end + trigger
            self.prev_trigger_abs = trigger_abs

        self._update_buffer(samples[trigger - a:trigger + b].astype(np.float64), period)

        return trigger - a, trigger + b, (float(period) if period else None)

    @staticmethod
    def _find_peak(corr, peaks, radius):
        mid = len(corr) // 2
        begin = 0
        if radius is not None:
            begin = max(mid - radius, 0)
            corr = corr[begin:mid + radius + 1]
            peaks = peaks[begin:mid + radius + 1]

        if len(corr) < 3:
            return mid

        # only local maxima of `peaks` (endpoints excluded) can win
        local_max = np.zeros(len(peaks), dtype=bool)
        local_max[1:-1] = (peaks[1:-1] >= peaks[:-2]) & (peaks[1:-1] >= peaks[2:])
        if not local_max.any():
            return mid

        masked = np.where(local_max, corr, -np.inf)
        return begin + int(np.argmax(masked))

    def _update_buffer(self, aligned, period):
        if not (CORR_BUFFER_STRENGTH and CORR_RESPONSIVENESS):
            return
        aligned = aligned - aligned.mean()
        corr_normalize(aligned)
        window = corr_gaussian(self.kernel_size, period * CORR_BUFFER_FALLOFF)
        aligned *= window
        self.prev_window = window
        corr_normalize(self.buffer)
        self.buffer += CORR_RESPONSIVENESS * (aligned - self.buffer)


corr_triggers = {} # one state per signal (mix / left / right)


# ==========================
# Legacy trigger (simplified from old fluentscope.py ADVANCED_TRIGGERING)
# ==========================


class LegacyCorrTrigger:
    """
    Simplified version of the old FluentScope "ADVANCED_TRIGGERING".

    The original picked candidate peaks, correlated a window around each with
    the previous frame's fragment, then shifted by the correlation lag.
    Here every position in the search region is compared at once: the window
    that best matches the previous frame's fragment (cross-correlation via FFT)
    becomes the new display center, and the shown fragment becomes the next
    reference. Unlike Corr, the reference is just the last frame (no blending,
    no edge finder, no period limit).
    """

    def __init__(self, display_len: int):
        self.display_len = display_len
        self.window = LEGACY_WINDOW_LEN
        self.ref = None
        self.cross_offset = None # display center - matched center, last frame
        self.residual = 0.0 # true (sub-sample) position - integer position, last frame
        self.position_weight = LEGACY_POSITION_WEIGHT

    def _match(self, data: np.ndarray) -> np.ndarray:
        """Normalized cross-correlation of every window in `data` with self.ref."""
        corr = correlate_valid(data, self.ref)
        return corr / max(float(np.dot(self.ref, self.ref)), 1e-9)

    def find(self, samples: np.ndarray, end_abs=None):
        d = self.display_len
        w = self.window
        end = len(samples)

        # newest LEGACY_SEARCH_LEN centers whose window and display still fit
        last_center = end - max(d - d // 2, w - w // 2)
        first_center = max(last_center - LEGACY_SEARCH_LEN, max(d // 2, w // 2))

        data = samples[first_center - w // 2:last_center + (w - w // 2)].astype(np.float64)

        if self.ref is None or np.max(np.abs(data)) < FFT_TRIGGER_SILENCE:
            # first frame / silence: newest position, take it as reference
            center = last_center
        else:
            # corr[i]: how well the window centered at first_center + i matches the
            # previous fragment (higher = better, 1.0 = perfect)
            corr = self._match(data)
            score = corr + np.linspace(0.0, 1.0, len(corr)) * self.position_weight
            i = int(np.argmax(score))
            center = first_center + i

            if LEGACY_SUBSAMPLE:
                # Positions are whole samples but periods are not (330Hz = 145.45 samples),
                # so rounding every frame drifts the same way each time. Track the true
                # position with sub-sample precision and carry the fraction forward.
                delta = 0.0
                if 0 < i < len(corr) - 1:
                    a, b, c = corr[i - 1], corr[i], corr[i + 1]
                    denominator = a - 2.0 * b + c
                    if denominator < 0.0:
                        delta = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
                true_center = center + delta + self.residual
                center = int(round(true_center))
                center = min(max(center, first_center), last_center)
                self.residual = float(np.clip(true_center - center, -0.5, 0.5))

        self.ref = samples[center - w // 2:center + (w - w // 2)].astype(np.float64)
        ref_mean = float(self.ref.mean())
        self.ref = self.ref - ref_mean

        display_center = center
        if LEGACY_ZERO_CROSS:
            display_center = self._snap_rising_cross(
                samples, center, ref_mean, d // 2, end - (d - d // 2)
            )
            self.cross_offset = display_center - center

        return display_center - d // 2, display_center + (d - d // 2), None

    def _snap_rising_cross(self, samples, center, level, min_center, max_center):
        """Rising crossing (samples[i-1] < level <= samples[i]) to center the display on."""
        lo = max(center - LEGACY_ZERO_CROSS_RADIUS, min_center, 1)
        hi = min(center + LEGACY_ZERO_CROSS_RADIUS, max_center)
        if hi < lo:
            return center
        seg = samples[lo - 1:hi + 1] - level
        rising = np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0)) + lo
        if len(rising) == 0:
            return center

        # hysteresis: the crossing at the same offset from the matched position as last frame
        prev = self.cross_offset
        if LEGACY_ZERO_CROSS_HYSTERESIS > 0 and prev is not None:
            target = center + prev
            i = int(np.argmin(np.abs(rising - target)))
            if abs(int(rising[i]) - target) <= LEGACY_ZERO_CROSS_HYSTERESIS:
                return int(rising[i])

        return int(rising[np.argmin(np.abs(rising - center))])


class LegacyDiffTrigger(LegacyCorrTrigger):
    """
    Legacy variant: instead of cross-correlation, pick the window whose plain
    sample-by-sample difference from the previous frame's fragment is smallest
    (sum of absolute differences, or squared if DIFF_SQUARED).

    Each window has its own mean removed first (the reference already has), so a
    DC drift does not count as difference. Everything else (search region,
    position bias, sub-sample tracking, zero-cross snap) is shared with Legacy.
    """

    def __init__(self, display_len: int):
        super().__init__(display_len)
        self.position_weight = DIFF_POSITION_WEIGHT

    def _match(self, data: np.ndarray) -> np.ndarray:
        w = len(self.ref)
        if DIFF_SQUARED:
            # sum((x - mean - r)^2) = sum(x^2) - w*mean^2 - 2*corr  + sum(r^2)   (r has zero mean)
            # -> no per-window loop needed, same cost as Legacy
            csum = np.concatenate(([0.0], np.cumsum(data)))
            csum2 = np.concatenate(([0.0], np.cumsum(data * data)))
            win_sum = csum[w:] - csum[:-w]
            win_sq = csum2[w:] - csum2[:-w]
            scale = float(np.dot(self.ref, self.ref))
            err = win_sq - win_sum * win_sum / w - 2.0 * correlate_valid(data, self.ref) + scale
            err = np.maximum(err, 0.0)
        else:
            windows = np.lib.stride_tricks.sliding_window_view(data.astype(np.float32), w)
            diff = windows - self.ref.astype(np.float32)
            # per-window DC: subtract each row's mean difference (ref has zero mean)
            diff -= diff.mean(axis=1, keepdims=True)
            np.abs(diff, out=diff)
            err = diff.sum(axis=1, dtype=np.float64)
            scale = float(np.abs(self.ref).sum())
        # 1.0 = identical, 0.0 = as different as the reference is from silence
        return 1.0 - err / max(scale, 1e-9)


legacy_triggers = {} # one state per signal (mix / left / right)
diff_triggers = {}


def find_trigger(samples: np.ndarray, mode: str, key: str = "mix", end_abs=None):
    """Returns (start, end, period_in_samples or None) into `samples`."""

    display_len = scope_samples

    if mode == "Off":
        # free-run: always the newest samples, no alignment
        end = len(samples)
        return end - display_len, end, None

    if mode == "Corr":
        trigger = corr_triggers.get(key)
        if trigger is None:
            trigger = corr_triggers[key] = CorrelationTrigger(display_len)
        return trigger.find(samples, end_abs)

    if mode == "Legacy":
        trigger = legacy_triggers.get(key)
        if trigger is None:
            trigger = legacy_triggers[key] = LegacyCorrTrigger(display_len)
        return trigger.find(samples, end_abs)

    if mode == "Diff":
        trigger = diff_triggers.get(key)
        if trigger is None:
            trigger = diff_triggers[key] = LegacyDiffTrigger(display_len)
        return trigger.find(samples, end_abs)

    if mode == "FFT":
        start, wave_len = fft_trigger_range(samples, display_len)
        return start, start + display_len, wave_len

    if mode == "Hybrid":
        return hybrid_triggers.setdefault(key, HybridTrigger()).find(samples, display_len)

    offset = len(samples) - 2 * display_len
    start, end = peak_trigger_range(samples[offset:])
    return offset + start, offset + end, None


# ==========================
# Level Meter
# ==========================


class LevelMeter:
    """Per-channel RMS bar + peak-hold + clip indicator (dBFS)."""

    def __init__(self, channels=2):
        self.rms_db = np.full(channels, METER_DB_MIN, dtype=np.float32)
        self.peak_db = np.full(channels, METER_DB_MIN, dtype=np.float32)
        self.hold_db = np.full(channels, METER_DB_MIN, dtype=np.float32)
        self.hold_time = np.zeros(channels, dtype=np.float32)
        self.clip_time = np.zeros(channels, dtype=np.float32)

    @staticmethod
    def _to_db(values):
        return 20.0 * np.log10(np.maximum(values, 1e-9))

    def update(self, channel_samples, dt):
        """channel_samples: (channels, n) newest samples."""
        rms = self._to_db(np.sqrt(np.mean(channel_samples ** 2, axis=1)))
        # +3dB so a full-scale sine reads 0 dBFS on the RMS bar (AES-17 style)
        rms = np.clip(rms + 3.0103, METER_DB_MIN, 0.0)
        peak_linear = np.max(np.abs(channel_samples), axis=1)
        peak = np.clip(self._to_db(peak_linear), METER_DB_MIN, 0.0)

        fall = METER_FALL_DB_PER_SEC * dt
        self.rms_db = np.maximum(rms, self.rms_db - fall)
        self.peak_db = np.maximum(peak, self.peak_db - fall)

        # peak hold: hold for METER_PEAK_HOLD_SEC, then fall
        self.hold_time -= dt
        expired = self.hold_time <= 0.0
        self.hold_db = np.where(expired, np.maximum(self.hold_db - fall, METER_DB_MIN), self.hold_db)
        rising = peak >= self.hold_db
        self.hold_db = np.where(rising, peak, self.hold_db)
        self.hold_time = np.where(rising, METER_PEAK_HOLD_SEC, self.hold_time)

        self.clip_time = np.where(peak_linear >= METER_CLIP_LEVEL, METER_CLIP_HOLD_SEC, self.clip_time - dt)


METER_TICKS = [0, -3, -6, -12, -18, -24, -36, -48, -60]


def _meter_y(db, top, height):
    ratio = (db - METER_DB_MIN) / (0.0 - METER_DB_MIN)
    return int(top + (1.0 - ratio) * (height - 1))


def draw_level_meter(surface, meter, theme, small_font, left, top, width, height, colors):
    """Draws the meter panel inside (left, top, width, height)."""

    pygame.draw.rect(surface, theme["bg"], (left, 0, width, surface.get_height()))
    pygame.draw.line(surface, theme["center"], (left, 0), (left, surface.get_height()))

    label_w = 26
    clip_h = 8
    bar_top = top + clip_h + 4
    bar_h = height - clip_h - 4 - 16 # leave room for L/R labels
    gap = 4
    bar_w = (width - label_w - gap * 3) // 2
    bars_left = left + label_w + gap

    for db in METER_TICKS:
        y = _meter_y(db, bar_top, bar_h)
        pygame.draw.line(surface, theme["grid"], (bars_left - 2, y), (left + width - gap, y))
        text = small_font.render(f"{db}", True, theme["text"])
        surface.blit(text, (bars_left - 4 - text.get_width(), y - text.get_height() // 2))

    warn_y = _meter_y(METER_WARN_DB, bar_top, bar_h)
    bottom = bar_top + bar_h - 1

    for ch, (name, color) in enumerate(zip(("L", "R"), colors)):
        x = bars_left + ch * (bar_w + gap)

        # background track
        pygame.draw.rect(surface, theme["grid"], (x, bar_top, bar_w, bar_h), 1)

        # peak (dim) and RMS (solid) bars
        for value, bar_color in (
            (meter.peak_db[ch], blend(color, theme["bg"], 0.35)),
            (meter.rms_db[ch], color),
        ):
            y = _meter_y(value, bar_top, bar_h)
            if y < bottom:
                pygame.draw.rect(surface, bar_color, (x + 1, y, bar_w - 2, bottom - y))
                if y < warn_y:
                    warn = METER_WARN_COLOR if bar_color == color else blend(METER_WARN_COLOR, theme["bg"], 0.35)
                    pygame.draw.rect(surface, warn, (x + 1, y, bar_w - 2, warn_y - y))

        # peak hold marker
        y = _meter_y(meter.hold_db[ch], bar_top, bar_h)
        if meter.hold_db[ch] > METER_DB_MIN:
            pygame.draw.line(surface, theme["text"], (x + 1, y), (x + bar_w - 2, y), 2)

        # clip indicator
        clip_color = METER_CLIP_COLOR if meter.clip_time[ch] > 0.0 else theme["grid"]
        pygame.draw.rect(surface, clip_color, (x, top, bar_w, clip_h))

        text = small_font.render(name, True, color)
        surface.blit(text, (x + (bar_w - text.get_width()) // 2, bar_top + bar_h + 2))


# ==========================
# Spectrum
# ==========================


class SpectrumAnalyzer:
    """FFT magnitude (dBFS) mapped onto `width` log-spaced frequency columns."""

    def __init__(self, fft_size, rate, width, f_min, f_max, db_min, db_max):
        self.fft_size = fft_size
        self.rate = rate
        self.width = width
        self.f_min = f_min
        self.f_max = f_max
        self.db_min = db_min
        self.db_max = db_max

        self.window = np.hanning(fft_size).astype(np.float32)
        # a full-scale sine reads 0 dBFS
        self.scale = 2.0 / self.window.sum()

        bin_count = fft_size // 2 + 1
        self.bin_positions = np.arange(bin_count, dtype=np.float32)

        # column edges in Hz (log spaced) -> fractional FFT bin positions
        edges_hz = f_min * (f_max / f_min) ** (np.arange(width + 1) / width)
        edges_bin = edges_hz * fft_size / rate
        centers_bin = np.sqrt(edges_bin[:-1] * edges_bin[1:])

        edge_idx = np.clip(np.ceil(edges_bin).astype(np.int64), 0, bin_count - 1)
        self.reduce_idx = edge_idx[:-1]
        # columns that contain at least one bin take the max; sparse (low freq)
        # columns are interpolated instead
        self.dense = edge_idx[1:] > edge_idx[:-1]
        self.centers_bin = centers_bin.astype(np.float32)

        self.columns = np.full(width, db_min, dtype=np.float32)
        self.smoothed = np.full(width, db_min, dtype=np.float32)

    def freq_to_x(self, freq):
        return self.width * math.log(freq / self.f_min) / math.log(self.f_max / self.f_min)

    def update(self, samples, dt):
        spectrum = np.abs(np.fft.rfft(samples * self.window)) * self.scale
        db = 20.0 * np.log10(spectrum + 1e-12)

        peak = np.maximum.reduceat(db, self.reduce_idx)
        interp = np.interp(self.centers_bin, self.bin_positions, db)
        columns = np.where(self.dense, peak, interp)
        self.columns = np.clip(columns, self.db_min, self.db_max).astype(np.float32)

        # instant attack, linear release
        self.smoothed = np.maximum(
            self.columns,
            self.smoothed - SPEC_FALL_DB_PER_SEC * dt,
        )

    def normalized(self, values):
        return (values - self.db_min) / (self.db_max - self.db_min)

    def line_points(self, top, height, round_pixels=True):
        x = np.arange(self.width, dtype=np.float32)
        y = top + (1.0 - self.normalized(self.smoothed)) * (height - 1)
        points = np.column_stack((x, y))
        if not round_pixels:
            return points.astype(np.float32)
        return np.rint(points).astype(np.int32)


SPEC_FREQ_TICKS = [
    (50, "50"), (100, "100"), (200, "200"), (500, "500"),
    (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k"), (20000, "20k"),
]


def _visible_freq_ticks(analyzer):
    for freq, label in SPEC_FREQ_TICKS:
        if analyzer.f_min < freq < analyzer.f_max:
            yield int(analyzer.freq_to_x(freq)), label


def _db_ticks(analyzer, top, height):
    db = math.ceil(analyzer.db_min / 20) * 20
    while db <= analyzer.db_max:
        yield int(top + (1.0 - analyzer.normalized(db)) * (height - 1)), db
        db += 20


def draw_spectrum_grid(surface, analyzer, theme, top, height):
    """Grid lines only; draw before the spectrum."""
    bottom = top + height
    for x, _ in _visible_freq_ticks(analyzer):
        pygame.draw.line(surface, theme["grid"], (x, top), (x, bottom))
    for y, _ in _db_ticks(analyzer, top, height):
        pygame.draw.line(surface, theme["grid"], (0, y), (analyzer.width, y))


def draw_spectrum_labels(surface, analyzer, theme, small_font, top, height, show_db):
    """Axis labels; draw after the spectrum so the fill does not hide them."""
    bottom = top + height
    for x, label in _visible_freq_ticks(analyzer):
        text = small_font.render(label, True, theme["text"])
        surface.blit(text, (x + 3, bottom - text.get_height() - 2))

    if show_db:
        for y, db in _db_ticks(analyzer, top, height):
            text = small_font.render(f"{db:.0f}dB", True, theme["text"])
            surface.blit(text, (analyzer.width - text.get_width() - 4, y + 2))


def draw_spectrum(surface, analyzer, theme, top, height, filled, line_queue=None):
    """
    line_queue: list to receive (points, color, scissor, width scale) for the
    OpenGL line renderer instead of drawing the line with pygame (the fill is
    always drawn here).
    """
    points = analyzer.line_points(top, height, round_pixels=line_queue is None)

    if filled:
        bottom = top + height - 1
        polygon = np.vstack((points, [[analyzer.width - 1, bottom], [0, bottom]]))
        pygame.draw.polygon(surface, theme["fill"], polygon)

    if line_queue is not None:
        # overlay at half the trace width, like the pygame 1 px / 2 px lines
        line_queue.append((points, theme["spectrum"], (0, top, analyzer.width, height), 1.0 if filled else 0.5))
    else:
        pygame.draw.lines(surface, theme["spectrum"], False, points, 1 if not filled else 2)


class Waterfall:
    """Scrolling spectrogram. Newest row at the top."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.history = np.zeros((height, width), dtype=np.uint8)
        self.head = 0
        self.surface = pygame.Surface((width, height))
        self.row_surface = pygame.Surface((width, 1))
        self.pending_rows = 0.0

    def push(self, levels, lut, dt):
        self.pending_rows += dt * WATERFALL_ROWS_PER_SEC
        rows = int(self.pending_rows)
        if rows <= 0:
            return
        self.pending_rows -= rows
        rows = min(rows, self.height)

        row_rgb = lut[levels][:, None, :]
        pygame.surfarray.blit_array(self.row_surface, row_rgb)

        self.surface.scroll(0, rows)
        for r in range(rows):
            self.head = (self.head + 1) % self.height
            self.history[self.head] = levels
            self.surface.blit(self.row_surface, (0, r))

    def rerender(self, lut):
        order = (self.head - np.arange(self.height)) % self.height
        image = lut[self.history[order]]  # (h, w, 3)
        pygame.surfarray.blit_array(self.surface, image.transpose(1, 0, 2))


# ==========================
# PyAudio Callback
# ==========================


def callback(in_data, frame_count, time_info, status):

    global write_pos, samples_written

    samples = np.frombuffer(
        in_data,
        dtype=np.int16
    ).astype(np.float32)

    samples /= 32768.0

    # interleaved -> (channels, n); mono input is duplicated to L and R
    samples = samples.reshape(-1, input_channels).T
    if input_channels == 1:
        samples = np.repeat(samples, 2, axis=0)
    else:
        samples = samples[:2]

    with lock:

        n = samples.shape[1]

        if write_pos + n < RING_SIZE:

            ring[:, write_pos:write_pos+n] = samples

        else:

            first = RING_SIZE - write_pos

            ring[:, write_pos:] = samples[:, :first]

            ring[:, :n-first] = samples[:, first:]

        write_pos = (write_pos + n) % RING_SIZE
        samples_written += n

    return (None, pyaudio.paContinue)


# ==========================
# Audio Start
# ==========================

pa = pyaudio.PyAudio()

def prepare_capture(device):
    return open_capture(pa, pyaudio, device, callback, CHUNK, RATE, CHANNELS)


try:
    devices = capture_devices(pa)
    if len(sys.argv) > 2:
        raise ValueError('Pass one device index/name, --loopback, or --list-devices.')
    if len(sys.argv) > 1:
        if sys.argv[1] in ('--list-devices', '--help', '-h'):
            print('Usage: fluentscope_next.py [DEVICE | --loopback | --list-devices]')
            print('Without arguments, opens the audio source picker.')
            print_devices(devices)
            pa.terminate()
            sys.exit(0)
        device_info = resolve_device(pa, devices, sys.argv[1])
        stream, STREAM_OPEN_KWARGS = prepare_capture(device_info)
    else:
        selection = choose_device(pa, devices, prepare_capture)
        if selection is None:
            pa.terminate()
            sys.exit(0)
        device_info, (stream, STREAM_OPEN_KWARGS) = selection
except Exception as exc:
    print(f'Unable to start audio capture: {exc}')
    print_devices(locals().get('devices', []))
    pa.terminate()
    sys.exit(1)

INPUT_DEVICE_INDEX = int(device_info['index'])
input_channels = STREAM_OPEN_KWARGS['channels']
RATE = STREAM_OPEN_KWARGS['rate']
SPEC_F_MAX = RATE / 2
print(f"Audio source: {device_info['name']} ({input_channels} channels, {RATE} Hz)")


def open_audio_stream():
    audio_stream, _ = prepare_capture(device_info)
    audio_stream.start_stream()
    return audio_stream


def recover_audio_stream():
    global stream

    try:
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

        stream = open_audio_stream()
        print("Audio stream recreated.")
        return True
    except Exception as exc:
        print(f"Warning: failed to recover audio stream: {exc}")
        return False


stream.start_stream()

# ==========================
# Pygame
# ==========================

pygame.init()

# Load fonts before showing a window so its event loop can start immediately.
font = ui_font(18)
small_font = ui_font(13)


def open_gl_window():
    """
    SDL OpenGL 3.3 core window + moderngl context. All pygame drawing goes to an
    offscreen surface which CRTRenderer uploads and composites every frame.
    Returns (offscreen surface, renderer) or None if OpenGL is unavailable.
    """
    try:
        import moderngl
        from crt_renderer import CRTRenderer

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        if sys.platform == 'darwin':
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS,
                                            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1 if VSYNC else 0
        )
        renderer = CRTRenderer(moderngl.create_context(), WIDTH, HEIGHT)
    except Exception as exc:
        print(f"Warning: OpenGL renderer unavailable, CRT / GL line modes disabled ({exc})")
        return None

    print(f"OpenGL: {renderer.ctx.info['GL_RENDERER']} ({renderer.ctx.info['GL_VERSION']})")
    return pygame.Surface((WIDTH, HEIGHT), 0, 32), renderer


gl_renderer = None
gl_window = open_gl_window() if USE_OPENGL else None

if gl_window is not None:
    screen, gl_renderer = gl_window
else:
    screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=1 if VSYNC else 0)

pygame.display.set_caption("FluentScope Next")

clock = pygame.time.Clock()

# ==========================
# View State
# ==========================

VIEW_SCOPE = 0     # waveform (+ optional spectrum overlay)
VIEW_SPECTRUM = 1  # spectrum (top) + waterfall (bottom)
VIEW_LISSAJOUS = 2 # XY plot in a square (L/R or autocorrelation)
VIEW_NAMES = ["Scope", "Spectrum", "Lissajous"]

view = VIEW_SCOPE
show_overlay = True
show_scope_graticule = True
show_scope_dbfs_scale = True
scope_samples = DEFAULT_SCOPE_SAMPLES
scope_agc_mode = "Off"
show_scope_agc_dbfs_scale = False
scope_agc_peak_hold = False
scope_held_peak, scope_hold_remaining = 0.0, 0.0
scope_gain = SCOPE_GAIN
menu_open = False
trigger_mode = TRIGGER_MODE
stereo_mode = STEREO_MODE
lissajous_mode = LISSAJOUS_MODE
lissajous_delay = LISSAJOUS_AUTOCORR_DELAY
lissajous_swap = LISSAJOUS_SWAP
lissajous_invert = LISSAJOUS_INVERT
detected_period = None
theme_index = 0
theme = THEMES[theme_index]

trace_renderer = TRACE_RENDERER if gl_renderer is not None else "Pygame"
line_width = LINE_WIDTH
line_aa = LINE_AA
spectrum_lines = [] # spectrum lines for the OpenGL line renderer, refilled every frame


def gl_line_queue():
    """
    Where plain lines (spectrum) go: a list drawn by the OpenGL line renderer, or
    None to draw them with pygame (no ModernGL, or the Pygame renderer selected).
    """
    if gl_renderer is None or trace_renderer == "Pygame":
        return None
    return spectrum_lines
crt_beam_width = CRT_BEAM_WIDTH
crt_persistence = CRT_PERSISTENCE
crt_exposure = CRT_EXPOSURE
crt_gamma = CRT_GAMMA
crt_bloom = CRT_BLOOM
crt_scene = None         # (view, stereo, theme, ...) - the phosphor is erased when it changes
lissajous_beam_end = None # absolute sample index the Lissajous beam has been drawn up to

# plot area = window minus the level meter panel on the right
PLOT_WIDTH = WIDTH - (METER_PANEL_WIDTH if SHOW_LEVEL_METER else 0)

analyzer = SpectrumAnalyzer(
    FFT_SIZE, RATE, PLOT_WIDTH,
    SPEC_F_MIN, SPEC_F_MAX,
    SPEC_DB_MIN, SPEC_DB_MAX,
)

SPLIT_Y = HEIGHT // 2
waterfall = Waterfall(PLOT_WIDTH, HEIGHT - SPLIT_Y)
waterfall.rerender(theme["lut"])

meter = LevelMeter(2)

overlay_height = int(HEIGHT * SPEC_OVERLAY_RATIO)
overlay_top = HEIGHT - overlay_height


def set_theme(index):
    global theme_index, theme
    theme_index = index % len(THEMES)
    theme = THEMES[theme_index]
    waterfall.rerender(theme["lut"])


# ==========================
# Persistent Settings
# ==========================

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fluentscope_settings.json")


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"Warning: failed to load settings ({exc})")
        return {}


saved_settings = load_settings()


def apply_settings(data):
    """Apply saved values, ignoring anything missing or invalid."""
    global view, show_overlay, trigger_mode, stereo_mode, lissajous_mode, lissajous_delay
    global lissajous_swap, lissajous_invert
    global show_scope_graticule, show_scope_dbfs_scale, scope_samples
    global scope_agc_mode, show_scope_agc_dbfs_scale, scope_agc_peak_hold
    global trace_renderer, line_width, line_aa
    global crt_beam_width, crt_persistence, crt_exposure, crt_gamma, crt_bloom

    def number(key, lo, hi):
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi:
            return value
        return None

    if data.get("view") in VIEW_NAMES:
        view = VIEW_NAMES.index(data["view"])
    if isinstance(data.get("show_overlay"), bool):
        show_overlay = data["show_overlay"]
    if isinstance(data.get("show_scope_graticule"), bool):
        show_scope_graticule = data["show_scope_graticule"]
    if isinstance(data.get("show_scope_dbfs_scale"), bool):
        show_scope_dbfs_scale = data["show_scope_dbfs_scale"]
    samples = data.get("scope_samples")
    if type(samples) is int and samples in SCOPE_SAMPLE_COUNTS:
        scope_samples = samples
    if data.get("scope_agc_mode") in SCOPE_AGC_MODES:
        scope_agc_mode = data["scope_agc_mode"]
    elif data.get("scope_agc") is True:
        scope_agc_mode = "Med"
    if isinstance(data.get("show_scope_agc_dbfs_scale"), bool):
        show_scope_agc_dbfs_scale = data["show_scope_agc_dbfs_scale"]
    if isinstance(data.get("scope_agc_peak_hold"), bool):
        scope_agc_peak_hold = data["scope_agc_peak_hold"]
    if data.get("trigger_mode") in TRIGGER_MODES:
        trigger_mode = data["trigger_mode"]
    if data.get("stereo_mode") in STEREO_MODES:
        stereo_mode = data["stereo_mode"]
    if data.get("lissajous_mode") in LISSAJOUS_MODES:
        lissajous_mode = data["lissajous_mode"]
    if isinstance(data.get("lissajous_swap"), bool):
        lissajous_swap = data["lissajous_swap"]
    if data.get("lissajous_invert") in LISSAJOUS_INVERTS:
        lissajous_invert = data["lissajous_invert"]
    value = number("lissajous_delay", LISSAJOUS_DELAY_MIN, LISSAJOUS_DELAY_MAX)
    if value is not None:
        lissajous_delay = int(value)
    names = [t["name"] for t in THEMES]
    if data.get("theme") in names:
        set_theme(names.index(data["theme"]))

    if gl_renderer is not None:
        if data.get("trace_renderer") in TRACE_RENDERERS:
            trace_renderer = data["trace_renderer"]
        elif isinstance(data.get("crt_enabled"), bool): # older settings files
            trace_renderer = "CRT" if data["crt_enabled"] else "Pygame"
        value = number("line_width", 0.05, 64.0)
        if value is not None:
            line_width = float(value)
        if isinstance(data.get("line_aa"), bool):
            line_aa = data["line_aa"]
        if isinstance(data.get("crt_bloom"), bool):
            crt_bloom = data["crt_bloom"]
        value = number("crt_beam_width", 0.01, 100.0)
        if value is not None:
            crt_beam_width = float(value)
        value = number("crt_persistence", 0.0001, 100.0)
        if value is not None:
            crt_persistence = float(value)
        value = number("crt_exposure", 0.05, 100.0)
        if value is not None:
            crt_exposure = float(value)
        value = number("crt_gamma", 0.1, 10.0)
        if value is not None:
            crt_gamma = float(value)


def current_settings():
    data = dict(saved_settings) # keeps CRT values when OpenGL is unavailable this run
    data.pop("scope_agc", None) # superseded by scope_agc_mode
    data.update({
        "view": VIEW_NAMES[view],
        "show_overlay": show_overlay,
        "show_scope_graticule": show_scope_graticule,
        "show_scope_dbfs_scale": show_scope_dbfs_scale,
        "scope_samples": scope_samples,
        "scope_agc_mode": scope_agc_mode,
        "show_scope_agc_dbfs_scale": show_scope_agc_dbfs_scale,
        "scope_agc_peak_hold": scope_agc_peak_hold,
        "trigger_mode": trigger_mode,
        "stereo_mode": stereo_mode,
        "lissajous_mode": lissajous_mode,
        "lissajous_delay": lissajous_delay,
        "lissajous_swap": lissajous_swap,
        "lissajous_invert": lissajous_invert,
        "theme": theme["name"],
    })
    if gl_renderer is not None:
        data.pop("crt_enabled", None) # superseded by trace_renderer
        data.update({
            "trace_renderer": trace_renderer,
            "line_width": line_width,
            "line_aa": line_aa,
            "crt_beam_width": crt_beam_width,
            "crt_persistence": crt_persistence,
            "crt_exposure": crt_exposure,
            "crt_gamma": crt_gamma,
            "crt_bloom": crt_bloom,
        })
    return data


def save_settings_if_changed():
    global saved_settings
    data = current_settings()
    if data == saved_settings:
        return
    try:
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, SETTINGS_PATH)
        saved_settings = data
    except Exception as exc:
        print(f"Warning: failed to save settings ({exc})")


apply_settings(saved_settings)


def draw_trace(waveform, color, top, height):
    points = build_waveform_points(
        waveform,
        PLOT_WIDTH,
        height,
        scope_gain,
        LINE_IS_STEP,
        top,
    )

    if points is not None:
        pygame.draw.lines(screen, color, False, points, 2)


def draw_scope_graticule(lanes):
    right, bottom = PLOT_WIDTH - 1, HEIGHT - 1
    center_x = round(right / 2)
    for division in range(11):
        x = round(right * division / 10)
        pygame.draw.line(screen, theme["grid"], (x, 0), (x, bottom))
    pygame.draw.line(screen, theme["center"], (center_x, 0), (center_x, bottom))

    divisions = 8 // len(lanes)
    # Rebase by powers of two to keep the grid readable across the gain range.
    zoom = scope_gain / SCOPE_GAIN
    spacing = zoom / 2 ** math.floor(math.log2(zoom))
    for top, height in lanes:
        lane_bottom = min(top + height, bottom)
        center_y = round(top + height / 2)
        span = lane_bottom - top
        for division in range(-divisions // 2, divisions // 2 + 1):
            y = round(top + span * (0.5 + division * spacing / divisions))
            if top <= y <= lane_bottom:
                pygame.draw.line(screen, theme["grid"], (0, y), (right, y))
        pygame.draw.line(screen, theme["center"], (0, center_y), (right, center_y))

        for subdivision in range(1, 50):
            x = round(right * subdivision / 50)
            tick = 5 if subdivision % 5 == 0 else 2
            pygame.draw.line(screen, theme["center"], (x, center_y - tick), (x, center_y + tick))
        half_ticks = divisions * 5 // 2
        for subdivision in range(1 - half_ticks, half_ticks):
            y = round(top + span * (0.5 + subdivision * spacing / (divisions * 5)))
            if not top <= y <= lane_bottom:
                continue
            tick = 5 if subdivision % 5 == 0 else 2
            pygame.draw.line(screen, theme["center"], (center_x - tick, y), (center_x + tick, y))

    for top, _ in lanes[1:]:
        pygame.draw.line(screen, theme["center"], (0, top), (right, top))


def draw_scope_dbfs_scale(top, height):
    scope_agc = scope_agc_mode != "Off"
    x = PLOT_WIDTH // 4
    bottom = top + height - 1
    upper = max(top, round(waveform_y(1, top, height, scope_gain)))
    lower = min(bottom, round(waveform_y(-1, top, height, scope_gain)))
    pygame.draw.line(screen, theme["center"], (x, upper), (x, lower))

    levels = (0, -3, -6, -12, -18)
    if scope_agc:
        levels += (-24, -30, -36, -42, -48, -54, -60)
    for polarity in (1, -1):
        last_label_y = None
        for db in levels:
            amplitude = 10 ** (db / 20)
            y = round(waveform_y(polarity * amplitude, top, height, scope_gain))
            if not top <= y <= bottom:
                continue
            if scope_agc:
                if abs(y - (top + height / 2)) < small_font.get_height() / 2:
                    break
                if last_label_y is not None and abs(y - last_label_y) < small_font.get_height():
                    break
            pygame.draw.line(screen, theme["center"], (x - 4, y), (x + 4, y))
            label = f"{db} dBFS" if scope_agc or db == 0 else str(db)
            text = small_font.render(label, True, theme["text"])
            rect = text.get_rect(midleft=(x + 7, y))
            if rect.top >= max(top, 78 if scope_agc else 60) and rect.bottom <= bottom + 1:
                screen.blit(text, rect)
                last_label_y = y


def draw_scope(traces):
    """traces: list of (waveform, color, top, height)."""
    lanes = sorted({(top, height) for _, _, top, height in traces})
    if show_scope_graticule:
        draw_scope_graticule(lanes)

    if show_overlay:
        draw_spectrum(screen, analyzer, theme, overlay_top, overlay_height, False, gl_line_queue())
        draw_spectrum_labels(screen, analyzer, theme, small_font, overlay_top, overlay_height, False)

    show_dbfs_scale = show_scope_dbfs_scale if scope_agc_mode == "Off" else show_scope_agc_dbfs_scale
    if show_dbfs_scale:
        for top, height in lanes:
            draw_scope_dbfs_scale(top, height)

    if trace_renderer == "Pygame": # otherwise drawn on the GPU in render_gl()
        for waveform, color, top, height in traces:
            draw_trace(waveform, color, top, height)

    # channel labels
    if stereo_mode != "Mix":
        for i, (name, color) in enumerate((("L", theme["trace_l"]), ("R", theme["trace_r"]))):
            label = font.render(name, True, color)
            if stereo_mode == "Split":
                pos = (PLOT_WIDTH - label.get_width() - 8, i * SPLIT_Y + (8 if i else 64))
            else:
                pos = (PLOT_WIDTH - 24 * (2 - i) - 4, 64)
            screen.blit(label, pos)


def draw_spectrum_view():
    draw_spectrum_grid(screen, analyzer, theme, 0, SPLIT_Y)
    draw_spectrum(screen, analyzer, theme, 0, SPLIT_Y, True, gl_line_queue())
    draw_spectrum_labels(screen, analyzer, theme, small_font, 0, SPLIT_Y, True)

    screen.blit(waterfall.surface, (0, SPLIT_Y))
    pygame.draw.line(screen, theme["center"], (0, SPLIT_Y), (PLOT_WIDTH, SPLIT_Y), 2)


def effective_lissajous_mode():
    return "Auto" if input_channels == 1 else lissajous_mode


def orient_lissajous(x, y):
    """Apply [W] swap and [I] invert to the X / Y signals."""
    if lissajous_swap:
        x, y = y, x
    if "X" in lissajous_invert:
        x = -x
    if "Y" in lissajous_invert:
        y = -y
    return x, y


def lissajous_rect():
    """Square plot area (left, top, side), centered in the plot width."""
    side = min(PLOT_WIDTH - 16, HEIGHT - LISSAJOUS_TOP - 8)
    left = (PLOT_WIDTH - side) // 2
    return left, LISSAJOUS_TOP, side


def build_lissajous_points(left, right, mix, round_pixels=True):
    """(N, 2) screen points of the XY figure (int32, or float32 subpixel)."""

    if effective_lissajous_mode() == "Auto":
        # X = s[i], Y = s[i + delay]: newest sample pairs that both exist
        count = min(LISSAJOUS_POINTS, len(mix) - lissajous_delay)
        x = mix[-count - lissajous_delay:-lissajous_delay]
        y = mix[-count:]
    else:
        x = left[-LISSAJOUS_POINTS:]
        y = right[-LISSAJOUS_POINTS:]
    x, y = orient_lissajous(x, y)

    rect_left, rect_top, side = lissajous_rect()
    half = side / 2
    scale = half * LISSAJOUS_GAIN
    px = rect_left + half + np.clip(x * scale, -half, half)
    py = rect_top + half - np.clip(y * scale, -half, half)
    points = np.column_stack((px, py))
    if not round_pixels:
        return points.astype(np.float32)
    return np.rint(points).astype(np.int32)


def draw_lissajous(points):
    rect_left, rect_top, side = lissajous_rect()
    right = rect_left + side - 1
    bottom = rect_top + side - 1
    mid_x = rect_left + side // 2
    mid_y = rect_top + side // 2

    # grid: 8 divisions, center cross, diagonals (mono / anti-phase lines in L/R)
    for i in range(1, 8):
        offset = side * i // 8
        pygame.draw.line(screen, theme["grid"], (rect_left + offset, rect_top), (rect_left + offset, bottom))
        pygame.draw.line(screen, theme["grid"], (rect_left, rect_top + offset), (right, rect_top + offset))
    pygame.draw.line(screen, theme["grid"], (rect_left, bottom), (right, rect_top))
    pygame.draw.line(screen, theme["grid"], (rect_left, rect_top), (right, bottom))
    pygame.draw.line(screen, theme["center"], (mid_x, rect_top), (mid_x, bottom), 2)
    pygame.draw.line(screen, theme["center"], (rect_left, mid_y), (right, mid_y), 2)
    pygame.draw.rect(screen, theme["center"], (rect_left, rect_top, side, side), 1)

    if points is not None and len(points) > 1:
        pygame.draw.lines(screen, theme["trace"], False, points, 1)

    # axis labels
    if effective_lissajous_mode() == "Auto":
        x_name, y_name = "s[t]", f"s[t+{lissajous_delay}]"
        x_color = y_color = theme["text"]
    else:
        x_name, y_name = "L", "R"
        x_color, y_color = theme["trace_l"], theme["trace_r"]
    if lissajous_swap:
        x_name, y_name, x_color, y_color = y_name, x_name, y_color, x_color
    if "X" in lissajous_invert:
        x_name = "-" + x_name
    if "Y" in lissajous_invert:
        y_name = "-" + y_name
    x_label = small_font.render(x_name, True, x_color)
    y_label = small_font.render(y_name, True, y_color)
    screen.blit(x_label, (right - x_label.get_width() - 4, mid_y + 4))
    screen.blit(y_label, (mid_x + 6, rect_top + 4))


def build_traces(left, right, mix, end_abs):
    """Trigger and slice the waveforms for the current stereo mode."""

    start, end, period = find_trigger(mix, trigger_mode, "mix", end_abs)

    if stereo_mode == "Mix":
        return [(mix[start:end], theme["trace"], 0, HEIGHT)], period

    if STEREO_SHARED_TRIGGER:
        l_wave, r_wave = left[start:end], right[start:end]
    else:
        ls, le, _ = find_trigger(left, trigger_mode, "left", end_abs)
        rs, re, _ = find_trigger(right, trigger_mode, "right", end_abs)
        l_wave, r_wave = left[ls:le], right[rs:re]

    if stereo_mode == "Overlay":
        # R first so L stays on top
        return [
            (r_wave, theme["trace_r"], 0, HEIGHT),
            (l_wave, theme["trace_l"], 0, HEIGHT),
        ], period

    # Split: L on top, R on bottom
    return [
        (l_wave, theme["trace_l"], 0, SPLIT_Y),
        (r_wave, theme["trace_r"], SPLIT_Y, HEIGHT - SPLIT_Y),
    ], period


# ==========================
# CRT Beam (OpenGL)
# ==========================

if gl_renderer is not None:
    from crt_renderer import upsample


def color01(color):
    return tuple(c / 255.0 for c in color)


def theme_is_light():
    r, g, b = theme["bg"]
    return 0.299 * r + 0.587 * g + 0.114 * b > 128


def beam_scope(traces):
    """Triggered sweep: the whole window is one beam sweep, redrawn every frame."""

    energy = gl_renderer.sweep_energy() / CRT_UPSAMPLE
    for waveform, color, top, height in traces:
        if len(waveform) < 2:
            continue
        y = upsample(waveform_y(waveform, top, height, scope_gain), CRT_UPSAMPLE)
        x = np.linspace(0, PLOT_WIDTH - 1, len(y), dtype=np.float32)
        gl_renderer.draw_beam(
            np.column_stack((x, y)), energy, color01(color),
            scissor=(0, top, PLOT_WIDTH, height),
        )


def beam_lissajous(left, right, mix, end_abs):
    """
    XY mode runs continuously like a real vector display: only the samples that
    arrived since the last frame are drawn, the phosphor keeps the rest.
    """

    global lissajous_beam_end

    auto = effective_lissajous_mode() == "Auto"
    delay = lissajous_delay if auto else 0
    margin = CRT_UPSAMPLE_MARGIN if CRT_UPSAMPLE > 1 else 0

    available = min(len(mix), LISSAJOUS_HISTORY) - delay - margin - 1
    new = available if lissajous_beam_end is None else end_abs - lissajous_beam_end
    new = int(min(max(new, 0), available))
    lissajous_beam_end = end_abs
    if new <= 0:
        return

    # +1: start at the last sample drawn before, so the path stays continuous
    count = new + 1 + margin
    if auto:
        x = mix[-count - delay:-delay]
        y = mix[-count:]
    else:
        x = left[-count:]
        y = right[-count:]
    x, y = orient_lissajous(x, y)

    rect_left, rect_top, side = lissajous_rect()
    half = side / 2
    scale = half * LISSAJOUS_GAIN
    px = upsample(rect_left + half + x * scale, CRT_UPSAMPLE, margin)
    py = upsample(rect_top + half - y * scale, CRT_UPSAMPLE, margin)

    energy = np.repeat(gl_renderer.stream_energy(new, RATE, CRT_LISSAJOUS_REF), CRT_UPSAMPLE)
    energy /= CRT_UPSAMPLE

    # off the square = off the tube face
    gl_renderer.draw_beam(
        np.column_stack((px, py)), energy, color01(theme["trace"]),
        scissor=(rect_left, rect_top, side, side),
    )


def line_scope(traces):
    for waveform, color, top, height in traces:
        points = build_waveform_points(
            waveform, PLOT_WIDTH, height, scope_gain, LINE_IS_STEP, top, round_pixels=False,
        )
        if points is not None:
            gl_renderer.draw_line(points, color01(color), scissor=(0, top, PLOT_WIDTH, height))


def line_lissajous(left, right, mix):
    rect_left, rect_top, side = lissajous_rect()
    points = build_lissajous_points(left, right, mix, round_pixels=False)
    gl_renderer.draw_line(points, color01(theme["trace"]), scissor=(rect_left, rect_top, side, side))


def render_gl(traces, left, right, mix, end_abs):
    """Upload the pygame-drawn UI, add the traces (CRT / GL line) and composite to the window."""

    global crt_scene, lissajous_beam_end

    r = gl_renderer
    r.upload_ui(screen)
    ui_rect = tuple(MENU_PANEL) if menu_open else None
    trace_view = view in (VIEW_SCOPE, VIEW_LISSAJOUS)

    # plain line layer: spectrum, then the GL Line traces on top
    gl_traces = trace_renderer == "GL Line" and trace_view
    if spectrum_lines or gl_traces:
        r.line_width = line_width
        r.line_aa = line_aa
        r.begin_lines()
        for points, color, scissor, scale in spectrum_lines:
            r.draw_line(points, color01(color), scissor=scissor, width=line_width * scale)
        if gl_traces:
            if view == VIEW_SCOPE:
                line_scope(traces)
            else:
                line_lissajous(left, right, mix)
    spectrum_lines.clear()

    if not (trace_renderer == "CRT" and trace_view):
        crt_scene = None
        lissajous_beam_end = None
        r.present(False)
        return

    scene = (view, stereo_mode, theme_index, effective_lissajous_mode(), lissajous_swap,
             lissajous_invert, view == VIEW_SCOPE and scope_agc_mode != "Off",
             scope_samples if view == VIEW_SCOPE else None)
    if scene != crt_scene:
        r.clear()
        crt_scene = scene
        lissajous_beam_end = None

    r.beam_sigma = crt_beam_width
    r.persistence = crt_persistence
    r.exposure = crt_exposure
    r.bloom = crt_bloom
    r.bloom_near = CRT_BLOOM_NEAR
    r.bloom_far = CRT_BLOOM_FAR
    r.gamma = crt_gamma
    r.knee = CRT_SATURATION_KNEE
    r.white_core = CRT_WHITE_CORE
    r.vignette = CRT_VIGNETTE
    r.ink = theme_is_light()

    r.begin_frame(dt)
    if view == VIEW_SCOPE:
        beam_scope(traces)
    else:
        beam_lissajous(left, right, mix, end_abs)
    r.present(True, ui_rect=ui_rect)


def step_option(options, current, step):
    """Next / previous entry of a sorted option list (current may be off the list)."""
    if step > 0:
        larger = [o for o in options if o > current]
        return larger[0] if larger else options[-1]
    smaller = [o for o in options if o < current]
    return smaller[-1] if smaller else options[0]


def format_seconds(value):
    return f"{value * 1000:.0f}ms" if value < 1.0 else f"{value:.1f}s"


def format_line_width(value):
    return f"{value:g}px"


def cycle(options, current):
    return options[(options.index(current) + 1) % len(options)]


MENU_BUTTON = pygame.Rect(PLOT_WIDTH - 122, 6, 112, 24)
MENU_PANEL = pygame.Rect(PLOT_WIDTH - 894, 36, 884, 232)


def scale_controls():
    return (
        (pygame.K_j, "J", "Grid", show_scope_graticule),
        (pygame.K_k, "K", "Fixed dBFS", show_scope_dbfs_scale),
        (pygame.K_d, "D", "AGC dBFS", show_scope_agc_dbfs_scale),
    )


def scope_time_span():
    return f"{scope_samples * 1000 / RATE:.2f} ms"


def menu_items():
    agc = scope_agc_mode
    hold = format_seconds(SCOPE_AGC_HOLD_TIME) if scope_agc_peak_hold else "Off"
    groups = [
        ("Display", (
            (pygame.K_TAB, "Tab", f"View: {VIEW_NAMES[view]}"),
            (pygame.K_t, "T", f"Theme: {theme['name']}"),
            (pygame.K_s, "S", f"Spectrum overlay: {'On' if show_overlay else 'Off'}"),
            (pygame.K_r, "R", f"Trigger: {trigger_mode}"),
            (pygame.K_m, "M", f"Stereo: {stereo_mode}"),
        )),
        ("Scope", (
            (pygame.K_x, "X", f"Time span: {scope_time_span()}"),
            (pygame.K_a, "A", f"AGC: {agc}"),
            (pygame.K_h, "H", f"AGC peak hold: {hold}"),
            *((key, shortcut, f"{name}: {'On' if enabled else 'Off'}")
              for key, shortcut, name, enabled in scale_controls()),
        )),
        ("Lissajous", (
            (pygame.K_l, "L", f"XY mode: {lissajous_mode}"),
            (pygame.K_w, "W", f"Swap X/Y: {'On' if lissajous_swap else 'Off'}"),
            (pygame.K_i, "I", f"Invert: {lissajous_invert}"),
            (pygame.K_COMMA, ",", "Decrease XY delay"),
            (pygame.K_PERIOD, ".", "Increase XY delay"),
        )),
    ]
    if gl_renderer is not None:
        items = [(pygame.K_c, "C", f"Renderer: {trace_renderer}")]
        if trace_renderer == "CRT":
            items += [
                (pygame.K_e, "E", f"Exposure: {crt_exposure:.2f}"),
                (pygame.K_f, "F", f"Focus: {crt_beam_width:.1f}px"),
                (pygame.K_g, "G", f"Afterglow: {format_seconds(crt_persistence)}"),
                (pygame.K_b, "B", f"Bloom: {'On' if crt_bloom else 'Off'}"),
                (pygame.K_v, "V", f"Gamma: {crt_gamma:.1f}"),
            ]
        if trace_renderer != "Pygame": # GL lines: traces in GL Line mode, spectrum in both
            items += [
                (pygame.K_n, "N", f"Line width: {format_line_width(line_width)}"),
                (pygame.K_o, "O", f"Anti-aliasing: {'On' if line_aa else 'Off'}"),
            ]
        groups.append(("Renderer", tuple(items)))
    return groups


def menu_row(column, row):
    return pygame.Rect(MENU_PANEL.x + 10 + 217 * column, MENU_PANEL.y + 32 + 21 * row, 209, 20)


def draw_button(rect, text):
    pygame.draw.rect(screen, theme["bg"], rect)
    pygame.draw.rect(screen, theme["center"], rect, 1)
    label = small_font.render(text, True, theme["text"])
    screen.blit(label, label.get_rect(center=rect.center))


def draw_menu():
    draw_button(MENU_BUTTON, "Close [F1]" if menu_open else "Menu [F1]")
    if not menu_open:
        return

    pygame.draw.rect(screen, theme["bg"], MENU_PANEL)
    pygame.draw.rect(screen, theme["center"], MENU_PANEL, 1)
    pointer = pygame.mouse.get_pos()
    for column, (title, items) in enumerate(menu_items()):
        x = menu_row(column, 0).x
        screen.blit(font.render(title, True, theme["text"]), (x + 4, MENU_PANEL.y + 7))
        for row, (_, shortcut, text) in enumerate(items):
            rect = menu_row(column, row)
            if rect.collidepoint(pointer):
                pygame.draw.rect(screen, theme["grid"], rect)
            label = small_font.render(f"[{shortcut}] {text}", True, theme["text"])
            screen.blit(label, (rect.x + 4, rect.y + 2))
    hint = "Shift reverses supported controls or changes XY delay by x2 / x0.5."
    screen.blit(small_font.render(hint, True, theme["text"]), (MENU_PANEL.x + 14, MENU_PANEL.bottom - 22))


# ==========================
# Main Loop
# ==========================

print(r"""     ______              __  ____
    / __/ /_ _____ ___  / /_/ __/______  ___  ___
   / _// / // / -_) _ \/ __/\ \/ __/ _ \/ _ \/ -_)
  /_/ /_/\_,_/\__/_//_/\__/___/\__/\___/ .__/\__/
                                      /_/  [Next]
  ~ Completely Rewritten ~
  (c) 2026 src3453 Released under The MIT License.""")
print("  Keys: [Tab]/[1]/[2]/[3] view  [S] spectrum overlay  [T]/[Shift+T] theme  [R] trigger mode  [M] stereo mode")
print("        [F1] Menu  [A] Scope AGC Off/High/Med/Low  [D] AGC dBFS labels  [H] AGC peak hold")
print("        [J] Scope grid  [K] Fixed-gain dBFS labels")
print("        [X]/[Shift+X] Scope time span")
print("        [W] Lissajous swap X/Y  [I] invert Off/X/Y/XY (+Shift: back)")
print("        [L] Lissajous L/R <-> Auto  [,] [.] (or [ and ]) autocorrelation delay -1 / +1 (+Shift: x0.5 / x2)")
if gl_renderer is not None:
    print("        [C] renderer CRT / GL Line / Pygame (+Shift: back)")
    print("        CRT:     [E] exposure  [F] focus  [G] afterglow  [V] gamma (+Shift: down)  [B] bloom")
    print("        Lines:   [N] line width (+Shift: thinner)  [O] anti-aliasing (GL Line traces, spectrum)")

running = True
dt = 0.0
focus_window()

while running:

    for event in pygame.event.get():

        if event.type == pygame.WINDOWTAKEFOCUS:
            focus_window()
            continue

        if event.type == pygame.MOUSEBUTTONDOWN and not pygame.key.get_focused():
            focus_window()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if MENU_BUTTON.collidepoint(event.pos):
                menu_open = not menu_open
                continue
            if menu_open:
                key = next((
                    key for column, (_, items) in enumerate(menu_items())
                    for row, (key, _, _) in enumerate(items)
                    if menu_row(column, row).collidepoint(event.pos)
                ), None)
                if key is None:
                    if not MENU_PANEL.collidepoint(event.pos):
                        menu_open = False
                    continue
                event = pygame.event.Event(pygame.KEYDOWN, key=key, mod=pygame.key.get_mods())

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_F1:
                menu_open = not menu_open
                continue
            elif event.key == pygame.K_ESCAPE and menu_open:
                menu_open = False
                continue
            elif event.key == pygame.K_TAB:
                view = (view + 1) % len(VIEW_NAMES)
            elif event.key == pygame.K_1:
                view = VIEW_SCOPE
            elif event.key == pygame.K_2:
                view = VIEW_SPECTRUM
            elif event.key == pygame.K_3:
                view = VIEW_LISSAJOUS
            elif event.key == pygame.K_l:
                lissajous_mode = cycle(LISSAJOUS_MODES, lissajous_mode)
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_COMMA, pygame.K_PERIOD):
                # autocorrelation delay: x2 / x0.5 with Shift, otherwise +-1 sample.
                # [ ] or , . (the latter sit at the same place on US and JIS keyboards)
                up = event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD)
                if view == VIEW_LISSAJOUS and lissajous_mode != "Auto":
                    lissajous_mode = "Auto" # the delay only exists in Auto mode
                if event.mod & pygame.KMOD_SHIFT:
                    lissajous_delay = lissajous_delay * 2 if up else lissajous_delay // 2
                else:
                    lissajous_delay += 1 if up else -1
                lissajous_delay = min(max(lissajous_delay, LISSAJOUS_DELAY_MIN), LISSAJOUS_DELAY_MAX)
            elif event.key == pygame.K_w:
                lissajous_swap = not lissajous_swap
            elif event.key == pygame.K_i:
                step = -1 if event.mod & pygame.KMOD_SHIFT else 1
                index = LISSAJOUS_INVERTS.index(lissajous_invert) + step
                lissajous_invert = LISSAJOUS_INVERTS[index % len(LISSAJOUS_INVERTS)]
            elif event.key == pygame.K_s:
                show_overlay = not show_overlay
            elif event.key == pygame.K_t:
                step = -1 if event.mod & pygame.KMOD_SHIFT else 1
                set_theme(theme_index + step)
            elif event.key == pygame.K_r:
                trigger_mode = cycle(TRIGGER_MODES, trigger_mode)
            elif event.key == pygame.K_m:
                stereo_mode = cycle(STEREO_MODES, stereo_mode)
                scope_gain = SCOPE_GAIN
                scope_held_peak, scope_hold_remaining = 0.0, 0.0
            elif event.key == pygame.K_x:
                step = -1 if event.mod & pygame.KMOD_SHIFT else 1
                index = SCOPE_SAMPLE_COUNTS.index(scope_samples) + step
                scope_samples = SCOPE_SAMPLE_COUNTS[index % len(SCOPE_SAMPLE_COUNTS)]
                for triggers in (hybrid_triggers, corr_triggers, legacy_triggers, diff_triggers):
                    triggers.clear()
            elif event.key == pygame.K_a:
                scope_agc_mode = cycle(SCOPE_AGC_MODES, scope_agc_mode)
                if scope_agc_mode == "Off":
                    scope_gain = SCOPE_GAIN
            elif event.key == pygame.K_d:
                show_scope_agc_dbfs_scale = not show_scope_agc_dbfs_scale
            elif event.key == pygame.K_j:
                show_scope_graticule = not show_scope_graticule
            elif event.key == pygame.K_k:
                show_scope_dbfs_scale = not show_scope_dbfs_scale
            elif event.key == pygame.K_h:
                scope_agc_peak_hold = not scope_agc_peak_hold
                scope_held_peak, scope_hold_remaining = 0.0, 0.0
            elif gl_renderer is not None:
                # renderer / CRT beam / GL line controls (Shift = the other direction)
                step = -1 if event.mod & pygame.KMOD_SHIFT else 1
                if event.key == pygame.K_c:
                    index = TRACE_RENDERERS.index(trace_renderer) + step
                    trace_renderer = TRACE_RENDERERS[index % len(TRACE_RENDERERS)]
                elif event.key == pygame.K_n:
                    line_width = step_option(LINE_WIDTHS, line_width, step)
                elif event.key == pygame.K_o:
                    line_aa = not line_aa
                elif event.key == pygame.K_g:
                    crt_persistence = step_option(CRT_PERSISTENCES, crt_persistence, step)
                elif event.key == pygame.K_f:
                    crt_beam_width = step_option(CRT_BEAM_WIDTHS, crt_beam_width, step)
                elif event.key == pygame.K_e:
                    crt_exposure *= CRT_EXPOSURE_STEP ** step
                    crt_exposure = min(max(crt_exposure, 0.05), 100.0)
                elif event.key == pygame.K_b:
                    crt_bloom = not crt_bloom
                elif event.key == pygame.K_v:
                    crt_gamma = step_option(CRT_GAMMAS, crt_gamma, step)

            save_settings_if_changed() # auto-save whenever a key changed a setting

    if not stream.is_active():
        recover_audio_stream()

    #
    # Get samples
    #

    with lock:

        channels = np.concatenate((
            ring[:, write_pos:],
            ring[:, :write_pos]
        ), axis=1)

        end_abs = samples_written

    left, right = channels
    mix = (left + right) * 0.5

    traces, detected_period = build_traces(left, right, mix, end_abs)

    #
    # Spectrum / waterfall (always updated so history is ready on view switch)
    #

    analyzer.update(mix[-FFT_SIZE:], dt)

    levels = np.rint(analyzer.normalized(analyzer.columns) * 255).astype(np.uint8)
    waterfall.push(levels, theme["lut"], dt)

    meter.update(channels[:, -METER_WINDOW:], dt)

    #
    # Draw
    #

    screen.fill(theme["bg"])

    if view == VIEW_SCOPE and scope_agc_mode != "Off":
        peak = max(float(np.max(np.abs(waveform), initial=0.0)) for waveform, _, _, _ in traces)
        if scope_agc_peak_hold:
            scope_held_peak, scope_hold_remaining = update_scope_peak_hold(
                peak, scope_held_peak, scope_hold_remaining, dt,
            )
            peak = scope_held_peak
        scope_gain = update_scope_gain(peak, scope_gain, dt, SCOPE_AGC_DECAY_TIMES[scope_agc_mode])
    else:
        scope_gain = SCOPE_GAIN
        scope_held_peak, scope_hold_remaining = 0.0, 0.0

    if view == VIEW_SCOPE:
        draw_scope(traces)
    elif view == VIEW_LISSAJOUS:
        draw_lissajous(build_lissajous_points(left, right, mix) if trace_renderer == "Pygame" else None)
    else:
        draw_spectrum_view()

    if SHOW_LEVEL_METER:
        draw_level_meter(
            screen, meter, theme, small_font,
            PLOT_WIDTH, 10, METER_PANEL_WIDTH, HEIGHT - 20,
            (theme["trace_l"], theme["trace_r"]),
        )

    #
    # Text
    #

    info = font.render(
        f"Rate:{RATE}Hz   FFT:{FFT_SIZE}   FPS:{clock.get_fps():.1f}   VSYNC:{'ON' if VSYNC else 'OFF'}",
        True,
        theme["text"]
    )

    screen.blit(info, (10, 10))

    if trigger_mode in ("FFT", "Hybrid", "Corr"):
        trigger_info = f"{trigger_mode} ({RATE / detected_period:.1f}Hz)" if detected_period else f"{trigger_mode} (no lock)"
    else:
        trigger_info = trigger_mode

    stereo_info = stereo_mode if input_channels >= 2 else f"{stereo_mode} (mono in)"

    if view == VIEW_LISSAJOUS:
        if effective_lissajous_mode() == "Auto":
            mono = " (mono in)" if input_channels == 1 else ""
            liss_info = f"Auto{mono}   Delay:{lissajous_delay} ({lissajous_delay / RATE * 1000:.2f}ms)"
        else:
            liss_info = "L/R"
        help_text = (
            f"View:{VIEW_NAMES[view]}   Theme:{theme['name']}   Lissajous:{liss_info}"
            f"   Swap:{'ON' if lissajous_swap else 'OFF'}   Invert:{lissajous_invert}"
        )
    else:
        help_text = f"View:{VIEW_NAMES[view]}   Overlay:{'ON' if show_overlay else 'OFF'}   Theme:{theme['name']}   Trigger:{trigger_info}   Stereo:{stereo_info}"
        if view == VIEW_SCOPE:
            help_text += f"   Time span: {scope_time_span()}"

    keys_help = small_font.render(help_text, True, theme["text"])

    screen.blit(keys_help, (10, 32))

    if view == VIEW_SCOPE:
        agc_status = scope_agc_mode
        if scope_agc_mode != "Off":
            agc_status += f" x{scope_gain / SCOPE_GAIN:.2f}"
        hold_status = format_seconds(SCOPE_AGC_HOLD_TIME) if scope_agc_peak_hold else "OFF"
        scale_status = "   ".join(
            f"{name}:{'On' if enabled else 'Off'}" for _, _, name, enabled in scale_controls()
        )
        agc_text = f"AGC:{agc_status}   Peak hold:{hold_status}   {scale_status}"
        screen.blit(small_font.render(agc_text, True, theme["text"]), (10, 62))

    if gl_renderer is not None:
        if trace_renderer == "CRT":
            renderer_text = (
                f"Renderer:CRT   Exposure:{crt_exposure:.2f}   Focus:{crt_beam_width:.1f}px"
                f"   Afterglow:{format_seconds(crt_persistence)}   Bloom:{'ON' if crt_bloom else 'OFF'}"
                f"   Gamma:{crt_gamma:.1f}{' (ink: 1.0)' if theme_is_light() else ''}"
                f"   Line:{format_line_width(line_width)} AA:{'ON' if line_aa else 'OFF'}"
            )
        elif trace_renderer == "GL Line":
            renderer_text = (
                f"Renderer:GL Line   Width:{format_line_width(line_width)}"
                f"   AA:{'ON' if line_aa else 'OFF'}"
            )
        else:
            renderer_text = "Renderer:Pygame"
        screen.blit(small_font.render(renderer_text, True, theme["text"]), (10, 47))

    draw_menu()
    if gl_renderer is not None:
        render_gl(traces, left, right, mix, end_abs)

    pygame.display.flip()

    if VSYNC:
        dt = clock.tick() / 1000.0
    else:
        dt = clock.tick(FPS) / 1000.0

# ==========================
# Shutdown
# ==========================

try:
    if stream.is_active():
        stream.stop_stream()
except Exception:
    try:
        stream.abort_stream()
    except Exception:
        pass

stream.close()

pa.terminate()

pygame.quit()
