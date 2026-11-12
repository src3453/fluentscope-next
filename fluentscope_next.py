import json
import math
import os
import sys
import threading

from spectrum_analysis import (
    SpectrumAnalyzer, SPECTRUM_MODES, SPECTRUM_SMOOTHING,
    SPECTRUM_DB_RANGES, CEPSTRUM_ZOOMS,
)

from window_support import (
    configure_windowing, create_window, desktop_size, render_size, mouse_position, activate_window,
)

configure_windowing()

import numpy as np
import pyglet
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

SHOW_LEVEL_METER = True # L/R meter on the right edge (reserves METER_PANEL_WIDTH pixels), [F2] toggles
SHOW_HUD = True # status text + menu button, [F3] toggles ([F1] opens the menu even when hidden)
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

WIDTH = 1000 # initial window size; the window is resizable and remembers its last size
HEIGHT = 600
MIN_WIDTH = 640 # smallest window the layout (menu, status text, meter) is designed for
MIN_HEIGHT = 480
SCOPE_GAIN = 0.9
SCOPE_AGC_MAX_BOOST = 32.0
SCOPE_AGC_SILENCE = 0.0001
SCOPE_AGC_DECAY_TIMES = {"High": 0.12, "Med": 0.35, "Low": 1.0} # seconds
SCOPE_AGC_MODES = ("Off", *SCOPE_AGC_DECAY_TIMES)
SCOPE_AGC_HOLD_TIME = 0.5 # seconds

FPS = 180 # frame-rate cap; no effect when VSYNC is enabled ([F5] cycles FPS_OPTIONS)
FPS_OPTIONS = (30, 60, 120, 144, 180, 240, 0) # 0 = unlimited
VSYNC = False # [F6] if True, the display is synchronized with the monitor's refresh rate, otherwise it runs at up to FPS
RETINA = False # [F7] native plot resolution on macOS; HUD/menu always use native pixels

LINE_IS_STEP = True # start with step lines ([P] toggles; smooth = straight lines between samples)

SPEC_OVERLAY_RATIO = 0.4 # height ratio of the spectrum overlay on the scope view

# ==========================
# CRT Beam Settings
# ==========================
# Shader-based analog oscilloscope look (crt_renderer.py, needs `pip install moderngl`).
# Pyglet owns the window and draws vector UI; ModernGL composites the traces.

USE_OPENGL = True # False = pyglet lines only (no CRT / GL line option)

# Trace renderer ([C] cycles, needs OpenGL):
#   "CRT"     - analog beam (settings below)
#   "GL Line" - plain lines drawn with OpenGL: free width, optional anti-aliasing
#   "Pyglet"  - plain 2D lines drawn by pyglet
TRACE_RENDERERS = ["CRT", "GL Line", "Pyglet"]
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


def update_scope_gain(peak, gain, dt, decay_time, base=SCOPE_GAIN):
    target = base
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

    ui.rect(surface, theme["bg"], (left, 0, width, surface.get_height()))
    ui.line(surface, theme["center"], (left, 0), (left, surface.get_height()))

    label_w = 26
    clip_h = 8
    bar_top = top + clip_h + 4
    bar_h = height - clip_h - 4 - 16 # leave room for L/R labels
    gap = 4
    bar_w = (width - label_w - gap * 3) // 2
    bars_left = left + label_w + gap

    for db in METER_TICKS:
        y = _meter_y(db, bar_top, bar_h)
        ui.line(surface, theme["grid"], (bars_left - 2, y), (left + width - gap, y))
        text = small_font.render(f"{db}", True, theme["text"])
        surface.blit(text, (bars_left - 4 - text.get_width(), y - text.get_height() // 2))

    warn_y = _meter_y(METER_WARN_DB, bar_top, bar_h)
    bottom = bar_top + bar_h - 1

    for ch, (name, color) in enumerate(zip(("L", "R"), colors)):
        x = bars_left + ch * (bar_w + gap)

        # background track
        ui.rect(surface, theme["grid"], (x, bar_top, bar_w, bar_h), 1)

        # peak (dim) and RMS (solid) bars
        for value, bar_color in (
            (meter.peak_db[ch], blend(color, theme["bg"], 0.35)),
            (meter.rms_db[ch], color),
        ):
            y = _meter_y(value, bar_top, bar_h)
            if y < bottom:
                ui.rect(surface, bar_color, (x + 1, y, bar_w - 2, bottom - y))
                if y < warn_y:
                    warn = METER_WARN_COLOR if bar_color == color else blend(METER_WARN_COLOR, theme["bg"], 0.35)
                    ui.rect(surface, warn, (x + 1, y, bar_w - 2, warn_y - y))

        # peak hold marker
        y = _meter_y(meter.hold_db[ch], bar_top, bar_h)
        if meter.hold_db[ch] > METER_DB_MIN:
            ui.line(surface, theme["text"], (x + 1, y), (x + bar_w - 2, y), 2)

        # clip indicator
        clip_color = METER_CLIP_COLOR if meter.clip_time[ch] > 0.0 else theme["grid"]
        ui.rect(surface, clip_color, (x, top, bar_w, clip_h))

        text = small_font.render(name, True, color)
        surface.blit(text, (x + (bar_w - text.get_width()) // 2, bar_top + bar_h + 2))


# ==========================
# Spectrum
# ==========================


SPEC_FREQ_TICKS = [
    (50, "50"), (100, "100"), (200, "200"), (500, "500"),
    (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k"), (20000, "20k"),
]


def _visible_freq_ticks(analyzer):
    if analyzer.mode == "Cepstrum":
        for ms in (1, 2, 5, 10, 20, 50, 100):
            if analyzer.q_min < ms / 1000 < analyzer.q_max:
                yield int(analyzer.quefrency_to_x(ms / 1000)), f"{ms}ms"
        return
    for freq, label in SPEC_FREQ_TICKS:
        if analyzer.f_min < freq < analyzer.f_max:
            yield int(analyzer.freq_to_x(freq)), label


def _db_ticks(analyzer, top, height):
    if analyzer.mode == "Cepstrum":
        for value in np.linspace(analyzer.db_min, analyzer.db_max, 5):
            yield int(top + (1.0 - analyzer.normalized(value)) * (height - 1)), value
        return
    db = math.ceil(analyzer.db_min / 20) * 20
    while db <= analyzer.db_max:
        yield int(top + (1.0 - analyzer.normalized(db)) * (height - 1)), db
        db += 20


def draw_spectrum_grid(surface, analyzer, theme, top, height):
    """Grid lines only; draw before the spectrum."""
    bottom = top + height
    for x, _ in _visible_freq_ticks(analyzer):
        ui.line(surface, theme["grid"], (x, top), (x, bottom))
    for y, _ in _db_ticks(analyzer, top, height):
        ui.line(surface, theme["grid"], (0, y), (analyzer.width, y))


def draw_spectrum_labels(surface, analyzer, theme, small_font, top, height, show_db):
    """Axis labels; draw after the spectrum so the fill does not hide them."""
    bottom = top + height
    for x, label in _visible_freq_ticks(analyzer):
        text = small_font.render(label, True, theme["text"])
        surface.blit(text, (x + 3, bottom - text.get_height() - 2))

    if show_db:
        for y, db in _db_ticks(analyzer, top, height):
            label = f"{db:g}" if analyzer.mode == "Cepstrum" else f"{db:.0f}dB"
            text = small_font.render(label, True, theme["text"])
            surface.blit(text, (analyzer.width - text.get_width() - 4, y + 2))


def draw_spectrum(surface, analyzer, theme, top, height, filled, line_queue=None):
    """
    line_queue: list to receive (points, color, scissor, width scale) for the
    OpenGL line renderer instead of drawing the line with pyglet (the fill is
    always drawn here).
    """
    points = analyzer.line_points(top, height, round_pixels=line_queue is None)

    if filled:
        bottom = top + height - 1
        polygon = np.vstack((points, [[analyzer.width - 1, bottom], [0, bottom]]))
        ui.polygon(surface, theme["fill"], polygon)

    if line_queue is not None:
        # overlay at half the trace width, like the pyglet 1 px / 2 px lines
        line_queue.append((points, theme["spectrum"], (0, top, analyzer.width, height), 1.0 if filled else 0.5))
    else:
        ui.lines(surface, theme["spectrum"], False, points, 1 if not filled else 2)


class Waterfall:
    """Scrolling spectrogram. Newest row at the top."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.history = np.zeros((height, width), dtype=np.uint8)
        self.head = 0
        self.image = ui.Image(width, height)
        self.pending_rows = 0.0

    def push(self, levels, lut, dt):
        self.pending_rows += dt * WATERFALL_ROWS_PER_SEC
        rows = int(self.pending_rows)
        if rows <= 0:
            return
        self.pending_rows -= rows
        rows = min(rows, self.height)

        self.image.pixels[rows:] = self.image.pixels[:-rows].copy()
        self.image.pixels[:rows] = lut[levels]
        self.image.changed()
        for _ in range(rows):
            self.head = (self.head + 1) % self.height
            self.history[self.head] = levels

    def resized(self, width, height):
        """A copy at another size: columns resampled, newest rows kept."""
        new = Waterfall(width, height)
        order = (self.head - np.arange(self.height)) % self.height
        # Resample columns on the current analysis axis.
        columns = np.minimum((np.arange(width) * self.width) // width, self.width - 1)
        rows = min(height, self.height)
        new.history[(-np.arange(rows)) % height] = self.history[order[:rows]][:, columns]
        new.pending_rows = self.pending_rows
        return new

    def rerender(self, lut):
        order = (self.head - np.arange(self.height)) % self.height
        image = lut[self.history[order]]  # (h, w, 3)
        self.image.pixels[:] = image
        self.image.changed()


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
# needed before the window opens (the swap interval is set at creation)
if isinstance(saved_settings.get("vsync"), bool):
    VSYNC = saved_settings["vsync"]
if isinstance(saved_settings.get("retina"), bool):
    RETINA = saved_settings["retina"]
if saved_settings.get("max_fps") in FPS_OPTIONS and type(saved_settings["max_fps"]) is int:
    FPS = saved_settings["max_fps"]


# ==========================
# Pyglet
# ==========================

from pyglet.window import key, mouse
import ui_renderer as ui


def initial_window_size():
    """Remember logical window size, constrained to the desktop and layout minimum."""
    width, height = WIDTH, HEIGHT
    size = saved_settings.get("window_size")
    if isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size):
        width, height = size
    try:
        desktop_w, desktop_h = desktop_size()
        width, height = min(width, desktop_w), min(height, desktop_h)
    except Exception:
        pass
    return max(width, MIN_WIDTH), max(height, MIN_HEIGHT)


DEFAULT_WINDOW_SIZE = (WIDTH, HEIGHT)
WIDTH, HEIGHT = initial_window_size()
try:
    window = create_window(WIDTH, HEIGHT, vsync=VSYNC, minimum_size=(MIN_WIDTH, MIN_HEIGHT))
except Exception:
    stream.close()
    pa.terminate()
    raise

font, small_font = ui.Font(18), ui.Font(13)
screen, overlay = ui.Canvas((WIDTH, HEIGHT)), ui.Canvas((WIDTH, HEIGHT))
pointer_position = (0, 0)
gl_renderer = None
if USE_OPENGL:
    try:
        import moderngl
        from crt_renderer import CRTRenderer

        gl_renderer = CRTRenderer(moderngl.create_context(), WIDTH, HEIGHT,
                                 drawable_size=render_size((WIDTH, HEIGHT),
                                                           window.get_framebuffer_size(), RETINA))
        print(f"OpenGL: {gl_renderer.ctx.info['GL_RENDERER']} ({gl_renderer.ctx.info['GL_VERSION']})")
    except Exception as exc:
        print(f"Warning: ModernGL unavailable, using pyglet lines ({exc})")


# ==========================
# View State
# ==========================

VIEW_SCOPE = 0     # waveform (+ optional spectrum overlay)
VIEW_SPECTRUM = 1  # spectrum (top) + waterfall (bottom)
VIEW_LISSAJOUS = 2 # XY plot in a square (L/R or autocorrelation)
VIEW_NAMES = ["Scope", "Spectrum", "Lissajous"]

view = VIEW_SCOPE
show_overlay = True
spectrum_mode = "Log"
spectrum_smoothing = "Off"
spectrum_db_range = 100
cepstrum_zoom = 1
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
lissajous_gain = LISSAJOUS_GAIN # follows the [A] AGC mode like scope_gain
lissajous_held_peak, lissajous_hold_remaining = 0.0, 0.0
detected_period = None
theme_index = 0
theme = THEMES[theme_index]

trace_renderer = TRACE_RENDERER if gl_renderer is not None else "Pyglet"
line_width = LINE_WIDTH
line_aa = LINE_AA
line_is_step = LINE_IS_STEP
spectrum_lines = [] # spectrum lines for the OpenGL line renderer, refilled every frame


def gl_line_queue():
    """
    Where plain lines (spectrum) go: a list drawn by the OpenGL line renderer, or
    None to draw them with pyglet (no ModernGL, or the Pyglet renderer selected).
    """
    if gl_renderer is None or trace_renderer == "Pyglet":
        return None
    return spectrum_lines
crt_beam_width = CRT_BEAM_WIDTH
crt_persistence = CRT_PERSISTENCE
crt_exposure = CRT_EXPOSURE
crt_gamma = CRT_GAMMA
crt_bloom = CRT_BLOOM
crt_scene = None         # (view, stereo, theme, ...) - the phosphor is erased when it changes
lissajous_beam_end = None # absolute sample index the Lissajous beam has been drawn up to

show_level_meter = SHOW_LEVEL_METER
show_hud = SHOW_HUD # status text + menu button ([F1] opens the menu either way)

SPLIT_Y = None
PLOT_WIDTH = None
overlay_height = overlay_top = None
analyzer = None
waterfall = None
MENU_BUTTON = None
hud_bottom = 0 # lowest pixel row of the status text (0 while it is hidden)

MENU_COLUMN_WIDTH = 209
MENU_COLUMN_GAP = 8
MENU_ROW_HEIGHT = 21
MENU_PADDING = 10
MENU_HINT = "Shift reverses supported controls or changes XY delay by x2 / x0.5."


def layout_menu():
    global MENU_BUTTON
    MENU_BUTTON = ui.Rect(PLOT_WIDTH - 122, 6, 112, 24)


def update_layout():
    """
    Lay everything out for the current window size: plot area = window minus the
    level meter panel on the right. Rebuilds what is sized to it; the waterfall
    history is resampled so it survives a meter toggle or a window resize.
    """
    global PLOT_WIDTH, SPLIT_Y, overlay_height, overlay_top, analyzer, waterfall, crt_scene
    width = max(1, WIDTH - (METER_PANEL_WIDTH if show_level_meter else 0))
    split_y = HEIGHT // 2
    overlay_height = int(HEIGHT * SPEC_OVERLAY_RATIO)
    overlay_top = HEIGHT - overlay_height
    waterfall_size = (width, HEIGHT - split_y)
    changed = (width, split_y) != (PLOT_WIDTH, SPLIT_Y) or (
        waterfall is not None and (waterfall.width, waterfall.height) != waterfall_size
    )
    PLOT_WIDTH, SPLIT_Y = width, split_y
    layout_menu()
    if not changed:
        return
    if analyzer is None or analyzer.width != PLOT_WIDTH:
        analyzer = create_spectrum_analyzer()
    old_waterfall = waterfall
    waterfall = Waterfall(*waterfall_size) if waterfall is None else waterfall.resized(*waterfall_size)
    if old_waterfall is not None:
        old_waterfall.image.delete()
    waterfall.rerender(theme["lut"])
    crt_scene = None # the phosphor was drawn for the old layout


def create_spectrum_analyzer():
    return SpectrumAnalyzer(
        FFT_SIZE, RATE, PLOT_WIDTH, SPEC_F_MIN, SPEC_F_MAX,
        -spectrum_db_range, SPEC_DB_MAX, spectrum_mode, SPEC_FALL_DB_PER_SEC,
        smoothing=spectrum_smoothing, cepstrum_zoom=cepstrum_zoom,
    )


def set_spectrum_options(mode=None, smoothing=None, db_range=None, zoom=None):
    """Apply valid options and reset history whose axis or averaging has changed."""
    global spectrum_mode, spectrum_smoothing, spectrum_db_range, cepstrum_zoom
    global analyzer, crt_scene
    previous = (spectrum_mode, spectrum_smoothing, spectrum_db_range, cepstrum_zoom)
    if mode in SPECTRUM_MODES:
        spectrum_mode = mode
    if isinstance(smoothing, str) and smoothing in SPECTRUM_SMOOTHING:
        spectrum_smoothing = smoothing
    if type(db_range) in (int, float) and db_range in SPECTRUM_DB_RANGES:
        spectrum_db_range = db_range
    if type(zoom) in (int, float) and zoom in CEPSTRUM_ZOOMS:
        cepstrum_zoom = zoom
    if previous == (spectrum_mode, spectrum_smoothing, spectrum_db_range, cepstrum_zoom):
        return
    analyzer = create_spectrum_analyzer()
    # Keep the image/texture: retained sprites may still refer to it this frame.
    waterfall.history.fill(0)
    waterfall.head = 0
    waterfall.pending_rows = 0.0
    waterfall.rerender(theme["lut"])
    crt_scene = None


def set_spectrum_mode(mode):
    set_spectrum_options(mode=mode)


def spectrum_range_label():
    return f"Zoom: {cepstrum_zoom:g}x" if spectrum_mode == "Cepstrum" else f"Range: {spectrum_db_range:g}dB"


def apply_window_size(width, height):
    """Keep layout in logical points and query native pixels independently."""
    global WIDTH, HEIGHT
    width, height = max(1, int(width)), max(1, int(height))
    if gl_renderer is not None:
        gl_renderer.resize(width, height, drawable_size=render_size(
            (width, height), window.get_framebuffer_size(), RETINA))
    if (width, height) == (WIDTH, HEIGHT):
        return
    WIDTH, HEIGHT = width, height
    screen.width, screen.height = WIDTH, HEIGHT
    overlay.width, overlay.height = WIDTH, HEIGHT
    update_layout()


def content_top():
    """First pixel row below the status text (views keep their labels / plots under it)."""
    return max(LISSAJOUS_TOP, hud_bottom + 4) if show_hud else LISSAJOUS_TOP


update_layout()

meter = LevelMeter(2)


def set_theme(index):
    global theme_index, theme
    theme_index = index % len(THEMES)
    theme = THEMES[theme_index]
    waterfall.rerender(theme["lut"])




def apply_settings(data):
    """Apply saved values, ignoring anything missing or invalid."""
    global view, show_overlay, trigger_mode, stereo_mode, lissajous_mode, lissajous_delay
    global lissajous_swap, lissajous_invert
    global show_scope_graticule, show_scope_dbfs_scale, scope_samples
    global scope_agc_mode, show_scope_agc_dbfs_scale, scope_agc_peak_hold
    global trace_renderer, line_width, line_aa, line_is_step
    global crt_beam_width, crt_persistence, crt_exposure, crt_gamma, crt_bloom
    global show_level_meter, show_hud

    def number(key, lo, hi):
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi:
            return value
        return None

    if data.get("view") in VIEW_NAMES:
        view = VIEW_NAMES.index(data["view"])
    set_spectrum_options(
        mode=data.get("spectrum_mode"), smoothing=data.get("spectrum_smoothing"),
        db_range=data.get("spectrum_db_range"), zoom=data.get("cepstrum_zoom"),
    )
    if isinstance(data.get("show_overlay"), bool):
        show_overlay = data["show_overlay"]
    if isinstance(data.get("line_is_step"), bool):
        line_is_step = data["line_is_step"]
    if isinstance(data.get("show_level_meter"), bool):
        show_level_meter = data["show_level_meter"]
    if isinstance(data.get("show_hud"), bool):
        show_hud = data["show_hud"]
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
        renderer_name = data.get("trace_renderer")
        if renderer_name == "Pygame":
            renderer_name = "Pyglet"
        if renderer_name in TRACE_RENDERERS:
            trace_renderer = renderer_name
        elif isinstance(data.get("crt_enabled"), bool): # older settings files
            trace_renderer = "CRT" if data["crt_enabled"] else "Pyglet"
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
    if data.get("trace_renderer") == "Pygame":
        data["trace_renderer"] = "Pyglet"
    data.pop("scope_agc", None) # superseded by scope_agc_mode
    data.update({
        "view": VIEW_NAMES[view],
        "show_overlay": show_overlay,
        "spectrum_mode": spectrum_mode,
        "spectrum_smoothing": spectrum_smoothing,
        "spectrum_db_range": spectrum_db_range,
        "cepstrum_zoom": cepstrum_zoom,
        "show_level_meter": show_level_meter,
        "show_hud": show_hud,
        "show_scope_graticule": show_scope_graticule,
        "line_is_step": line_is_step,
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
        "window_size": [WIDTH, HEIGHT],
        "max_fps": FPS,
        "vsync": VSYNC,
        "retina": RETINA,
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
update_layout() # the saved meter visibility may change the plot width


def draw_trace(waveform, color, top, height):
    points = build_waveform_points(
        waveform,
        PLOT_WIDTH,
        height,
        scope_gain,
        line_is_step,
        top,
    )

    if points is not None:
        ui.lines(screen, color, False, points, 2)


def draw_scope_graticule(lanes):
    right, bottom = PLOT_WIDTH - 1, HEIGHT - 1
    center_x = round(right / 2)
    for division in range(11):
        x = round(right * division / 10)
        ui.line(screen, theme["grid"], (x, 0), (x, bottom))
    ui.line(screen, theme["center"], (center_x, 0), (center_x, bottom))

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
                ui.line(screen, theme["grid"], (0, y), (right, y))
        ui.line(screen, theme["center"], (0, center_y), (right, center_y))

        for subdivision in range(1, 50):
            x = round(right * subdivision / 50)
            tick = 5 if subdivision % 5 == 0 else 2
            ui.line(screen, theme["center"], (x, center_y - tick), (x, center_y + tick))
        half_ticks = divisions * 5 // 2
        for subdivision in range(1 - half_ticks, half_ticks):
            y = round(top + span * (0.5 + subdivision * spacing / (divisions * 5)))
            if not top <= y <= lane_bottom:
                continue
            tick = 5 if subdivision % 5 == 0 else 2
            ui.line(screen, theme["center"], (center_x - tick, y), (center_x + tick, y))

    for top, _ in lanes[1:]:
        ui.line(screen, theme["center"], (0, top), (right, top))


def draw_scope_dbfs_scale(top, height):
    scope_agc = scope_agc_mode != "Off"
    x = PLOT_WIDTH // 4
    bottom = top + height - 1
    upper = max(top, round(waveform_y(1, top, height, scope_gain)))
    lower = min(bottom, round(waveform_y(-1, top, height, scope_gain)))
    ui.line(screen, theme["center"], (x, upper), (x, lower))

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
            ui.line(screen, theme["center"], (x - 4, y), (x + 4, y))
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
        draw_spectrum_labels(overlay, analyzer, theme, small_font, overlay_top, overlay_height, False)

    show_dbfs_scale = show_scope_dbfs_scale if scope_agc_mode == "Off" else show_scope_agc_dbfs_scale
    if show_dbfs_scale:
        for top, height in lanes:
            draw_scope_dbfs_scale(top, height)

    if trace_renderer == "Pyglet": # otherwise drawn on the GPU in render_gl()
        for waveform, color, top, height in traces:
            draw_trace(waveform, color, top, height)

    # channel labels
    if stereo_mode != "Mix":
        for i, (name, color) in enumerate((("L", theme["trace_l"]), ("R", theme["trace_r"]))):
            label = font.render(name, True, color)
            if stereo_mode == "Split":
                pos = (PLOT_WIDTH - label.get_width() - 8, i * SPLIT_Y + (8 if i else content_top()))
            else:
                pos = (PLOT_WIDTH - 24 * (2 - i) - 4, content_top())
            overlay.blit(label, pos)


def draw_spectrum_view():
    draw_spectrum_grid(screen, analyzer, theme, 0, SPLIT_Y)
    draw_spectrum(screen, analyzer, theme, 0, SPLIT_Y, True, gl_line_queue())
    draw_spectrum_labels(overlay, analyzer, theme, small_font, 0, SPLIT_Y, True)

    screen.blit(waterfall.image, (0, SPLIT_Y))
    ui.line(screen, theme["center"], (0, SPLIT_Y), (PLOT_WIDTH, SPLIT_Y), 2)


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
    top = content_top()
    side = max(1, min(PLOT_WIDTH - 16, HEIGHT - top - 8))
    left = (PLOT_WIDTH - side) // 2
    return left, top, side


def lissajous_signals(left, right, mix):
    """The newest LISSAJOUS_POINTS (x, y) samples of the XY figure, before orientation."""
    if effective_lissajous_mode() == "Auto":
        # X = s[i], Y = s[i + delay]: newest sample pairs that both exist
        count = min(LISSAJOUS_POINTS, len(mix) - lissajous_delay)
        return mix[-count - lissajous_delay:-lissajous_delay], mix[-count:]
    return left[-LISSAJOUS_POINTS:], right[-LISSAJOUS_POINTS:]


def lissajous_peak(left, right, mix):
    """Largest |x| or |y| in the drawn window (the AGC keeps it inside the square)."""
    x, y = lissajous_signals(left, right, mix)
    return max(float(np.max(np.abs(x), initial=0.0)), float(np.max(np.abs(y), initial=0.0)))


def build_lissajous_points(left, right, mix, round_pixels=True):
    """(N, 2) screen points of the XY figure (int32, or float32 subpixel)."""

    x, y = orient_lissajous(*lissajous_signals(left, right, mix))

    rect_left, rect_top, side = lissajous_rect()
    half = side / 2
    scale = half * lissajous_gain
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
        ui.line(screen, theme["grid"], (rect_left + offset, rect_top), (rect_left + offset, bottom))
        ui.line(screen, theme["grid"], (rect_left, rect_top + offset), (right, rect_top + offset))
    ui.line(screen, theme["grid"], (rect_left, bottom), (right, rect_top))
    ui.line(screen, theme["grid"], (rect_left, rect_top), (right, bottom))
    ui.line(screen, theme["center"], (mid_x, rect_top), (mid_x, bottom), 2)
    ui.line(screen, theme["center"], (rect_left, mid_y), (right, mid_y), 2)
    ui.rect(screen, theme["center"], (rect_left, rect_top, side, side), 1)

    if points is not None and len(points) > 1:
        ui.lines(screen, theme["trace"], False, points, 1)

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
    overlay.blit(x_label, (right - x_label.get_width() - 4, mid_y + 4))
    overlay.blit(y_label, (mid_x + 6, rect_top + 4))


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
    scale = half * lissajous_gain
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
        if line_is_step:
            points = build_waveform_points(
                waveform, PLOT_WIDTH, height, scope_gain, True, top, round_pixels=False,
            )
        elif len(waveform) > 1:
            # every sample at its subpixel position (no decimation), so the
            # anti-aliasing follows the real slope of the signal
            x = np.linspace(0, PLOT_WIDTH - 1, len(waveform), dtype=np.float32)
            points = np.column_stack((x, waveform_y(waveform, top, height, scope_gain)))
        else:
            points = None
        if points is not None:
            gl_renderer.draw_line(points, color01(color), scissor=(0, top, PLOT_WIDTH, height))


def line_lissajous(left, right, mix):
    rect_left, rect_top, side = lissajous_rect()
    points = build_lissajous_points(left, right, mix, round_pixels=False)
    gl_renderer.draw_line(points, color01(theme["trace"]), scissor=(rect_left, rect_top, side, side))


def render_gl(traces, left, right, mix, end_abs, dt):
    """Draw the UI target, add traces, and composite to the native framebuffer."""

    global crt_scene, lissajous_beam_end

    r = gl_renderer
    screen.render(window, target=r.ui_fbo, framebuffer_size=(r.width, r.height))
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
        r.present(False, framebuffer_size=window.get_framebuffer_size())
        return

    scene = (view, stereo_mode, theme_index, effective_lissajous_mode(), lissajous_swap,
             lissajous_invert, scope_agc_mode != "Off",
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
    r.present(True, framebuffer_size=window.get_framebuffer_size())


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


def scale_controls():
    return (
        (key.J, "J", "Grid", show_scope_graticule),
        (key.K, "K", "Fixed dBFS", show_scope_dbfs_scale),
        (key.D, "D", "AGC dBFS", show_scope_agc_dbfs_scale),
    )


def scope_time_span():
    return f"{scope_samples * 1000 / RATE:.2f} ms"


def menu_items():
    agc = scope_agc_mode
    hold = format_seconds(SCOPE_AGC_HOLD_TIME) if scope_agc_peak_hold else "Off"
    groups = [
        ("Display", (
            (key.TAB, "Tab", f"View: {VIEW_NAMES[view]}"),
            (key.T, "T", f"Theme: {theme['name']}"),
            (key.S, "S", f"Spectrum overlay: {'On' if show_overlay else 'Off'}"),
            (key.R, "R", f"Trigger: {trigger_mode}"),
            (key.M, "M", f"Stereo: {stereo_mode}"),
            (key.F2, "F2", f"Level meter: {'On' if show_level_meter else 'Off'}"),
            (key.F3, "F3", f"Status & button: {'On' if show_hud else 'Off'}"),
            (key.F4, "F4", "Reset size ({}x{})".format(*DEFAULT_WINDOW_SIZE)),
            (key.F5, "F5", f"Max FPS: {format_fps(FPS)}{' (VSync)' if VSYNC else ''}"),
            (key.F6, "F6", f"VSync: {'On' if VSYNC else 'Off'}"),
            *([(key.F7, "F7", f"Retina resolution: {'On' if RETINA else 'Off'}")]
              if sys.platform == 'darwin' and gl_renderer is not None else []),
        )),
        ("Spectrum", (
            (key.Q, "Q", f"Mode: {spectrum_mode}"),
            (key.U, "U", f"Smoothing: {spectrum_smoothing}"),
            (key.Y, "Y", spectrum_range_label()),
        )),
        ("Scope", (
            (key.X, "X", f"Time span: {scope_time_span()}"),
            (key.P, "P", f"Line: {'Step' if line_is_step else 'Smooth'}"),
            (key.A, "A", f"AGC: {agc}"),
            (key.H, "H", f"AGC peak hold: {hold}"),
            *((key, shortcut, f"{name}: {'On' if enabled else 'Off'}")
              for key, shortcut, name, enabled in scale_controls()),
        )),
        ("Lissajous", (
            (key.L, "L", f"XY mode: {lissajous_mode}"),
            (key.W, "W", f"Swap X/Y: {'On' if lissajous_swap else 'Off'}"),
            (key.I, "I", f"Invert: {lissajous_invert}"),
            (key.A, "A", f"AGC: {agc}"),
            (key.COMMA, ",", "Decrease XY delay"),
            (key.PERIOD, ".", "Increase XY delay"),
        )),
    ]
    if gl_renderer is not None:
        items = [(key.C, "C", f"Renderer: {trace_renderer}")]
        if trace_renderer == "CRT":
            items += [
                (key.E, "E", f"Exposure: {crt_exposure:.2f}"),
                (key.F, "F", f"Focus: {crt_beam_width:.1f}px"),
                (key.G, "G", f"Afterglow: {format_seconds(crt_persistence)}"),
                (key.B, "B", f"Bloom: {'On' if crt_bloom else 'Off'}"),
                (key.V, "V", f"Gamma: {crt_gamma:.1f}"),
            ]
        if trace_renderer != "Pyglet": # GL lines: traces in GL Line mode, spectrum in both
            items += [
                (key.N, "N", f"Line width: {format_line_width(line_width)}"),
                (key.O, "O", f"Anti-aliasing: {'On' if line_aa else 'Off'}"),
            ]
        groups.append(("Renderer", tuple(items)))
    return groups


def wrap_words(text, text_font, max_width, separator=" "):
    """Greedy wrap at `separator` so every line fits max_width (a single piece may not)."""
    lines, current = [], ""
    for piece in text.split(separator):
        candidate = piece if not current else current + separator + piece
        if current and text_font.size(candidate)[0] > max_width:
            lines.append(current)
            current = piece
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def menu_layout():
    """
    Place the menu groups for the current window: as many columns as fit, the
    rest wrapped into further bands below. Returns (panel rect, titles, rows,
    hint lines) with titles = [(text, (x, y))] and rows = [(key, label, rect)].
    """
    groups = menu_items()
    available = WIDTH - 2 * MENU_PADDING # the panel may cover the level meter
    step = MENU_COLUMN_WIDTH + MENU_COLUMN_GAP
    columns = max(1, min(len(groups), (available - 2 * MENU_PADDING + MENU_COLUMN_GAP) // step))
    column_width = min(MENU_COLUMN_WIDTH, available - 2 * MENU_PADDING)
    width = 2 * MENU_PADDING + columns * (column_width + MENU_COLUMN_GAP) - MENU_COLUMN_GAP
    hint = wrap_words(MENU_HINT, small_font, width - 2 * MENU_PADDING)

    # right-aligned to the plot like the button, shifted left only as far as needed
    left = min(max(MENU_PADDING, PLOT_WIDTH - MENU_PADDING - width), WIDTH - MENU_PADDING - width)
    top = 36 if show_hud else 6

    titles, rows = [], []
    y = top
    for band in range(0, len(groups), columns):
        band_groups = groups[band:band + columns]
        for column, (title, items) in enumerate(band_groups):
            x = left + MENU_PADDING + column * (column_width + MENU_COLUMN_GAP)
            titles.append((title, (x + 4, y + 7)))
            for row, (key, shortcut, text) in enumerate(items):
                rect = ui.Rect(x, y + 32 + MENU_ROW_HEIGHT * row, column_width, MENU_ROW_HEIGHT - 1)
                rows.append((key, f"[{shortcut}] {text}", rect))
        y += 32 + MENU_ROW_HEIGHT * max(len(items) for _, items in band_groups) + 4
    height = y - top + 12 + small_font.get_linesize() * len(hint)
    # a window too short for the whole panel cuts it off (rows below stay unreachable)
    panel = ui.Rect(left, top, width, max(1, min(height, HEIGHT - top - 4)))
    rows = [row for row in rows if panel.contains(row[2])]
    return panel, titles, rows, hint


def menu_key_at(pos):
    """(clicked inside the panel, menu key under pos or None)."""
    panel, _, rows, _ = menu_layout()
    if not panel.collidepoint(pos):
        return False, None
    return True, next((key for key, _, rect in rows if rect.collidepoint(pos)), None)


def draw_button(rect, text):
    ui.rect(overlay, theme["bg"], rect)
    ui.rect(overlay, theme["center"], rect, 1)
    label = small_font.render(text, True, theme["text"])
    overlay.blit(label, label.get_rect(center=rect.center))


def draw_menu():
    if show_hud:
        draw_button(MENU_BUTTON, "Close [F1]" if menu_open else "Menu [F1]")
    if not menu_open:
        return

    panel, titles, rows, hint = menu_layout()
    ui.rect(overlay, theme["bg"], panel)
    overlay.set_clip(panel)
    pointer = pointer_position
    for title, pos in titles:
        overlay.blit(font.render(title, True, theme["text"]), pos)
    for _, label, rect in rows:
        if rect.collidepoint(pointer):
            ui.rect(overlay, theme["grid"], rect)
        overlay.blit(small_font.render(label, True, theme["text"]), (rect.x + 4, rect.y + 2))
    line_height = small_font.get_linesize()
    y = panel.bottom - 8 - line_height * len(hint)
    for line in hint:
        overlay.blit(small_font.render(line, True, theme["text"]), (panel.x + 14, y))
        y += line_height
    overlay.set_clip(None)
    ui.rect(overlay, theme["center"], panel, 1)


def hud_texts(detected_period):
    """Status lines, top to bottom, as (font, text); fields are separated by 3 spaces."""
    texts = [(font, f"Rate:{RATE}Hz   FFT:{FFT_SIZE}   FPS:{frame_fps:.1f}   Max:{'VSync' if VSYNC else format_fps(FPS)}")]

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
        agc_status = scope_agc_mode
        if scope_agc_mode != "Off":
            agc_status += f" x{lissajous_gain / LISSAJOUS_GAIN:.2f}"
        help_text += f"   AGC:{agc_status}"
    else:
        help_text = f"View:{VIEW_NAMES[view]}   Spectrum:{spectrum_mode}   Smooth:{spectrum_smoothing}   {spectrum_range_label()}   Overlay:{'ON' if show_overlay else 'OFF'}   Theme:{theme['name']}   Trigger:{trigger_info}   Stereo:{stereo_info}"
        if view == VIEW_SCOPE:
            help_text += f"   Time span: {scope_time_span()}"
    texts.append((small_font, help_text))

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
            renderer_text = "Renderer:Pyglet"
        texts.append((small_font, renderer_text))

    if view == VIEW_SCOPE:
        agc_status = scope_agc_mode
        if scope_agc_mode != "Off":
            agc_status += f" x{scope_gain / SCOPE_GAIN:.2f}"
        hold_status = format_seconds(SCOPE_AGC_HOLD_TIME) if scope_agc_peak_hold else "OFF"
        scale_status = "   ".join(
            f"{name}:{'On' if enabled else 'Off'}" for _, _, name, enabled in scale_controls()
        )
        texts.append((small_font, (
            f"AGC:{agc_status}   Peak hold:{hold_status}   {scale_status}"
            f"   Line:{'Step' if line_is_step else 'Smooth'}"
        )))
    return texts


def layout_hud(texts):
    """
    Stack the status lines from the top left, each wrapped at its field
    separators to the plot width (the first line also keeps clear of the menu
    button). Returns [(font, text, (x, y))].
    """
    lines = []
    y = 10
    for index, (text_font, text) in enumerate(texts):
        right = MENU_BUTTON.x - 10 if index == 0 else PLOT_WIDTH - 10
        for line in wrap_words(text, text_font, right - 10, separator="   "):
            lines.append((text_font, line, (10, y)))
            y += text_font.get_linesize()
        if index == 0:
            y = max(y, 32)
    return lines


def format_fps(value):
    return "Unlimited" if value == 0 else str(value)


def schedule_frames():
    pyglet.clock.unschedule(draw_frame)
    if VSYNC or FPS == 0:
        pyglet.clock.schedule(draw_frame)
    else:
        pyglet.clock.schedule_interval(draw_frame, 1.0 / FPS)


def set_vsync(enabled):
    global VSYNC
    window.set_vsync(enabled)
    VSYNC = enabled
    schedule_frames()


def reset_window_size():
    window.set_fullscreen(False)
    window.set_size(*DEFAULT_WINDOW_SIZE)
    apply_window_size(*window.get_size())


def set_retina(enabled):
    global RETINA, crt_scene
    if sys.platform == 'darwin' and gl_renderer is not None:
        RETINA = enabled
        apply_window_size(*window.get_size())
        crt_scene = None


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
print("  Keys: [Tab]/[1]/[2]/[3] view  [S] spectrum overlay  [Q]/[Shift+Q] spectrum mode  [T]/[Shift+T] theme  [R] trigger mode  [M] stereo mode")
print("        [U]/[Shift+U] spectrum smoothing  [Y]/[Shift+Y] spectrum dB range / cepstrum zoom")
print("        [F1] Menu  [F2] Level meter  [F3] Status text & menu button  [F4] Reset window size")
print("        [F5] Max FPS (+Shift: back)  [F6] VSync")
if sys.platform == 'darwin' and gl_renderer is not None:
    print("        [F7] Retina resolution")
print("        [A] AGC Off/High/Med/Low (Scope & Lissajous)  [D] AGC dBFS labels  [H] AGC peak hold")
print("        [J] Scope grid  [K] Fixed-gain dBFS labels")
print("        [X]/[Shift+X] Scope time span  [P] Scope step / smooth line")
print("        [W] Lissajous swap X/Y  [I] invert Off/X/Y/XY (+Shift: back)")
print("        [L] Lissajous L/R <-> Auto  [,] [.] (or [ and ]) autocorrelation delay -1 / +1 (+Shift: x0.5 / x2)")
if gl_renderer is not None:
    print("        [C] renderer CRT / GL Line / Pyglet (+Shift: back)")
    print("        CRT:     [E] exposure  [F] focus  [G] afterglow  [V] gamma (+Shift: down)  [B] bloom")
    print("        Lines:   [N] line width (+Shift: thinner)  [O] anti-aliasing (GL Line traces, spectrum)")

frame_fps = 0.0


def handle_key(symbol, modifiers):
    global menu_open, show_level_meter, show_hud, FPS, view, lissajous_mode
    global lissajous_delay, lissajous_swap, lissajous_invert, show_overlay, trigger_mode
    global stereo_mode, scope_gain, scope_held_peak, scope_hold_remaining, scope_samples
    global scope_agc_mode, lissajous_gain, show_scope_agc_dbfs_scale, show_scope_graticule
    global show_scope_dbfs_scale, line_is_step, scope_agc_peak_hold, trace_renderer
    global line_width, line_aa, crt_persistence, crt_beam_width, crt_exposure, crt_bloom, crt_gamma
    old_fps = FPS
    if symbol == key.F1:
        menu_open = not menu_open
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.ESCAPE and menu_open:
        menu_open = False
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.F2:
        show_level_meter = not show_level_meter
        update_layout()
    elif symbol == key.F3:
        show_hud = not show_hud
        layout_menu()
    elif symbol == key.F4:
        reset_window_size()
    elif symbol == key.F5:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        FPS = FPS_OPTIONS[(FPS_OPTIONS.index(FPS) + step) % len(FPS_OPTIONS)] \
            if FPS in FPS_OPTIONS else FPS_OPTIONS[0]
    elif symbol == key.F6:
        set_vsync(not VSYNC)
    elif symbol == key.F7:
        set_retina(not RETINA)
    elif symbol == key.TAB:
        view = (view + 1) % len(VIEW_NAMES)
    elif symbol == key._1:
        view = VIEW_SCOPE
    elif symbol == key._2:
        view = VIEW_SPECTRUM
    elif symbol == key._3:
        view = VIEW_LISSAJOUS
    elif symbol == key.L:
        lissajous_mode = cycle(LISSAJOUS_MODES, lissajous_mode)
    elif symbol in (key.BRACKETLEFT, key.BRACKETRIGHT, key.COMMA, key.PERIOD):
        # autocorrelation delay: x2 / x0.5 with Shift, otherwise +-1 sample.
        # [ ] or , . (the latter sit at the same place on US and JIS keyboards)
        up = symbol in (key.BRACKETRIGHT, key.PERIOD)
        if view == VIEW_LISSAJOUS and lissajous_mode != "Auto":
            lissajous_mode = "Auto" # the delay only exists in Auto mode
        if modifiers & key.MOD_SHIFT:
            lissajous_delay = lissajous_delay * 2 if up else lissajous_delay // 2
        else:
            lissajous_delay += 1 if up else -1
        lissajous_delay = min(max(lissajous_delay, LISSAJOUS_DELAY_MIN), LISSAJOUS_DELAY_MAX)
    elif symbol == key.W:
        lissajous_swap = not lissajous_swap
    elif symbol == key.I:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        index = LISSAJOUS_INVERTS.index(lissajous_invert) + step
        lissajous_invert = LISSAJOUS_INVERTS[index % len(LISSAJOUS_INVERTS)]
    elif symbol == key.Q:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        set_spectrum_mode(SPECTRUM_MODES[(SPECTRUM_MODES.index(spectrum_mode) + step) % len(SPECTRUM_MODES)])
    elif symbol == key.U:
        options = tuple(SPECTRUM_SMOOTHING)
        step = -1 if modifiers & key.MOD_SHIFT else 1
        set_spectrum_options(smoothing=options[(options.index(spectrum_smoothing) + step) % len(options)])
    elif symbol == key.Y:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        if spectrum_mode == "Cepstrum":
            index = (CEPSTRUM_ZOOMS.index(cepstrum_zoom) + step) % len(CEPSTRUM_ZOOMS)
            set_spectrum_options(zoom=CEPSTRUM_ZOOMS[index])
        else:
            index = (SPECTRUM_DB_RANGES.index(spectrum_db_range) + step) % len(SPECTRUM_DB_RANGES)
            set_spectrum_options(db_range=SPECTRUM_DB_RANGES[index])
    elif symbol == key.S:
        show_overlay = not show_overlay
    elif symbol == key.T:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        set_theme(theme_index + step)
    elif symbol == key.R:
        trigger_mode = cycle(TRIGGER_MODES, trigger_mode)
    elif symbol == key.M:
        stereo_mode = cycle(STEREO_MODES, stereo_mode)
        scope_gain = SCOPE_GAIN
        scope_held_peak, scope_hold_remaining = 0.0, 0.0
    elif symbol == key.X:
        step = -1 if modifiers & key.MOD_SHIFT else 1
        index = SCOPE_SAMPLE_COUNTS.index(scope_samples) + step
        scope_samples = SCOPE_SAMPLE_COUNTS[index % len(SCOPE_SAMPLE_COUNTS)]
        for triggers in (hybrid_triggers, corr_triggers, legacy_triggers, diff_triggers):
            triggers.clear()
    elif symbol == key.A:
        scope_agc_mode = cycle(SCOPE_AGC_MODES, scope_agc_mode)
        if scope_agc_mode == "Off":
            scope_gain = SCOPE_GAIN
            lissajous_gain = LISSAJOUS_GAIN
    elif symbol == key.D:
        show_scope_agc_dbfs_scale = not show_scope_agc_dbfs_scale
    elif symbol == key.J:
        show_scope_graticule = not show_scope_graticule
    elif symbol == key.K:
        show_scope_dbfs_scale = not show_scope_dbfs_scale
    elif symbol == key.P:
        line_is_step = not line_is_step
    elif symbol == key.H:
        scope_agc_peak_hold = not scope_agc_peak_hold
        scope_held_peak, scope_hold_remaining = 0.0, 0.0
    elif gl_renderer is not None:
        # renderer / CRT beam / GL line controls (Shift = the other direction)
        step = -1 if modifiers & key.MOD_SHIFT else 1
        if symbol == key.C:
            index = TRACE_RENDERERS.index(trace_renderer) + step
            trace_renderer = TRACE_RENDERERS[index % len(TRACE_RENDERERS)]
        elif symbol == key.N:
            line_width = step_option(LINE_WIDTHS, line_width, step)
        elif symbol == key.O:
            line_aa = not line_aa
        elif symbol == key.G:
            crt_persistence = step_option(CRT_PERSISTENCES, crt_persistence, step)
        elif symbol == key.F:
            crt_beam_width = step_option(CRT_BEAM_WIDTHS, crt_beam_width, step)
        elif symbol == key.E:
            crt_exposure *= CRT_EXPOSURE_STEP ** step
            crt_exposure = min(max(crt_exposure, 0.05), 100.0)
        elif symbol == key.B:
            crt_bloom = not crt_bloom
        elif symbol == key.V:
            crt_gamma = step_option(CRT_GAMMAS, crt_gamma, step)

    if FPS != old_fps:
        schedule_frames()
    save_settings_if_changed()
    return pyglet.event.EVENT_HANDLED


@window.event
def on_key_press(symbol, modifiers):
    return handle_key(symbol, modifiers)


@window.event
def on_activate():
    if sys.platform == 'darwin':
        activate_window(window)


@window.event
def on_show():
    if sys.platform == 'darwin':
        activate_window(window)


@window.event
def on_mouse_motion(x, y, dx, dy):
    global pointer_position
    pointer_position = mouse_position(x, y, HEIGHT)


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    on_mouse_motion(x, y, dx, dy)


@window.event
def on_mouse_press(x, y, button, modifiers):
    global menu_open
    if sys.platform == 'darwin':
        activate_window(window)
    on_mouse_motion(x, y, 0, 0)
    if button != mouse.LEFT:
        return
    if show_hud and MENU_BUTTON.collidepoint(pointer_position):
        menu_open = not menu_open
    elif menu_open:
        inside, symbol = menu_key_at(pointer_position)
        if symbol is not None:
            handle_key(symbol, modifiers)
        elif not inside:
            menu_open = False


@window.event
def on_resize(width, height):
    apply_window_size(width, height)
    save_settings_if_changed()


@window.event
def on_close():
    save_settings_if_changed()
    pyglet.app.exit()
    return pyglet.event.EVENT_HANDLED


def draw_frame(dt):
    global scope_gain, scope_held_peak, scope_hold_remaining, lissajous_gain
    global lissajous_held_peak, lissajous_hold_remaining, hud_bottom, detected_period, frame_fps
    window.switch_to()
    apply_window_size(*window.get_size())
    if dt > 0:
        frame_fps = 0.9 * frame_fps + 0.1 / dt if frame_fps else 1 / dt
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
    overlay.fill((0, 0, 0, 0))

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

    if view == VIEW_LISSAJOUS and scope_agc_mode != "Off":
        peak = lissajous_peak(left, right, mix)
        if scope_agc_peak_hold:
            lissajous_held_peak, lissajous_hold_remaining = update_scope_peak_hold(
                peak, lissajous_held_peak, lissajous_hold_remaining, dt,
            )
            peak = lissajous_held_peak
        lissajous_gain = update_scope_gain(
            peak, lissajous_gain, dt, SCOPE_AGC_DECAY_TIMES[scope_agc_mode], LISSAJOUS_GAIN,
        )
    else:
        lissajous_gain = LISSAJOUS_GAIN
        lissajous_held_peak, lissajous_hold_remaining = 0.0, 0.0

    hud_lines = layout_hud(hud_texts(detected_period)) if show_hud else []
    hud_bottom = max((pos[1] + text_font.get_height() for text_font, _, pos in hud_lines), default=0)

    if view == VIEW_SCOPE:
        draw_scope(traces)
    elif view == VIEW_LISSAJOUS:
        draw_lissajous(build_lissajous_points(left, right, mix) if trace_renderer == "Pyglet" else None)
    else:
        draw_spectrum_view()

    if show_level_meter:
        draw_level_meter(
            screen, meter, theme, small_font,
            PLOT_WIDTH, 10, METER_PANEL_WIDTH, HEIGHT - 20,
            (theme["trace_l"], theme["trace_r"]),
        )

    #
    # Text
    #

    for text_font, text, pos in hud_lines:
        overlay.blit(text_font.render(text, True, theme["text"]), pos)

    draw_menu()
    if gl_renderer is not None:
        render_gl(traces, left, right, mix, end_abs, dt)
    else:
        screen.render(window)
    overlay.render(window, clear=False)
    window.flip()



window.set_visible(True)
activate_window(window)
schedule_frames()
try:
    pyglet.app.run(interval=None)
finally:
    pyglet.clock.unschedule(draw_frame)
    try:
        if stream.is_active():
            stream.stop_stream()
    finally:
        stream.close()
        pa.terminate()
        screen.delete()
        overlay.delete()
        waterfall.image.delete()
        if gl_renderer is not None:
            gl_renderer.release()
        window.close()
