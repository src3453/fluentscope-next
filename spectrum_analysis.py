"""NumPy spectrum and real-cepstrum analysis, independent of the window backend."""

import numpy as np

SPECTRUM_MODES = ("Log", "Linear", "Mel", "Cepstrum")
SPECTRUM_SMOOTHING = {"Off": 0.0, "Low": 0.05, "Med": 0.15, "High": 0.4}
SPECTRUM_DB_RANGES = (40, 60, 80, 100, 120, 140)
CEPSTRUM_ZOOMS = (1, 2, 4, 8, 16)


class SpectrumAnalyzer:
    """Peak-preserving frequency columns, or absolute real cepstrum coefficients."""

    def __init__(self, fft_size, rate, width, f_min, f_max, db_min, db_max,
                 mode="Log", fall_db_per_sec=240.0, smoothing="Off", cepstrum_zoom=1):
        if mode not in SPECTRUM_MODES:
            raise ValueError(f"Unknown spectrum mode: {mode}")
        if smoothing not in SPECTRUM_SMOOTHING:
            raise ValueError(f"Unknown spectrum smoothing: {smoothing}")
        if cepstrum_zoom not in CEPSTRUM_ZOOMS:
            raise ValueError(f"Unknown cepstrum zoom: {cepstrum_zoom}")
        self.smoothing = smoothing
        self.smoothing_seconds = SPECTRUM_SMOOTHING[smoothing]
        self.cepstrum_zoom = cepstrum_zoom
        self.fft_size = fft_size
        self.rate = rate
        self.width = width
        self.f_min = f_min
        self.f_max = min(f_max, rate / 2)
        self.mode = mode
        self.db_min, self.db_max = (0.0, 1.0 / cepstrum_zoom) if mode == "Cepstrum" else (db_min, db_max)
        self.fall_per_sec = fall_db_per_sec * (0.01 if mode == "Cepstrum" else 1.0)
        self.window = np.hanning(fft_size).astype(np.float32)
        self.scale = 2.0 / self.window.sum()  # a full-scale sine reads 0 dBFS
        bin_count = fft_size // 2 + 1
        self.bin_positions = np.arange(bin_count, dtype=np.float32)

        if mode == "Cepstrum":
            # Exclude the zero-quefrency coefficient (overall log energy).
            self.q_min, self.q_max = 1.0 / rate, (fft_size // 2) / rate
            edges_bin = np.linspace(1, fft_size // 2, width + 1)
            centers_bin = (edges_bin[:-1] + edges_bin[1:]) / 2
        else:
            edges = np.linspace(self._axis(f_min), self._axis(self.f_max), width + 1)
            edges_hz = self._inverse_axis(edges)
            edges_bin = edges_hz * fft_size / rate
            centers_bin = self._inverse_axis((edges[:-1] + edges[1:]) / 2) * fft_size / rate

        edge_idx = np.clip(np.ceil(edges_bin).astype(np.int64), 0, bin_count - 1)
        self.reduce_idx = edge_idx[:-1]
        self.dense = edge_idx[1:] > edge_idx[:-1]
        self.centers_bin = centers_bin.astype(np.float32)
        self.columns = np.full(width, self.db_min, dtype=np.float32)
        self.smoothed = self.columns.copy()
        self._averaged = self.columns.copy()

    def _axis(self, freq):
        if self.mode == "Log":
            return np.log(freq)
        if self.mode == "Mel":
            return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)
        return freq

    def _inverse_axis(self, value):
        if self.mode == "Log":
            return np.exp(value)
        if self.mode == "Mel":
            return 700.0 * np.expm1(np.asarray(value) * np.log(10.0) / 2595.0)
        return value

    def freq_to_x(self, freq):
        return self.width * (self._axis(freq) - self._axis(self.f_min)) / (
            self._axis(self.f_max) - self._axis(self.f_min))

    def quefrency_to_x(self, seconds):
        return self.width * (seconds - self.q_min) / (self.q_max - self.q_min)

    def update(self, samples, dt):
        magnitude = np.abs(np.fft.rfft(samples * self.window)) * self.scale
        if self.mode == "Cepstrum":
            values = np.abs(np.fft.irfft(np.log(np.maximum(magnitude, 1e-12)), self.fft_size))
            values = values[:self.fft_size // 2 + 1]
        else:
            values = 20.0 * np.log10(magnitude + 1e-12)
        peak = np.maximum.reduceat(values, self.reduce_idx)
        interp = np.interp(self.centers_bin, self.bin_positions, values)
        columns = np.where(self.dense, peak, interp)
        self.columns = np.clip(columns, self.db_min, self.db_max).astype(np.float32)
        if self.smoothing_seconds:
            # Exponential time averaging uses elapsed time, independent of FPS.
            alpha = -np.expm1(-max(0.0, dt) / self.smoothing_seconds)
            self._averaged += alpha * (self.columns - self._averaged)
            self.columns = self._averaged.copy()
            self.smoothed = self.columns.copy()
        else:
            # Preserve the original instant attack / peak-release display.
            self.smoothed = np.maximum(self.columns, self.smoothed - self.fall_per_sec * dt)

    def normalized(self, values):
        return (values - self.db_min) / (self.db_max - self.db_min)

    def line_points(self, top, height, round_pixels=True):
        x = np.arange(self.width, dtype=np.float32)
        y = top + (1.0 - self.normalized(self.smoothed)) * (height - 1)
        points = np.column_stack((x, y))
        return np.rint(points).astype(np.int32) if round_pixels else points.astype(np.float32)
