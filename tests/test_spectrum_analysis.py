import unittest

import numpy as np

from spectrum_analysis import (
    SpectrumAnalyzer, SPECTRUM_MODES, SPECTRUM_SMOOTHING,
    SPECTRUM_DB_RANGES, CEPSTRUM_ZOOMS,
)


class SpectrumAnalysisTests(unittest.TestCase):
    def analyzer(self, mode, width=1000):
        return SpectrumAnalyzer(2048, 48000, width, 20, 24000, -100, 0, mode)

    def test_frequency_axes_and_tone_peak(self):
        # An exact-bin tone should retain its calibrated level on every axis.
        freq = 1500
        samples = np.sin(2 * np.pi * freq * np.arange(2048) / 48000)
        for mode in SPECTRUM_MODES[:3]:
            with self.subTest(mode=mode):
                analyzer = self.analyzer(mode)
                self.assertAlmostEqual(analyzer.freq_to_x(20), 0)
                self.assertAlmostEqual(analyzer.freq_to_x(24000), analyzer.width)
                analyzer.update(samples, 1 / 60)
                peak = np.argmax(analyzer.columns)
                self.assertLess(abs(peak - analyzer.freq_to_x(freq)), 4)
                self.assertAlmostEqual(float(analyzer.columns.max()), 0, delta=0.1)
        linear, log, mel = (self.analyzer(mode).freq_to_x(1500)
                            for mode in ('Linear', 'Log', 'Mel'))
        self.assertLess(linear, mel)
        self.assertLess(mel, log)

    def test_cepstrum_matches_real_log_magnitude_transform(self):
        analyzer = self.analyzer('Cepstrum', width=1023)
        samples = np.random.default_rng(42).normal(0, 0.1, 2048)
        analyzer.update(samples, 1 / 60)
        magnitude = np.abs(np.fft.rfft(samples * analyzer.window)) * analyzer.scale
        expected = np.abs(np.fft.irfft(np.log(np.maximum(magnitude, 1e-12)), 2048))
        # One bin per column, skipping the energy coefficient at zero.
        np.testing.assert_allclose(analyzer.columns, np.clip(expected[1:1024], 0, 1), atol=1e-7)
        self.assertAlmostEqual(analyzer.quefrency_to_x(1 / 48000), 0)
        self.assertAlmostEqual(analyzer.quefrency_to_x(1024 / 48000), analyzer.width)

    def test_cepstrum_reveals_periodic_signal(self):
        analyzer = self.analyzer('Cepstrum', width=1023)
        samples = np.zeros(2048)
        samples[::240] = 0.5  # 200 Hz, 5 ms period
        analyzer.update(samples, 1 / 60)
        # Ignore short-quefrency spectral-envelope structure.
        peak_bin = np.argmax(analyzer.columns[100:400]) + 101
        self.assertLessEqual(abs(peak_bin - 240), 1)

    def test_silence_and_tiny_or_wide_plots_are_finite(self):
        for mode in SPECTRUM_MODES:
            for width in (1, 200, 4000):
                with self.subTest(mode=mode, width=width):
                    analyzer = self.analyzer(mode, width)
                    analyzer.update(np.zeros(2048), 1 / 60)
                    levels = analyzer.normalized(analyzer.columns)
                    self.assertTrue(np.isfinite(levels).all())
                    self.assertTrue(((0 <= levels) & (levels <= 1)).all())
                    self.assertEqual(analyzer.line_points(0, 300).shape, (width, 2))
                    np.testing.assert_allclose(levels, 0, atol=1e-12)

    def test_time_smoothing_is_independent_of_frame_rate(self):
        samples = np.sin(2 * np.pi * 1500 * np.arange(2048) / 48000)
        for mode in SPECTRUM_MODES:
            for smoothing in tuple(SPECTRUM_SMOOTHING)[1:]:
                with self.subTest(mode=mode, smoothing=smoothing):
                    def make():
                        return SpectrumAnalyzer(2048, 48000, 200, 20, 24000, -100, 0,
                                                mode, smoothing=smoothing)
                    once, split = make(), make()
                    once.update(samples, 0.1)
                    for _ in range(10):
                        split.update(samples, 0.01)
                    np.testing.assert_allclose(once.columns, split.columns, atol=2e-5)
                    # Alternating input should move gradually rather than jump.
                    before = once.columns.copy()
                    once.update(np.zeros(2048), 0.01)
                    self.assertTrue((once.columns <= before + 1e-6).all())
                    self.assertTrue((once.columns > once.db_min).any())
                    np.testing.assert_array_equal(once.columns, once.smoothed)

    def test_db_ranges_map_floor_and_full_scale(self):
        samples = 0.01 * np.sin(2 * np.pi * 1500 * np.arange(2048) / 48000)
        for span in SPECTRUM_DB_RANGES:
            analyzer = SpectrumAnalyzer(2048, 48000, 200, 20, 24000, -span, 0)
            analyzer.update(samples, 1 / 60)
            self.assertAlmostEqual(analyzer.normalized(-span), 0)
            self.assertAlmostEqual(analyzer.normalized(0), 1)
            self.assertAlmostEqual(float(analyzer.normalized(analyzer.columns).max()),
                                   1 - 40 / span, delta=0.001)

    def test_cepstrum_zoom_scales_levels_and_clips(self):
        samples = np.random.default_rng(42).normal(0, 0.1, 2048)
        baseline = self.analyzer('Cepstrum')
        baseline.update(samples, 1 / 60)
        for zoom in CEPSTRUM_ZOOMS:
            analyzer = SpectrumAnalyzer(2048, 48000, 1000, 20, 24000, -100, 0,
                                        'Cepstrum', cepstrum_zoom=zoom)
            analyzer.update(samples, 1 / 60)
            self.assertEqual(analyzer.db_max, 1 / zoom)
            np.testing.assert_allclose(analyzer.normalized(analyzer.columns),
                                       np.clip(baseline.columns * zoom, 0, 1), atol=1e-7)
            points = analyzer.line_points(0, 300)
            self.assertTrue(((points[:, 1] >= 0) & (points[:, 1] < 300)).all())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyzer('unknown')
