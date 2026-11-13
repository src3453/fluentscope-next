"""Opt-in real-window tests; synthetic audio never opens a capture device.

FLUENTSCOPE_NATIVE_TESTS=1 python -m unittest discover -s tests -v
"""
import importlib.util
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


@contextmanager
def injected_modules(modules):
    # Preserve native pyglet modules imported while the app runs. Restoring
    # the entire sys.modules dict would cause Cocoa classes to be registered
    # again when the second application test starts.
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@unittest.skipUnless(os.environ.get('FLUENTSCOPE_NATIVE_TESTS') == '1',
                     'set FLUENTSCOPE_NATIVE_TESTS=1 on a graphical desktop')
class PygletAppTests(unittest.TestCase):
    def exercise_app(self, modern):
        from window_support import configure_windowing, render_size
        configure_windowing()
        import pyglet
        import audio_devices

        test = self
        original_run = pyglet.app.run
        streams = []
        capture_callback = None
        position = 0

        def feed():
            nonlocal position
            t = (np.arange(8192) + position) / 48000
            position += len(t)
            samples = np.column_stack((0.65 * np.sin(t * 440 * 2 * np.pi),
                                       0.55 * np.sin(t * 440 * 2 * np.pi + 0.7)))
            capture_callback((samples * 32767).astype(np.int16).tobytes(), len(t), None, 0)

        def capture(audio, device, callback, buffer_frames=0, *args, **kwargs):
            nonlocal capture_callback
            capture_callback = callback
            stream = Mock()
            stream.is_active.return_value = True
            stream.start_stream.side_effect = feed
            streams.append(stream)
            return stream, {'channels': 2, 'rate': 48000, 'buffer_frames': buffer_frames,
                            'latency': 0.005, 'exclusive': False}

        def exercise(*args, **kwargs):
            # Called directly from the running application's module namespace.
            state = sys._getframe(1).f_globals
            key, window = state['key'], state['window']
            test.assertEqual(state['gl_renderer'] is not None, modern)
            test.assertEqual(state['trace_renderer'], 'Pyglet')  # legacy setting
            context = window.context

            def frame():
                feed()
                state['draw_frame'](1 / 60)
                test.assertEqual(pyglet.gl.glGetError(), pyglet.gl.GL_NO_ERROR)
                width, height = window.get_size()
                pixels = window.get_framebuffer_size()
                test.assertEqual(state['overlay']._density, (pixels[0] / width, pixels[1] / height))
                if modern:
                    renderer = state['gl_renderer']
                    test.assertEqual((renderer.width, renderer.height),
                                     render_size((width, height), pixels, state['RETINA']))

            for renderer in ('CRT', 'GL Line', 'Pyglet') if modern else ('Pyglet',):
                state['trace_renderer'] = renderer
                for view in range(3):
                    state['view'] = view
                    for retina in (False, True):
                        state['set_retina'](retina)
                        frame()
                        test.assertIs(window.context, context)

            # Mode changes update both plots, clear history, and persist on resize.
            state['view'] = state['VIEW_SPECTRUM']
            for mode in ('Linear', 'Mel', 'Cepstrum', 'Log'):
                old_waterfall = state['waterfall']
                state['on_key_press'](key.Q, 0)
                test.assertEqual(state['spectrum_mode'], mode)
                test.assertEqual(state['analyzer'].mode, mode)
                test.assertIs(state['waterfall'], old_waterfall)
                test.assertFalse(state['waterfall'].history.any())
                frame()
            state['on_key_press'](key.Q, key.MOD_SHIFT)
            test.assertEqual(state['spectrum_mode'], 'Cepstrum')
            state['apply_settings']({'spectrum_mode': 'invalid'})
            test.assertEqual(state['spectrum_mode'], 'Cepstrum')

            # Independent frequency range and cepstrum zoom, with persistent smoothing.
            state['on_key_press'](key.Y, 0)
            test.assertEqual(state['cepstrum_zoom'], 2)
            test.assertEqual(state['analyzer'].db_max, 0.5)
            state['on_key_press'](key.U, 0)
            test.assertEqual(state['analyzer'].smoothing, 'Low')
            frame()
            state['on_key_press'](key.U, key.MOD_SHIFT)
            test.assertEqual(state['analyzer'].smoothing, 'Off')
            state['on_key_press'](key.U, key.MOD_SHIFT)
            test.assertEqual(state['analyzer'].smoothing, 'High')
            state['on_key_press'](key.Q, 0)
            test.assertEqual(state['analyzer'].db_min, -100)
            state['on_key_press'](key.Y, 0)
            test.assertEqual(state['analyzer'].db_min, -120)
            state['on_key_press'](key.Y, key.MOD_SHIFT)
            test.assertEqual(state['analyzer'].db_min, -100)
            state['on_key_press'](key.Y, 0)
            state['on_key_press'](key.Q, key.MOD_SHIFT)
            test.assertEqual(state['analyzer'].db_max, 0.5)
            state['apply_settings']({'spectrum_smoothing': [], 'spectrum_db_range': True,
                                     'cepstrum_zoom': True})
            test.assertEqual(state['analyzer'].smoothing, 'High')
            test.assertEqual(state['spectrum_db_range'], 120)
            test.assertEqual(state['cepstrum_zoom'], 2)
            for renderer in ('CRT', 'GL Line', 'Pyglet') if modern else ('Pyglet',):
                state['trace_renderer'] = renderer
                frame()

            # Silence must render safely, including a stationary XY figure.
            with state['lock']:
                state['ring'].fill(0)
            state['view'] = state['VIEW_LISSAJOUS']
            state['draw_frame'](1 / 60)

            for symbol in (key.F1, key.F2, key.F3, key.F3, key.T, key.M, key.X, key.A,
                           key.H, key.J, key.K, key.D, key.L, key.W, key.I, key.COMMA,
                           key.PERIOD, key.F5, key.F6, key.F6):
                state['on_key_press'](symbol, 0)
                frame()
            old_fps = state['FPS']
            state['on_key_press'](key.F5, key.MOD_SHIFT)
            test.assertNotEqual(state['FPS'], old_fps)

            # FPS sync: the largest power-of-two buffer within one frame, following F5.
            state['set_vsync'](False)
            state['FPS'] = 60
            state['on_key_press'](key.F8, 0) # Auto -> FPS sync
            test.assertEqual(state['AUDIO_BUFFER'], state['BUFFER_FPS_SYNC'])
            test.assertEqual(state['STREAM_OPEN_KWARGS']['buffer_frames'], 512) # 800 / frame
            reopened = len(streams)
            state['on_key_press'](key.F5, 0) # 60 -> 120 FPS
            test.assertEqual(state['STREAM_OPEN_KWARGS']['buffer_frames'], 256) # 400 / frame
            test.assertEqual(len(streams), reopened + 1)
            streams[-2].close.assert_called_once()
            state['on_key_press'](key.J, 0) # unrelated keys keep the stream
            state['on_key_press'](key.J, 0)
            test.assertEqual(len(streams), reopened + 1)
            state['on_key_press'](key.F8, 0) # FPS sync -> 32 frames
            test.assertEqual(state['STREAM_OPEN_KWARGS']['buffer_frames'], 32)
            state['on_key_press'](key.F8, key.MOD_SHIFT)
            state['on_key_press'](key.F8, key.MOD_SHIFT)
            test.assertEqual(state['AUDIO_BUFFER'], 0)
            panel, titles, rows, hint = state['menu_layout']()
            test.assertIn('System', [title for title, _ in titles])
            for _, _, rect in rows: # no row runs outside its panel or into another
                test.assertTrue(panel.contains(rect))
            frame()

            window.set_size(800, 520)
            window.dispatch_events()
            frame()
            test.assertEqual((state['WIDTH'], state['HEIGHT']), (800, 520))
            test.assertEqual(state['analyzer'].mode, 'Cepstrum')
            test.assertEqual(state['analyzer'].smoothing, 'High')
            test.assertEqual(state['analyzer'].db_max, 0.5)
            state['show_hud'], state['menu_open'] = True, False
            x, y = state['MENU_BUTTON'].center
            state['on_mouse_press'](x, state['HEIGHT'] - y, state['mouse'].LEFT, 0)
            test.assertTrue(state['menu_open'])
            panel, titles, rows, hint = state['menu_layout']()
            grid_row = next(rect for symbol, _, rect in rows if symbol == key.J)
            x, y = grid_row.center
            old_grid = state['show_scope_graticule']
            state['on_mouse_press'](x, state['HEIGHT'] - y, state['mouse'].LEFT, 0)
            test.assertNotEqual(state['show_scope_graticule'], old_grid)
            frame()
            state['on_key_press'](key.F4, 0)
            frame()
            test.assertEqual(window.get_size(), state['DEFAULT_WINDOW_SIZE'])
            with open(state['SETTINGS_PATH']) as settings:
                saved = json.load(settings)
            test.assertEqual(saved['spectrum_mode'], 'Cepstrum')
            test.assertEqual(saved['spectrum_smoothing'], 'High')
            test.assertEqual(saved['spectrum_db_range'], 120)
            test.assertEqual(saved['cepstrum_zoom'], 2)
            test.assertEqual(saved['trace_renderer'], state['trace_renderer'])
            test.assertEqual(saved['window_size'], list(state['DEFAULT_WINDOW_SIZE']))
            state['set_spectrum_options'](mode='Linear', smoothing='Off', db_range=40, zoom=1)
            state['apply_settings'](saved)
            test.assertEqual(state['analyzer'].mode, 'Cepstrum')
            test.assertEqual(state['analyzer'].smoothing, 'High')
            test.assertEqual(state['analyzer'].db_max, 0.5)
            test.assertEqual(state['spectrum_db_range'], 120)
            # Check focus after Cocoa's real event loop has finished launching.
            # A topmost window alone does not prove it can receive keyboard input.
            focus_errors = []

            def check_focus_and_close(dt):
                try:
                    if sys.platform == 'darwin':
                        from pyglet.libs.darwin import cocoapy
                        app = cocoapy.ObjCClass('NSApplication').sharedApplication()
                        test.assertTrue(app.isActive())
                        test.assertTrue(window._nswindow.isKeyWindow())
                        for restore in (state['on_activate'], state['on_show'],
                                        lambda: state['on_mouse_press'](0, 0, state['mouse'].RIGHT, 0)):
                            window._nswindow.makeFirstResponder_(None)
                            restore()
                            responder = window._nswindow.firstResponder()
                            test.assertEqual(responder.ptr.value, window._nsview.ptr.value)
                        # Moving over the window must not issue a focus request.
                        with patch.dict(state, {'activate_window': Mock()}) as patched:
                            state['on_mouse_motion'](10, 10, 1, 1)
                            patched['activate_window'].assert_not_called()
                except BaseException as exc:
                    focus_errors.append(exc)
                finally:
                    state['on_close']()

            pyglet.clock.schedule_once(check_focus_and_close, 0.15)
            original_run(interval=None)
            if focus_errors:
                raise focus_errors[0]
            test.assertNotIn('pygame', sys.modules)

        repo = Path(__file__).resolve().parents[1]
        device = {'index': 0, 'name': 'Synthetic stereo', 'maxInputChannels': 2}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'fluentscope_next.py'
            target.write_text((repo / 'fluentscope_next.py').read_text())
            (Path(temp) / 'fluentscope_settings.json').write_text(json.dumps({
                'trace_renderer': 'Pygame', 'max_fps': 60, 'window_size': [1000, 600],
            }))
            audio_system = Mock()
            modules = {}
            if not modern:
                modules['moderngl'] = None
            with injected_modules(modules), \
                    patch.object(audio_devices, 'AudioSystem', return_value=audio_system), \
                    patch.object(audio_devices, 'capture_devices', return_value=[device]), \
                    patch.object(audio_devices, 'resolve_device', return_value=device), \
                    patch.object(audio_devices, 'open_capture', side_effect=capture), \
                    patch.object(pyglet.app, 'run', new=exercise), \
                    patch.object(sys, 'argv', [str(target), '0']):
                runpy.run_path(str(target))
        streams[-1].stop_stream.assert_called_once()
        for stream in streams:
            stream.close.assert_called_once()
        audio_system.terminate.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('moderngl'), 'optional ModernGL is not installed')
    def test_all_views_renderers_retina_input_and_shutdown(self):
        self.exercise_app(modern=True)

    def test_pyglet_fallback_without_moderngl(self):
        self.exercise_app(modern=False)
