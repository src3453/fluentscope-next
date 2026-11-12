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
        stream = Mock()
        stream.is_active.return_value = True
        capture_callback = None
        position = 0

        def feed():
            nonlocal position
            t = (np.arange(8192) + position) / 48000
            position += len(t)
            samples = np.column_stack((0.65 * np.sin(t * 440 * 2 * np.pi),
                                       0.55 * np.sin(t * 440 * 2 * np.pi + 0.7)))
            capture_callback((samples * 32767).astype(np.int16).tobytes(), len(t), None, 0)

        def capture(pa, module, device, callback, *args):
            nonlocal capture_callback
            capture_callback = callback
            stream.start_stream.side_effect = feed
            return stream, {'channels': 2, 'rate': 48000}

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

            window.set_size(800, 520)
            window.dispatch_events()
            frame()
            test.assertEqual((state['WIDTH'], state['HEIGHT']), (800, 520))
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
            test.assertEqual(saved['trace_renderer'], state['trace_renderer'])
            test.assertEqual(saved['window_size'], list(state['DEFAULT_WINDOW_SIZE']))
            # Run pyglet's actual event loop too, then exercise normal shutdown.
            pyglet.clock.schedule_once(lambda dt: state['on_close'](), 0.05)
            original_run(interval=None)
            test.assertNotIn('pygame', sys.modules)

        repo = Path(__file__).resolve().parents[1]
        device = {'index': 0, 'name': 'Synthetic stereo', 'maxInputChannels': 2}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'fluentscope_next.py'
            target.write_text((repo / 'fluentscope_next.py').read_text())
            (Path(temp) / 'fluentscope_settings.json').write_text(json.dumps({
                'trace_renderer': 'Pygame', 'max_fps': 60, 'window_size': [1000, 600],
            }))
            fake_audio = Mock(paInt16=8, paContinue=0)
            modules = {'pyaudio': fake_audio, 'pyaudiowpatch': fake_audio}
            if not modern:
                modules['moderngl'] = None
            with injected_modules(modules), \
                    patch.object(audio_devices, 'capture_devices', return_value=[device]), \
                    patch.object(audio_devices, 'resolve_device', return_value=device), \
                    patch.object(audio_devices, 'open_capture', side_effect=capture), \
                    patch.object(pyglet.app, 'run', new=exercise), \
                    patch.object(sys, 'argv', [str(target), '0']):
                runpy.run_path(str(target))
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        fake_audio.PyAudio.return_value.terminate.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('moderngl'), 'optional ModernGL is not installed')
    def test_all_views_renderers_retina_input_and_shutdown(self):
        self.exercise_app(modern=True)

    def test_pyglet_fallback_without_moderngl(self):
        self.exercise_app(modern=False)
