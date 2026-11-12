import unittest
from unittest.mock import Mock, patch

from window_support import configure_windowing, create_window, mouse_position, render_size


class WindowSupportTests(unittest.TestCase):
    def test_configuration_preserves_logical_coordinates_without_shadow_window(self):
        pyglet = Mock()
        with patch.dict('sys.modules', {'pyglet': pyglet}):
            configure_windowing()
        self.assertFalse(pyglet.options.shadow_window)
        self.assertEqual(pyglet.options.dpi_scaling, 'stretch')

    def test_window_uses_native_pyglet_controls_and_starts_hidden(self):
        pyglet = Mock()
        with patch.dict('sys.modules', {'pyglet': pyglet}):
            window = create_window(1000, 600, vsync=True, minimum_size=(640, 480))
        self.assertIs(window, pyglet.window.Window.return_value)
        self.assertTrue(pyglet.window.Window.call_args.kwargs['vsync'])
        self.assertFalse(pyglet.window.Window.call_args.kwargs['visible'])
        window.set_minimum_size.assert_called_once_with(640, 480)

    def test_macos_retina_selects_backing_pixels_without_changing_window_size(self):
        logical, pixels = (1000, 600), (2000, 1200)
        self.assertEqual(render_size(logical, pixels, False, platform='darwin'), logical)
        self.assertEqual(render_size(logical, pixels, True, platform='darwin'), pixels)

    def test_standard_density_display_uses_same_size_with_either_setting(self):
        for retina in (False, True):
            self.assertEqual(render_size((1000, 600), (1000, 600), retina, platform='darwin'),
                             (1000, 600))

    def test_other_platforms_use_actual_framebuffer_pixels(self):
        for platform in ('win32', 'linux'):
            self.assertEqual(render_size((1000, 600), (1250, 750), False, platform=platform),
                             (1250, 750))

    def test_mouse_coordinates_stay_in_logical_points(self):
        self.assertEqual(mouse_position(150, 350, 600), (150, 250))
