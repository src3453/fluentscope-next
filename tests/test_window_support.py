import unittest
from unittest.mock import Mock, patch

from window_support import activate_window, configure_windowing, create_window, mouse_position, render_size


class WindowSupportTests(unittest.TestCase):
    def cocoa(self):
        cocoa = Mock()
        app = cocoa.ObjCClass.return_value.sharedApplication.return_value
        app.isActive.return_value = False
        cocoa.get_selector.return_value = 'activate'
        return cocoa, app

    def test_cocoa_focus_restores_app_key_window_and_first_responder(self):
        cocoa, app = self.cocoa()
        window = Mock()
        with patch('window_support.sys.platform', 'darwin'), \
                patch.dict('sys.modules', {'pyglet.libs.darwin': Mock(cocoapy=cocoa)}):
            activate_window(window)
        app.activate.assert_called_once_with()
        app.activateIgnoringOtherApps_.assert_not_called()
        window._nswindow.makeKeyAndOrderFront_.assert_called_once_with(None)
        window._nswindow.makeFirstResponder_.assert_called_once_with(window._nsview)

    def test_active_cocoa_app_only_restores_window_and_input_view(self):
        cocoa, app = self.cocoa()
        app.isActive.return_value = True
        with patch('window_support.sys.platform', 'darwin'), \
                patch.dict('sys.modules', {'pyglet.libs.darwin': Mock(cocoapy=cocoa)}):
            activate_window(Mock())
        app.activate.assert_not_called()
        app.activateIgnoringOtherApps_.assert_not_called()

    def test_older_cocoa_uses_legacy_activation(self):
        cocoa, app = self.cocoa()
        app.respondsToSelector_.return_value = False
        with patch('window_support.sys.platform', 'darwin'), \
                patch.dict('sys.modules', {'pyglet.libs.darwin': Mock(cocoapy=cocoa)}):
            activate_window(Mock())
        app.activateIgnoringOtherApps_.assert_called_once_with(True)
        app.activate.assert_not_called()

    def test_synchronous_activation_callback_cannot_recurse(self):
        cocoa, app = self.cocoa()
        window = Mock()
        window._nswindow.makeKeyAndOrderFront_.side_effect = lambda sender: activate_window(window)
        with patch('window_support.sys.platform', 'darwin'), \
                patch.dict('sys.modules', {'pyglet.libs.darwin': Mock(cocoapy=cocoa)}):
            activate_window(window)
        window._nswindow.makeKeyAndOrderFront_.assert_called_once()

    def test_other_platforms_use_pyglet_activation(self):
        window = Mock()
        with patch('window_support.sys.platform', 'win32'):
            activate_window(window)
        window.activate.assert_called_once()

    def test_configuration_preserves_logical_coordinates_without_shadow_window(self):
        pyglet = Mock()
        with patch.dict('sys.modules', {'pyglet': pyglet}):
            configure_windowing()
        self.assertFalse(pyglet.options.shadow_window)
        self.assertEqual(pyglet.options.dpi_scaling, 'stretch')

    def test_all_apple_silicon_macs_use_the_native_cocoa_event_loop(self):
        pyglet = Mock()
        with patch('window_support.sys.platform', 'darwin'), \
                patch('window_support.platform.machine', return_value='arm64'), \
                patch.dict('sys.modules', {'pyglet': pyglet}):
            configure_windowing()
        self.assertTrue(pyglet.options.osx_alt_loop)

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
