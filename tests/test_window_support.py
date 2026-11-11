import os
import unittest
from unittest.mock import Mock, patch

import window_support


class WindowSupportTests(unittest.TestCase):
    def test_cocoa_hints_are_set_before_initializing_video(self):
        with patch('window_support.sys.platform', 'darwin'), patch.dict(os.environ, {}, clear=True):
            window_support.configure_windowing()
            self.assertEqual(os.environ['SDL_MAC_BACKGROUND_APP'], '0')
            self.assertEqual(os.environ['SDL_MOUSE_FOCUS_CLICKTHROUGH'], '1')

    def test_other_platforms_do_not_change_hints(self):
        with patch('window_support.sys.platform', 'win32'), patch.dict(os.environ, {}, clear=True):
            window_support.configure_windowing()
            self.assertNotIn('SDL_MOUSE_FOCUS_CLICKTHROUGH', os.environ)

    def test_macos_font_never_invokes_system_font_discovery(self):
        font = Mock()
        with patch('window_support.sys.platform', 'darwin'), \
                patch('window_support.Path.is_file', return_value=True), \
                patch.dict('sys.modules', {'pygame': Mock(font=font)}):
            window_support.ui_font(18)
        font.Font.assert_called_once_with('/System/Library/Fonts/Menlo.ttc', 18)
        font.SysFont.assert_not_called()

    def test_missing_macos_font_uses_bundled_font(self):
        font = Mock()
        with patch('window_support.sys.platform', 'darwin'), \
                patch('window_support.Path.is_file', return_value=False), \
                patch.dict('sys.modules', {'pygame': Mock(font=font)}):
            window_support.ui_font(13)
        font.Font.assert_called_once_with(None, 13)
        font.SysFont.assert_not_called()

    def test_focus_uses_existing_sdl_window(self):
        window = Mock()
        with patch('window_support.sys.platform', 'darwin'), \
                patch('window_support._display_window', None), \
                patch.dict('sys.modules', {'pygame': Mock(error=RuntimeError),
                                          'pygame._sdl2': Mock(Window=window)}):
            window_support.focus_window()
            window_support.focus_window()
            self.assertIs(window_support._display_window, window.from_display_module.return_value)
        window.from_display_module.assert_called_once_with()
        self.assertEqual(window.from_display_module.return_value.focus.call_count, 2)
