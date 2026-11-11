"""The macOS renderer must never initialize Tk in its own process."""
import json
import unittest
from unittest.mock import Mock, patch

from audio_devices import choose_device
from macos_loopback import system_audio_device


class DevicePickerTests(unittest.TestCase):
    def setUp(self):
        platform = patch('audio_devices.sys.platform', 'darwin')
        platform.start()
        self.addCleanup(platform.stop)
        self.pa = Mock()
        self.pa.get_default_input_device_info.side_effect = OSError('No input')
        self.devices = [system_audio_device()]

    def test_selection_opens_capture_in_parent_without_loading_tk(self):
        opened = Mock()
        with patch('audio_devices.subprocess.run', return_value=Mock(
                returncode=0, stdout='-100')) as run, \
                patch('audio_devices._choose_device') as local_picker:
            result = choose_device(self.pa, self.devices, opened)
        self.assertEqual(result, (self.devices[0], opened.return_value))
        opened.assert_called_once_with(self.devices[0])
        local_picker.assert_not_called()
        self.assertEqual(run.call_args.args[0][-1], '--picker')
        payload = json.loads(run.call_args.kwargs['input'])
        self.assertEqual(payload['devices'], self.devices)
        self.assertIsNone(payload['default_input'])

    def test_cancel_does_not_open_capture(self):
        opened = Mock()
        with patch('audio_devices.subprocess.run', return_value=Mock(
                returncode=0, stdout='null')):
            self.assertIsNone(choose_device(self.pa, self.devices, opened))
        opened.assert_not_called()

    def test_capture_error_reopens_picker_with_selection_and_error(self):
        opened = Mock(side_effect=[OSError('Permission denied'), 'stream'])
        with patch('audio_devices.subprocess.run', return_value=Mock(
                returncode=0, stdout='-100')) as run:
            self.assertEqual(choose_device(self.pa, self.devices, opened),
                             (self.devices[0], 'stream'))
        self.assertEqual(run.call_count, 2)
        payload = json.loads(run.call_args.kwargs['input'])
        self.assertEqual(payload['selected_index'], -100)
        self.assertIn('Permission denied', payload['initial_status'])

    def test_picker_failure_reports_error_without_opening_capture(self):
        opened = Mock()
        with patch('audio_devices.subprocess.run', return_value=Mock(
                returncode=1, stderr='Tk is unavailable')):
            with self.assertRaisesRegex(OSError, 'Tk is unavailable'):
                choose_device(self.pa, self.devices, opened)
        opened.assert_not_called()

    def test_invalid_selection_fails_without_opening_capture(self):
        for value in ('garbage', '999'):
            opened = Mock()
            with self.subTest(value=value), patch('audio_devices.subprocess.run',
                    return_value=Mock(returncode=0, stdout=value)):
                with self.assertRaises(OSError):
                    choose_device(self.pa, self.devices, opened)
            opened.assert_not_called()

    def test_other_platforms_keep_in_process_picker(self):
        with patch('audio_devices.sys.platform', 'win32'), \
                patch('audio_devices._choose_device') as local_picker, \
                patch('audio_devices.subprocess.run') as run:
            opened = Mock()
            self.assertIs(choose_device(self.pa, self.devices, opened),
                          local_picker.return_value)
        local_picker.assert_called_once_with(self.pa, self.devices, opened)
        run.assert_not_called()
