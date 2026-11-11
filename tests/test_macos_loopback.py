import io
import unittest
from unittest.mock import Mock, patch

from audio_devices import capture_devices, open_capture, resolve_device
from macos_loopback import MacOSLoopbackStream, system_audio_device


class MacOSDiscoveryTests(unittest.TestCase):
    def test_native_loopback_without_portaudio_devices(self):
        pa = Mock()
        pa.get_device_count.return_value = 0
        with patch('audio_devices.sys.platform', 'darwin'):
            devices = capture_devices(pa)
        self.assertEqual(devices, [system_audio_device()])
        self.assertEqual(resolve_device(pa, devices, '--loopback'), devices[0])
        self.assertEqual(resolve_device(pa, devices, 'macOS System Audio'), devices[0])
        pa.get_default_wasapi_loopback.assert_not_called()

    def test_native_open_does_not_call_portaudio(self):
        pa, callback = Mock(), Mock()
        with patch('macos_loopback.MacOSLoopbackStream') as stream:
            opened, config = open_capture(pa, object(), system_audio_device(), callback, 512)
        stream.assert_called_once_with(callback, 512)
        self.assertIs(opened, stream.return_value)
        self.assertEqual(config, dict(channels=2, rate=48000))
        pa.open.assert_not_called()


class MacOSStreamTests(unittest.TestCase):
    def process(self, pcm=b''):
        process = Mock(stdout=io.BytesIO(pcm), stderr=io.BytesIO(b'READY\n'))
        process.poll.return_value = None
        return process

    def test_delivers_complete_pcm_frames_and_closes_idempotently(self):
        pcm = bytes(range(32))
        process = self.process(pcm + b'partial')
        callback = Mock()
        with patch('macos_loopback.helper_path', return_value='/helper'), \
                patch('macos_loopback.subprocess.Popen', return_value=process):
            stream = MacOSLoopbackStream(callback, 4)
            self.assertFalse(stream.is_active())
            stream.start_stream()
            stream._reader.join(2)
            self.assertEqual(callback.call_count, 2)
            self.assertEqual(b''.join(c.args[0] for c in callback.call_args_list), pcm)
            self.assertFalse(stream.is_active())
            stream.stop_stream()
            stream.close()
        process.terminate.assert_called_once()
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        with self.assertRaises(OSError):
            stream.start_stream()

    def test_permission_failure_is_reported_and_process_cleaned_up(self):
        process = self.process()
        process.stderr = io.BytesIO(b'Allow Screen & System Audio Recording\n')
        with patch('macos_loopback.helper_path', return_value='/helper'), \
                patch('macos_loopback.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(OSError, 'Allow Screen & System Audio Recording'):
                MacOSLoopbackStream(Mock(), 512)
        process.terminate.assert_called_once()

    def test_unsupported_macos(self):
        from macos_loopback import helper_path
        with patch('macos_loopback.platform.mac_ver', return_value=('12.7', (), '')):
            with self.assertRaisesRegex(ValueError, 'macOS 13'):
                helper_path()
