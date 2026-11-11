import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from audio_devices import capture_devices, open_capture, resolve_device


def device(index, name, channels=2, loopback=False, rate=48000):
    return dict(index=index, name=name, maxInputChannels=channels,
                isLoopbackDevice=loopback, defaultSampleRate=rate, hostApi=0)


class AudioDeviceTests(unittest.TestCase):
    def setUp(self):
        self.devices = [device(1, 'Microphone'), device(4, 'Speakers', channels=0),
                        device(8, 'Speakers [Loopback]', 8, True, 44100)]
        self.pa = Mock()
        self.pa.get_device_count.return_value = 3
        self.pa.get_device_info_by_index.side_effect = self.devices
        self.pa.get_host_api_info_by_index.return_value = dict(name='WASAPI')

    def test_discovery_excludes_output_only_endpoints(self):
        devices = capture_devices(self.pa)
        self.assertEqual([d['name'] for d in devices],
                         ['Microphone', 'Speakers [Loopback]'])

    def test_selection_uses_portaudio_index(self):
        self.assertEqual(resolve_device(self.pa, self.devices, '8')['index'], 8)
        self.assertEqual(resolve_device(self.pa, self.devices, 'exact:Microphone')['index'], 1)
        self.assertEqual(resolve_device(self.pa, self.devices, 'micro')['index'], 1)

    def test_missing_and_ambiguous_names_fail(self):
        for selector in ('999', 'exact:Missing', 'Speakers', '-1'):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                resolve_device(self.pa, self.devices, selector)

    def test_default_loopback(self):
        self.pa.get_default_wasapi_loopback.return_value = self.devices[2]
        self.assertEqual(resolve_device(self.pa, self.devices, '--loopback')['index'], 8)
        with self.assertRaises(ValueError):
            resolve_device(object(), self.devices, '--loopback')

    def test_loopback_uses_native_format_and_waits_for_callback_setup(self):
        stream, kwargs = open_capture(self.pa, SimpleNamespace(paInt16=8),
                                      self.devices[2], None, 512)
        self.assertEqual(kwargs['channels'], 8)
        self.assertEqual(kwargs['rate'], 44100)
        self.assertFalse(kwargs['start'])
        stream.start_stream.assert_not_called()

    def test_input_falls_back_to_native_rate_and_mono(self):
        self.pa.open.side_effect = [OSError('Unsupported sample rate'), Mock()]
        _, kwargs = open_capture(self.pa, SimpleNamespace(paInt16=8),
                                 device(5, 'Mono', 1, rate=44100), None, 512)
        self.assertEqual(kwargs['rate'], 44100)
        self.assertEqual(kwargs['channels'], 1)
        self.assertEqual(self.pa.open.call_count, 2)

    def test_open_failure_propagates(self):
        self.pa.open.side_effect = OSError('Disconnected')
        with self.assertRaises(OSError):
            open_capture(self.pa, SimpleNamespace(paInt16=8), self.devices[2], None, 512)


if __name__ == '__main__':
    unittest.main()
