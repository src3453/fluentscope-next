import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from audio_devices import (AudioSystem, SoundDeviceStream, capture_devices, open_capture,
                           print_devices, resolve_device)


def device(index, name, channels=2, loopback=False, rate=48000, api='Windows WASAPI'):
    info = dict(index=index, name=name, maxInputChannels=channels, api_name=api,
                isLoopbackDevice=loopback, defaultSampleRate=rate)
    if loopback:
        info.update(pa_index=index, capture_backend='pyaudio')
    else:
        info.update(sd_index=index, capture_backend='sounddevice')
    return info


class PortAudioError(Exception):
    pass


def fake_sounddevice():
    """sounddevice stand-in: MME mic, WASAPI mic, an output, a WDM-KS mic."""
    sd = Mock(PortAudioError=PortAudioError)
    sd.query_hostapis.return_value = [dict(name='MME'), dict(name='Windows WASAPI'),
                                      dict(name='Windows WDM-KS')]
    sd.query_devices.return_value = [
        dict(name='Mic', hostapi=0, max_input_channels=2, default_samplerate=44100.0),
        dict(name='Mic', hostapi=1, max_input_channels=1, default_samplerate=48000.0),
        dict(name='Speakers', hostapi=1, max_input_channels=0, default_samplerate=48000.0),
        dict(name='Mic\r\n(KS)', hostapi=2, max_input_channels=8, default_samplerate=44100.0),
    ]
    sd.default.device = [1, 2]
    sd.RawInputStream.return_value.latency = 0.003
    return sd


def fake_loopback_backend():
    backend = Mock(paInt16=8)
    pa = backend.PyAudio.return_value
    pa.get_device_count.return_value = 3
    pa.get_device_info_by_index.side_effect = lambda i: [
        dict(name='Mic', maxInputChannels=2, defaultSampleRate=48000.0, hostApi=0),
        dict(name='Speakers', maxInputChannels=0, defaultSampleRate=48000.0, hostApi=0),
        dict(name='Speakers [Loopback]', maxInputChannels=8, defaultSampleRate=44100.0,
             hostApi=0, isLoopbackDevice=True),
    ][i]
    pa.get_host_api_info_by_index.return_value = dict(name='Windows WASAPI')
    pa.get_default_wasapi_loopback.return_value = dict(index=2)
    pa.open.return_value.get_input_latency.return_value = 0.022
    return backend


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.sd = fake_sounddevice()
        self.backend = fake_loopback_backend()
        self.audio = AudioSystem(self.sd, self.backend)

    def test_inputs_from_sounddevice_and_only_loopbacks_from_pyaudiowpatch(self):
        with patch('audio_devices.sys.platform', 'win32'):
            devices = capture_devices(self.audio)
        self.assertEqual([(d['index'], d['name'], d['api_name'], d['capture_backend'])
                          for d in devices], [
            (0, 'Mic', 'MME', 'sounddevice'),
            (1, 'Mic', 'Windows WASAPI', 'sounddevice'),
            (3, 'Mic (KS)', 'Windows WDM-KS', 'sounddevice'),
            (6, 'Speakers [Loopback]', 'Windows WASAPI', 'pyaudio'),
        ])
        self.assertEqual(devices[-1]['pa_index'], 2)
        self.assertTrue(devices[-1]['isLoopbackDevice'])

    def test_defaults_map_into_the_shared_index_space(self):
        devices = self.audio.devices()
        self.assertEqual(self.audio.get_default_input_device_info(), dict(index=1))
        self.assertEqual(resolve_device(self.audio, devices, '--loopback')['index'], 6)

    def test_without_pyaudiowpatch_loopback_is_unavailable(self):
        audio = AudioSystem(self.sd, False)
        with patch('audio_devices.sys.platform', 'linux'):
            devices = capture_devices(audio)
        self.assertFalse(any(d['isLoopbackDevice'] for d in devices))
        with self.assertRaises(ValueError):
            resolve_device(audio, devices, '--loopback')
        audio.terminate()

    def test_missing_default_input_is_a_lookup_error(self):
        self.sd.default.device = [-1, -1]
        with self.assertRaises(LookupError):
            self.audio.get_default_input_device_info()

    def test_terminate_releases_pyaudiowpatch_once(self):
        self.audio.terminate()
        self.audio.terminate()
        self.backend.PyAudio.return_value.terminate.assert_called_once()


class OpenTests(unittest.TestCase):
    def setUp(self):
        self.sd = fake_sounddevice()
        self.backend = fake_loopback_backend()
        self.audio = AudioSystem(self.sd, self.backend)

    def test_buffer_size_sets_blocksize_and_matching_latency(self):
        stream, opened = open_capture(self.audio, device(1, 'Mic'), Mock(), 64, 48000, 2)
        kwargs = self.sd.RawInputStream.call_args.kwargs
        self.assertEqual((kwargs['blocksize'], kwargs['dtype']), (64, 'int16'))
        self.assertAlmostEqual(kwargs['latency'], 64 / 48000)
        self.assertIsNone(kwargs['extra_settings'])
        self.assertEqual(opened, dict(channels=2, rate=48000, buffer_frames=64,
                                      latency=0.003, exclusive=False))
        self.assertIsInstance(stream, SoundDeviceStream)
        self.sd.RawInputStream.return_value.start.assert_not_called()

    def test_auto_buffer_asks_for_lowest_latency(self):
        open_capture(self.audio, device(1, 'Mic'), Mock())
        kwargs = self.sd.RawInputStream.call_args.kwargs
        self.assertEqual((kwargs['blocksize'], kwargs['latency']), (0, 'low'))

    def test_exclusive_only_applies_to_wasapi_inputs(self):
        _, opened = open_capture(self.audio, device(1, 'Mic'), Mock(), exclusive=True)
        self.sd.WasapiSettings.assert_called_once_with(exclusive=True)
        self.assertIs(self.sd.RawInputStream.call_args.kwargs['extra_settings'],
                      self.sd.WasapiSettings.return_value)
        self.assertTrue(opened['exclusive'])
        _, opened = open_capture(self.audio, device(3, 'KS', api='Windows WDM-KS'), Mock(),
                                 exclusive=True)
        self.assertIsNone(self.sd.RawInputStream.call_args.kwargs['extra_settings'])
        self.assertFalse(opened['exclusive'])

    def test_input_falls_back_to_native_rate_and_mono(self):
        self.sd.RawInputStream.side_effect = [PortAudioError('Invalid sample rate'),
                                              Mock(latency=0.01)]
        _, opened = open_capture(self.audio, device(5, 'Mono', 1, rate=44100), Mock())
        self.assertEqual((opened['rate'], opened['channels']), (44100, 1))
        self.assertEqual(self.sd.RawInputStream.call_count, 2)

    def test_open_failure_becomes_oserror_for_the_picker(self):
        self.sd.RawInputStream.side_effect = PortAudioError('Device unavailable')
        with self.assertRaises(OSError):
            open_capture(self.audio, device(1, 'Mic'), Mock())

    def test_callback_receives_pyaudio_style_arguments(self):
        callback = Mock()
        open_capture(self.audio, device(1, 'Mic'), callback)
        deliver = self.sd.RawInputStream.call_args.kwargs['callback']
        self.assertIsNone(deliver(b'\0\0\0\0', 1, None, 0))
        callback.assert_called_once_with(b'\0\0\0\0', 1, None, 0)

    def test_loopback_uses_native_format_on_pyaudiowpatch(self):
        loopback = device(6, 'Speakers [Loopback]', 8, True, 44100)
        loopback['pa_index'] = 2
        stream, opened = open_capture(self.audio, loopback, None, 0, exclusive=True)
        pa = self.backend.PyAudio.return_value
        kwargs = pa.open.call_args.kwargs
        self.assertEqual((kwargs['channels'], kwargs['rate']), (8, 44100))
        self.assertEqual(kwargs['input_device_index'], 2)
        self.assertEqual(kwargs['frames_per_buffer'], 512)
        self.assertFalse(kwargs['start'])
        self.assertFalse(opened['exclusive'])
        self.assertEqual(opened['latency'], 0.022)
        stream.start_stream.assert_not_called()
        self.sd.RawInputStream.assert_not_called()

    def test_loopback_open_failure_propagates(self):
        self.backend.PyAudio.return_value.open.side_effect = OSError('Disconnected')
        with self.assertRaises(OSError):
            open_capture(self.audio, device(6, 'L', 2, True), None)


class SoundDeviceStreamTests(unittest.TestCase):
    def test_pyaudio_lifecycle_and_idempotent_close(self):
        inner = Mock(active=True)
        stream = SoundDeviceStream(inner)
        stream.start_stream()
        self.assertTrue(stream.is_active())
        stream.close()
        stream.close()
        stream.stop_stream()
        self.assertFalse(stream.is_active())
        inner.start.assert_called_once()
        inner.close.assert_called_once()
        inner.stop.assert_not_called()


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = [device(1, 'Microphone'), device(8, 'Speakers [Loopback]', 8, True, 44100)]
        self.pa = Mock()

    def test_selection_uses_device_index(self):
        self.assertEqual(resolve_device(self.pa, self.devices, '8')['index'], 8)
        self.assertEqual(resolve_device(self.pa, self.devices, 'exact:Microphone')['index'], 1)
        self.assertEqual(resolve_device(self.pa, self.devices, 'micro')['index'], 1)

    def test_missing_and_ambiguous_names_fail(self):
        devices = self.devices + [device(9, 'Speakers 2')]
        for selector in ('999', 'exact:Missing', 'Speakers', '-1'):
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                resolve_device(self.pa, devices, selector)

    def test_default_loopback(self):
        self.pa.get_default_wasapi_loopback.return_value = self.devices[1]
        self.assertEqual(resolve_device(self.pa, self.devices, '--loopback')['index'], 8)
        with self.assertRaises(ValueError):
            resolve_device(object(), self.devices, '--loopback')

    def test_duplicate_exact_names_warn_and_list_numbered_choices(self):
        devices = [device(3, 'Microphone'), device(9, 'Microphone')]
        with self.assertRaises(ValueError) as error:
            resolve_device(self.pa, devices, 'exact:Microphone')
        self.assertIn('Warning: 2 capture devices', str(error.exception))
        self.assertIn('exact:Microphone:1', str(error.exception))
        self.assertIn('exact:Microphone:2', str(error.exception))

    def test_exact_occurrence_counts_only_same_names_in_listing_order(self):
        devices = [device(9, 'Mic: USB'), device(2, 'Other'), device(3, 'Mic: USB')]
        for number, index in ((1, 9), (2, 3)):
            self.assertEqual(resolve_device(
                self.pa, devices, f'exact:Mic: USB:{number}')['index'], index)
        for suffix in ('0', '3', '-1', 'bad', ''):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                resolve_device(self.pa, devices, f'exact:Mic: USB:{suffix}')
        with self.assertRaises(ValueError):
            resolve_device(self.pa, devices, 'exact:Missing:1')

    def test_exact_numeric_suffix_in_full_name_is_preserved(self):
        devices = [device(3, 'Mic'), device(9, 'Mic:1')]
        self.assertEqual(resolve_device(self.pa, devices, 'exact:Mic:1')['index'], 9)
        self.assertEqual(resolve_device(self.pa, devices, 'exact:Mic:1:1')['index'], 9)

    def test_listing_shows_occurrence_selectors_for_duplicate_names(self):
        devices = [device(9, 'Mic'), device(2, 'Other'), device(3, 'Mic', api='MME')]
        output = StringIO()
        with redirect_stdout(output):
            print_devices(devices)
        lines = output.getvalue().splitlines()
        self.assertIn('[exact:Mic:1]', lines[0])
        self.assertNotIn('[exact:', lines[1])
        self.assertIn('[exact:Mic:2]', lines[2])


if __name__ == '__main__':
    unittest.main()
