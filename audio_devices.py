"""Audio source discovery and the startup device picker (no capture on import)."""

import json
from pathlib import Path
import subprocess
import sys


def capture_devices(pa):
    devices = []
    for index in range(pa.get_device_count()):
        info = dict(pa.get_device_info_by_index(index))
        if info['maxInputChannels'] > 0:
            info['index'] = index
            info['api_name'] = pa.get_host_api_info_by_index(info['hostApi'])['name']
            devices.append(info)
    if sys.platform == 'darwin':
        from macos_loopback import system_audio_device
        devices.append(system_audio_device())
    return devices


def resolve_device(pa, devices, selector):
    if selector == '--loopback':
        native = [d for d in devices if d.get('capture_backend') == 'screencapturekit']
        if native:
            return native[0]
        if not hasattr(pa, 'get_default_wasapi_loopback'):
            raise ValueError('Native loopback requires macOS 13+ or PyAudioWPatch on Windows. '
                             'Otherwise select a monitor/virtual input by name or index.')
        index = int(pa.get_default_wasapi_loopback()['index'])
        matches = [d for d in devices if d['index'] == index]
    else:
        try:
            index = int(selector)
        except ValueError:
            if selector.startswith('exact:'):
                name = selector[6:]
                matches = [d for d in devices if d['name'] == name]
                # Preserve full names containing colons, including numeric suffixes.
                if not matches:
                    name, separator, occurrence = name.rpartition(':')
                    if separator and occurrence.isascii() and occurrence.isdecimal():
                        matches = [d for d in devices if d['name'] == name]
                        number = int(occurrence)
                        if not 1 <= number <= len(matches):
                            raise ValueError(
                                f'Exact device occurrence {number} is out of range '
                                f'for {name!r} ({len(matches)} matches; numbering starts at 1).')
                        return matches[number - 1]
                if len(matches) > 1:
                    choices = '\n'.join(
                        f'  "exact:{d["name"]}:{number}" '
                        f'(device index {d["index"]}, {d.get("api_name", "unknown API")})'
                        for number, d in enumerate(matches, 1))
                    raise ValueError(
                        f'Warning: {len(matches)} capture devices have the same exact name. '
                        f'Select an occurrence (1-based) or a device index:\n{choices}')
            else:
                matches = [d for d in devices if selector.casefold() in d['name'].casefold()]
        else:
            matches = [d for d in devices if d['index'] == index]
    if len(matches) != 1:
        raise ValueError('No capture device matched.' if not matches else
                         'Device name is ambiguous; use a device index or a unique name.')
    return matches[0]


def print_devices(devices):
    from collections import Counter

    counts = Counter(d['name'] for d in devices)
    occurrences = Counter()
    for device in devices:
        name = device['name']
        occurrences[name] += 1
        kind = 'Output loopback' if device.get('isLoopbackDevice') else 'Input'
        selector = (f'  [exact:{name}:{occurrences[name]}]'
                    if counts[name] > 1 else '')
        print(f"{device['index']:3}: [{kind} / {device['api_name']}] {name}{selector}")


def open_capture(pa, backend, device, callback, chunk, preferred_rate=48000,
                 preferred_channels=2):
    """Use the endpoint's native format for WASAPI loopback, including surround."""
    if device.get('capture_backend') == 'screencapturekit':
        from macos_loopback import MacOSLoopbackStream
        return MacOSLoopbackStream(callback, chunk), dict(channels=2, rate=48000)
    channels = int(device['maxInputChannels'])
    if not device.get('isLoopbackDevice'):
        channels = min(preferred_channels, 2, channels)
    native_rate = int(device['defaultSampleRate'])
    rates = [native_rate] if device.get('isLoopbackDevice') else list(dict.fromkeys(
        [preferred_rate, native_rate]))
    last_error = None
    for rate in rates:
        kwargs = dict(format=backend.paInt16, channels=channels, rate=rate,
                      input=True, frames_per_buffer=chunk,
                      input_device_index=int(device['index']), stream_callback=callback,
                      start=False)
        try:
            return pa.open(**kwargs), kwargs
        except (OSError, ValueError) as exc:
            last_error = exc
    raise last_error


def choose_device(pa, devices, open_device):
    """Isolate Cocoa Tk from SDL on macOS, including Tk's reopen handlers."""
    if sys.platform != 'darwin':
        return _choose_device(pa, devices, open_device)
    try:
        default_input = int(pa.get_default_input_device_info()['index'])
    except (AttributeError, OSError, ValueError, LookupError):
        default_input = None
    payload = dict(devices=devices, default_input=default_input,
                   selected_index=None, initial_status=None)
    while True:
        picker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--picker'],
            input=json.dumps(payload), capture_output=True, text=True)
        if picker.returncode:
            raise OSError(f'Audio source picker failed: {picker.stderr.strip()}')
        try:
            index = json.loads(picker.stdout)
        except ValueError as exc:
            raise OSError('Audio source picker returned an invalid selection.') from exc
        if index is None:
            return None
        device = next((d for d in devices if d['index'] == index), None)
        if device is None:
            raise OSError('The selected audio device is no longer available.')
        try:
            return device, open_device(device)
        except (OSError, ValueError) as exc:
            payload['selected_index'] = index
            payload['initial_status'] = (
                f'Could not open this device: {exc}. Select another device and retry.')


def _choose_device(pa, devices, open_device, initial_status=None, selected_index=None):
    """Keep the picker open on capture errors so another endpoint can be tried."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('FluentScope Next — Audio source')
    root.geometry('900x480')
    root.minsize(640, 360)
    result = None
    selected_device = next((d for d in devices if d['index'] == selected_index), {})
    mode = tk.StringVar(value='Output loopback' if selected_device.get('isLoopbackDevice')
                        else 'Input')
    status = tk.StringVar(value='Select an audio source, then click Start.')
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Select audio source', font=('', 16, 'bold')).pack(anchor='w')
    modes = ttk.Frame(frame)
    modes.pack(fill='x', pady=12)
    listing = ttk.Frame(frame)
    listing.pack(fill='both', expand=True)
    tree = ttk.Treeview(listing, columns=('api', 'channels', 'rate'), selectmode='browse')
    tree.heading('#0', text='Device')
    tree.column('#0', width=480)
    for key, label, width in [('api', 'Audio API', 130), ('channels', 'Channels', 70),
                              ('rate', 'Native rate', 90)]:
        tree.heading(key, text=label)
        tree.column(key, width=width, minwidth=60, stretch=False)
    tree.pack(side='left', fill='both', expand=True)
    scrollbar = ttk.Scrollbar(listing, orient='vertical', command=tree.yview)
    scrollbar.pack(side='right', fill='y')
    tree.configure(yscrollcommand=scrollbar.set)

    def populate():
        tree.delete(*tree.get_children())
        loopback = mode.get() == 'Output loopback'
        visible = [d for d in devices if bool(d.get('isLoopbackDevice')) == loopback]
        default_index = None
        try:
            default = (resolve_device(pa, devices, '--loopback') if loopback else
                       pa.get_default_input_device_info())
            default_index = int(default['index'])
        except (AttributeError, OSError, ValueError, LookupError):
            pass
        if any(d['index'] == selected_index for d in visible):
            default_index = selected_index
        for d in visible:
            tree.insert('', 'end', iid=str(d['index']), text=f"{d['index']}: {d['name']}",
                        values=(d['api_name'], int(d['maxInputChannels']),
                                f"{int(d['defaultSampleRate'])} Hz"))
        if visible:
            selected = str(default_index if any(d['index'] == default_index for d in visible)
                           else visible[0]['index'])
            tree.selection_set(selected)
            tree.focus(selected)
            tree.see(selected)
        start.configure(state='normal' if visible else 'disabled')
        status.set('Select an audio source, then click Start.' if visible else
                   ('No loopback devices. On Windows, install PyAudioWPatch; on Linux, '
                    'select a monitor/virtual input under Input.' if loopback else
                    'No input devices are available.'))
        if loopback and any(d.get('capture_backend') == 'screencapturekit' for d in visible):
            status.set('macOS 13+: captures system audio without a driver. On first use, '
                       'builds a helper using Apple Command Line Tools. Allow Screen & '
                       'System Audio Recording in System Settings > Privacy & Security.')

    for label in ('Input', 'Output loopback'):
        ttk.Radiobutton(modes, text=label, variable=mode, value=label,
                        command=populate).pack(side='left', padx=(0, 20))
    ttk.Label(frame, textvariable=status, wraplength=820).pack(fill='x', pady=12)
    buttons = ttk.Frame(frame)
    buttons.pack(fill='x')

    def launch(event=None):
        nonlocal result
        selection = tree.selection()
        if not selection:
            return
        device = next(d for d in devices if d['index'] == int(selection[0]))
        try:
            opened = open_device(device)
        except (OSError, ValueError) as exc:
            status.set(f'Could not open this device: {exc}. Select another device and retry.')
            return
        result = device, opened
        root.destroy()

    start = ttk.Button(buttons, text='Start', command=launch)
    start.pack(side='right')
    ttk.Button(buttons, text='Cancel', command=root.destroy).pack(side='right', padx=8)
    tree.bind('<Double-1>', launch)
    root.bind('<Return>', launch)
    root.bind('<Escape>', lambda event: root.destroy())
    populate()
    if initial_status:
        status.set(initial_status)
    tree.focus_set()
    root.mainloop()
    return result


if __name__ == '__main__' and sys.argv[1:] == ['--picker']:
    from types import SimpleNamespace

    payload = json.load(sys.stdin)
    pa = SimpleNamespace(get_default_input_device_info=lambda: dict(
        index=payload['default_input'] if payload['default_input'] is not None else -1))
    result = _choose_device(
        pa, payload['devices'], lambda device: device['index'],
        payload['initial_status'], payload['selected_index'])
    print(json.dumps(result[0]['index'] if result else None))
