"""Audio source discovery and the startup device picker (no capture on import)."""


def capture_devices(pa):
    devices = []
    for index in range(pa.get_device_count()):
        info = dict(pa.get_device_info_by_index(index))
        if info['maxInputChannels'] > 0:
            info['index'] = index
            info['api_name'] = pa.get_host_api_info_by_index(info['hostApi'])['name']
            devices.append(info)
    return devices


def resolve_device(pa, devices, selector):
    if selector == '--loopback':
        if not hasattr(pa, 'get_default_wasapi_loopback'):
            raise ValueError('WASAPI loopback requires PyAudioWPatch on Windows.')
        index = int(pa.get_default_wasapi_loopback()['index'])
        matches = [d for d in devices if d['index'] == index]
    else:
        try:
            index = int(selector)
        except ValueError:
            if selector.startswith('exact:'):
                matches = [d for d in devices if d['name'] == selector[6:]]
            else:
                matches = [d for d in devices if selector.casefold() in d['name'].casefold()]
        else:
            matches = [d for d in devices if d['index'] == index]
    if len(matches) != 1:
        raise ValueError('No capture device matched.' if not matches else
                         'Device name is ambiguous; use a device index or a unique name.')
    return matches[0]


def print_devices(devices):
    for device in devices:
        kind = 'Output loopback' if device.get('isLoopbackDevice') else 'Input'
        print(f"{device['index']:3}: [{kind} / {device['api_name']}] {device['name']}")


def open_capture(pa, backend, device, callback, chunk, preferred_rate=48000,
                 preferred_channels=2):
    """Use the endpoint's native format for WASAPI loopback, including surround."""
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
    """Keep the picker open on capture errors so another endpoint can be tried."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('FluentScope Next — Audio source')
    root.geometry('900x480')
    root.minsize(640, 360)
    result = None
    mode = tk.StringVar(value='Input')
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
            default = (pa.get_default_wasapi_loopback() if loopback else
                       pa.get_default_input_device_info())
            default_index = int(default['index'])
        except (AttributeError, OSError, ValueError, LookupError):
            pass
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
                   ('No loopback devices. On Windows, install PyAudioWPatch; on other '
                    'systems, select a monitor/virtual input under Input.' if loopback else
                    'No input devices are available.'))

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
    tree.focus_set()
    root.mainloop()
    return result
