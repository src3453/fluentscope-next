"""ScreenCaptureKit subprocess adapter with a PyAudio-like stream lifecycle."""
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading


DEVICE_INDEX = -100  # Synthetic endpoint; PortAudio indices are nonnegative.


def system_audio_device():
    return dict(index=DEVICE_INDEX, name='macOS System Audio', maxInputChannels=2,
                defaultSampleRate=48000, api_name='ScreenCaptureKit',
                isLoopbackDevice=True, capture_backend='screencapturekit')


def helper_path():
    if int(platform.mac_ver()[0].split('.')[0] or 0) < 13:
        raise ValueError('System audio capture requires macOS 13 or later.')
    source = Path(__file__).with_name('native') / 'macos_loopback.swift'
    fingerprint = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:16]
    cache = Path.home() / 'Library' / 'Caches' / 'FluentScopeNext'
    binary = cache / f'macos-loopback-{fingerprint}'
    if not binary.exists():
        cache.mkdir(parents=True, exist_ok=True)
        # Compile atomically so simultaneous launches cannot execute a partial binary.
        with tempfile.TemporaryDirectory(dir=cache) as build:
            target = Path(build) / 'macos-loopback'
            try:
                result = subprocess.run(
                    ['xcrun', 'swiftc', '-O', '-module-cache-path', str(cache / 'modules'),
                     str(source), '-o', str(target)], capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired as exc:
                raise OSError('Building the macOS audio helper timed out. Retry after the '
                              'Command Line Tools installation has finished.') from exc
            except OSError as exc:
                raise OSError('Cannot build macOS audio helper. Install Apple Command Line Tools '
                              'with xcode-select --install, then retry.') from exc
            if result.returncode:
                raise OSError(f'Cannot build macOS audio helper: {result.stderr.strip()}')
            os.replace(target, binary)
    return binary


class MacOSLoopbackStream:
    def __init__(self, callback, chunk):
        self.callback = callback
        self.chunk = chunk
        self._closed = False
        self._reader = None
        self._ready = threading.Event()
        self._started = False
        self._errors = []
        self.process = subprocess.Popen([str(helper_path())], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        self._monitor = threading.Thread(target=self._read_status, daemon=True)
        self._monitor.start()
        if not self._ready.wait(30) or not self._started or self.process.poll() is not None:
            self.close()
            detail = '\n'.join(self._errors) or 'Timed out waiting for screen/audio capture permission.'
            raise OSError(detail)

    def _read_status(self):
        for line in self.process.stderr:
            message = line.decode('utf-8', errors='replace').strip()
            if message == 'READY':
                self._started = True
                self._ready.set()
            elif message:
                self._errors.append(message)
        self._ready.set()  # Wake startup immediately when the helper exits.

    def _read_audio(self):
        size = self.chunk * 2 * 2
        try:
            while not self._closed:
                data = self.process.stdout.read(size)
                if len(data) != size:
                    break
                self.callback(data, self.chunk, None, 0)
        except (OSError, ValueError):
            pass  # Closing the pipe wakes a blocked reader during shutdown.

    def start_stream(self):
        if self._closed:
            raise OSError('System audio stream is closed; reopen it to restart.')
        if self._reader is None:
            self._reader = threading.Thread(target=self._read_audio, daemon=True)
            self._reader.start()

    def is_active(self):
        return (not self._closed and self.process.poll() is None
                and self._reader is not None and self._reader.is_alive())

    def stop_stream(self):
        self.close()

    abort_stream = stop_stream

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._reader is not None:
            self._reader.join(timeout=3)
        self._monitor.join(timeout=3)
        self.process.stdout.close()
        self.process.stderr.close()
