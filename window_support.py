"""Platform-specific SDL window activation and UI font initialization."""
import os
from pathlib import Path
import sys

_display_window = None


def configure_windowing():
    """Set Cocoa hints before pygame initializes SDL video."""
    if sys.platform == 'darwin':
        os.environ.setdefault('SDL_MAC_BACKGROUND_APP', '0')
        # A click that activates the app should also operate the clicked control.
        os.environ.setdefault('SDL_MOUSE_FOCUS_CLICKTHROUGH', '1')


def _window():
    """The SDL window of pygame.display (raises pygame.error without one)."""
    global _display_window
    from pygame._sdl2 import Window

    # pygame stores a borrowed pointer to this wrapper in SDL event data.
    # Keep it alive for the entire display lifetime.
    if _display_window is None:
        _display_window = Window.from_display_module()
    return _display_window


def focus_window():
    """Activate on startup or a user focus request, never on mere mouse hover."""
    if sys.platform != 'darwin':
        return
    import pygame

    try:
        _window().focus()
    except pygame.error as exc:
        print(f'Warning: could not activate the window: {exc}')


def _sdl_library():
    """The SDL2 library pygame itself loaded (bundled next to it in the wheels)."""
    import ctypes
    import ctypes.util
    import pygame

    base = Path(pygame.__file__).parent
    patterns = ('SDL2.dll', '.dylibs/libSDL2*.dylib', '../pygame.libs/libSDL2*.so*')
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            return ctypes.CDLL(str(path))
    name = ctypes.util.find_library('SDL2')
    if name is None:
        raise OSError('SDL2 library not found')
    return ctypes.CDLL(name)


def set_minimum_window_size(width, height):
    """Keep the user from shrinking the window below what the layout handles."""
    try:
        window = _window()
        if hasattr(window, 'minimum_size'): # pygame-ce
            window.minimum_size = (width, height)
            return
        import ctypes

        sdl = _sdl_library()
        sdl.SDL_GetWindowFromID.restype = ctypes.c_void_p
        sdl.SDL_GetWindowFromID.argtypes = [ctypes.c_uint32]
        sdl.SDL_SetWindowMinimumSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        handle = sdl.SDL_GetWindowFromID(window.id)
        if not handle:
            raise RuntimeError('no SDL window')
        sdl.SDL_SetWindowMinimumSize(handle, width, height)
    except Exception as exc:
        print(f'Warning: could not set the minimum window size: {exc}')


def resize_window(width, height):
    """
    Resize the existing window in place (keeps an OpenGL context alive),
    leaving maximized / fullscreen first. Returns False if it failed.
    """
    try:
        window = _window()
        window.restore()
        window.size = (width, height)
        return True
    except Exception as exc:
        print(f'Warning: could not resize the window: {exc}')
        return False


def ui_font(size):
    import pygame

    if sys.platform == 'darwin':
        # SysFont can launch X11's fc-list and block the Cocoa event loop.
        path = Path('/System/Library/Fonts/Menlo.ttc')
        return pygame.font.Font(str(path) if path.is_file() else None, size)
    return pygame.font.SysFont('consolas', size)
