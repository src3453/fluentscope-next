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


def focus_window():
    """Activate on startup or a user focus request, never on mere mouse hover."""
    global _display_window
    if sys.platform != 'darwin':
        return
    import pygame
    from pygame._sdl2 import Window

    try:
        # pygame stores a borrowed pointer to this wrapper in SDL event data.
        # Keep it alive for the entire display lifetime.
        if _display_window is None:
            _display_window = Window.from_display_module()
        _display_window.focus()
    except pygame.error as exc:
        print(f'Warning: could not activate the window: {exc}')


def ui_font(size):
    import pygame

    if sys.platform == 'darwin':
        # SysFont can launch X11's fc-list and block the Cocoa event loop.
        path = Path('/System/Library/Fonts/Menlo.ttc')
        return pygame.font.Font(str(path) if path.is_file() else None, size)
    return pygame.font.SysFont('consolas', size)
