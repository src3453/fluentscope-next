"""Pyglet window setup and logical/backing size helpers."""
import sys
import platform

_activating_window = False


def configure_windowing():
    # Set these before importing pyglet.window (and before the audio picker).
    import pyglet

    pyglet.options.shadow_window = False
    # In pyglet 2.1 this preserves logical Cocoa window and mouse coordinates
    # while get_framebuffer_size() exposes native backing pixels.
    pyglet.options.dpi_scaling = 'stretch'
    pyglet.options.debug_gl = False
    if sys.platform == 'darwin' and platform.machine() == 'arm64':
        # Pyglet 2.1 automatically uses NSApplication.run only on M1. Use it
        # on newer Apple Silicon too, so Cocoa processes activation events.
        pyglet.options.osx_alt_loop = True


def desktop_size():
    import pyglet

    screen = pyglet.display.get_display().get_default_screen()
    return screen.width, screen.height


def create_window(width, height, *, vsync=False, minimum_size=(640, 480)):
    import pyglet

    config = pyglet.gl.Config(double_buffer=True, major_version=3, minor_version=3,
                              forward_compatible=True, depth_size=0)
    window = pyglet.window.Window(width=width, height=height, caption='FluentScope Next',
                                  resizable=True, vsync=vsync, config=config, visible=False)
    window.set_minimum_size(*minimum_size)
    return window


def activate_window(window):
    """Restore app activation, key window, and input view on a focus request.

    On Cocoa a window can be ordered in front while its app remains inactive.
    Call at startup or from activation/click events, never from drawing/hover.
    """
    global _activating_window
    if _activating_window:
        return
    _activating_window = True
    try:
        if sys.platform != 'darwin':
            window.activate()
            return
        from pyglet.libs.darwin import cocoapy

        app = cocoapy.ObjCClass('NSApplication').sharedApplication()
        if not app.isActive():
            # activateIgnoringOtherApps: (used by pyglet 2.1) is deprecated
            # since macOS 14. Prefer the current app activation API.
            if app.respondsToSelector_(cocoapy.get_selector('activate')):
                app.activate()
            else:
                app.activateIgnoringOtherApps_(True)
        window._nswindow.makeKeyAndOrderFront_(None)
        window._nswindow.makeFirstResponder_(window._nsview)
    finally:
        # makeKeyAndOrderFront: may synchronously emit on_activate again.
        _activating_window = False


def render_size(logical_size, framebuffer_size, retina, *, platform=None):
    """Retina changes offscreen resolution, never the native window/context."""
    platform = sys.platform if platform is None else platform
    return framebuffer_size if platform != 'darwin' or retina else logical_size


def mouse_position(x, y, height):
    """Pyglet's bottom-left coordinates to the application's top-left layout."""
    return x, height - y
