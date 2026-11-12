"""Pyglet window setup and logical/backing size helpers."""
import sys


def configure_windowing():
    # Set these before importing pyglet.window (and before the audio picker).
    import pyglet

    pyglet.options.shadow_window = False
    # In pyglet 2.1 this preserves logical Cocoa window and mouse coordinates
    # while get_framebuffer_size() exposes native backing pixels.
    pyglet.options.dpi_scaling = 'stretch'
    pyglet.options.debug_gl = False


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


def render_size(logical_size, framebuffer_size, retina, *, platform=None):
    """Retina changes offscreen resolution, never the native window/context."""
    platform = sys.platform if platform is None else platform
    return framebuffer_size if platform != 'darwin' or retina else logical_size


def mouse_position(x, y, height):
    """Pyglet's bottom-left coordinates to the application's top-left layout."""
    return x, height - y
