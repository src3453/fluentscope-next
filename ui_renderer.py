"""Retained pyglet batches with layout coordinates measured from the top left.

The application records drawing commands in logical points. Batches are built
at the target's pixel density, including font glyphs, and unchanged commands
reuse their GPU objects across frames.
"""
from dataclasses import dataclass
import math
import sys

import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics import Batch, Group
from pyglet.math import Mat4


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2

    def contains(self, other):
        return (self.x <= other.x and self.y <= other.y
                and self.x + self.width >= other.x + other.width
                and self.bottom >= other.bottom)

    def collidepoint(self, pos):
        x, y = pos
        return self.x <= x < self.x + self.width and self.y <= y < self.bottom


def rgba(color):
    values = tuple(int(v) for v in color)
    return values if len(values) == 4 else (*values, 255)


class Font:
    def __init__(self, size):
        self.name = 'Menlo' if sys.platform == 'darwin' else ('Consolas', 'DejaVu Sans Mono', 'monospace')
        self.pixels = size
        self._font = pyglet.font.load(self.name, size * 0.75, dpi=96)
        self._sizes = {}

    def size(self, text):
        if text in self._sizes:
            return self._sizes[text]
        glyphs, positions = self._font.get_glyphs(text)
        measured = math.ceil(sum(g.advance + p.x_advance for g, p in zip(glyphs, positions))), self.get_height()
        if len(self._sizes) >= 512:
            self._sizes.pop(next(iter(self._sizes)))
        self._sizes[text] = measured
        return measured

    def get_height(self):
        return self._font.ascent - self._font.descent

    def get_linesize(self):
        return self.get_height()

    def render(self, text, antialias, color):
        return Text(self, text, rgba(color))


@dataclass(frozen=True)
class Text:
    font: Font
    text: str
    color: tuple

    def get_width(self):
        return self.font.size(self.text)[0]

    def get_height(self):
        return self.font.get_height()

    def get_rect(self, *, center=None, midleft=None):
        w, h = self.get_width(), self.get_height()
        if center is not None:
            return Rect(center[0] - w / 2, center[1] - h / 2, w, h)
        return Rect(midleft[0], midleft[1] - h / 2, w, h)


class Image:
    """A top-down RGB NumPy image, uploaded only after its version changes."""
    def __init__(self, width, height):
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)
        self.version = 0
        self.texture = None
        self._uploaded_version = -1

    def changed(self):
        self.version += 1

    def get_texture(self):
        h, w, _ = self.pixels.shape
        if self.texture is None:
            self.texture = pyglet.image.Texture.create(w, h)
            gl.glBindTexture(self.texture.target, self.texture.id)
            gl.glTexParameteri(self.texture.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(self.texture.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        if self._uploaded_version != self.version:
            data = pyglet.image.ImageData(w, h, 'RGB', self.pixels.tobytes(), pitch=-w * 3)
            self.texture.blit_into(data, 0, 0, 0)
            self._uploaded_version = self.version
        return self.texture

    def delete(self):
        if self.texture is not None:
            self.texture.delete()
            self.texture = None


class ClipGroup(Group):
    def __init__(self, order, clip):
        super().__init__(order=order)
        self.clip = clip

    def set_state(self):
        if self.clip is not None:
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glScissor(*self.clip)
        else:
            gl.glDisable(gl.GL_SCISSOR_TEST)

    def unset_state(self):
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class Canvas:
    def __init__(self, size):
        self.width, self.height = size
        self.commands = []
        self.clip = None
        self.background = (0, 0, 0, 0)
        self.batch = Batch()
        self._slots = []
        self._density = None

    def get_height(self):
        return self.height

    def fill(self, color):
        self.commands.clear()
        self.clip = None
        self.background = rgba(color)

    def set_clip(self, rect):
        self.clip = rect

    def blit(self, source, position):
        if isinstance(position, Rect):
            position = (position.x, position.y)
        self.commands.append(('blit', source, tuple(position), self.clip))

    def _shape(self, kind, color, points, width=0, closed=False):
        self.commands.append((kind, rgba(color), tuple(tuple(map(float, p)) for p in points),
                              float(width), closed, self.clip))

    def _delete_slots(self, start=0):
        for _, obj in self._slots[start:]:
            obj.delete()
        del self._slots[start:]

    def delete(self):
        self._delete_slots()

    def render(self, window, *, target=None, framebuffer_size=None, clear=True):
        pixels = framebuffer_size or window.get_framebuffer_size()
        pw, ph = pixels
        sx, sy = pw / self.width, ph / self.height
        density = (sx, sy)
        if density != self._density:
            self._delete_slots()
            self._density = density
        window.projection = Mat4.orthogonal_projection(0, pw, 0, ph, -255, 255)
        window.view = Mat4()
        if target is not None:
            target.use()
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, pw, ph)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        if clear:
            gl.glClearColor(*(c / 255 for c in self.background))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        for index, command in enumerate(self.commands):
            # Image contents can change without replacing its sprite.
            if command[0] == 'blit' and isinstance(command[1], Image):
                command[1].get_texture()
            if index < len(self._slots) and self._slots[index][0] == command:
                continue
            if index < len(self._slots):
                self._slots[index][1].delete()
            clip = command[-1]
            if clip is not None:
                x0, x1 = round(clip.x * sx), round((clip.x + clip.width) * sx)
                y0, y1 = round(clip.y * sy), round(clip.bottom * sy)
                clip = (x0, ph - y1, max(0, x1 - x0), max(0, y1 - y0))
            group = ClipGroup(index, clip)
            obj = self._build(command, group, sx, sy, ph)
            slot = (command, obj)
            if index == len(self._slots):
                self._slots.append(slot)
            else:
                self._slots[index] = slot
        self._delete_slots(len(self.commands))
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _build(self, command, group, sx, sy, ph):
        kind = command[0]
        common = dict(batch=self.batch, group=group)
        if kind == 'blit':
            source, (x, y) = command[1:3]
            if isinstance(source, Text):
                return pyglet.text.Label(source.text, font_name=source.font.name,
                                         font_size=source.font.pixels * 0.75, dpi=96 * sy,
                                         x=x * sx, y=ph - y * sy, anchor_y='top',
                                         color=source.color, **common)
            sprite = pyglet.sprite.Sprite(source.get_texture(), x=x * sx,
                                           y=ph - (y + source.pixels.shape[0]) * sy, **common)
            sprite.scale_x, sprite.scale_y = sx, sy
            return sprite
        color, points, width, closed = command[1:5]
        coordinates = [(x * sx, ph - y * sy) for x, y in points]
        if kind == 'rect':
            (x, top), (right, bottom) = coordinates
            return pyglet.shapes.Rectangle(x, bottom, right - x, top - bottom,
                                            color=color, **common)
        if kind == 'polygon':
            return pyglet.shapes.Polygon(*coordinates, color=color, **common)
        return pyglet.shapes.MultiLine(*coordinates, thickness=width * sx,
                                       closed=closed, color=color, **common)


def line(canvas, color, start, end, width=1):
    canvas._shape('lines', color, (start, end), width)


def lines(canvas, color, closed, points, width=1):
    if len(points) >= 2:
        canvas._shape('lines', color, points, width, closed)


def rect(canvas, color, bounds, width=0):
    if not isinstance(bounds, Rect):
        bounds = Rect(*bounds)
    x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height
    if w <= 0 or h <= 0:
        return
    if width:
        # An inward border keeps meter and menu bounds unchanged.
        width = min(width, w / 2, h / 2)
        for border in ((x, y, w, width), (x, y + h - width, w, width),
                       (x, y + width, width, h - 2 * width),
                       (x + w - width, y + width, width, h - 2 * width)):
            rect(canvas, color, border)
    else:
        canvas._shape('rect', color, ((x, y), (x + w, y + h)))


def polygon(canvas, color, points):
    if len(points) >= 3:
        canvas._shape('polygon', color, points)
