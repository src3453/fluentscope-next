"""Backing resolution must not change layout coordinates or apparent trace width."""
import importlib.util
import unittest
from unittest.mock import Mock

import numpy as np

if importlib.util.find_spec('moderngl') is not None:
    from crt_renderer import CRTRenderer
else:
    CRTRenderer = None


@unittest.skipIf(CRTRenderer is None, 'optional moderngl is not installed')
class RetinaRendererTests(unittest.TestCase):
    def renderer(self):
        renderer = CRTRenderer.__new__(CRTRenderer)
        renderer.logical_size = (640, 480)
        renderer.width, renderer.height = 1280, 960
        renderer.ctx = Mock()
        return renderer

    def test_lines_scale_geometry_width_and_clip_to_backing_pixels(self):
        renderer = self.renderer()
        renderer.line_width, renderer.line_aa = 3.0, True
        renderer.line_prog = {key: Mock() for key in ('uViewport', 'uWidth', 'uAA', 'uExtent')}
        renderer.line_resolve_prog = {key: Mock() for key in ('uCoverage', 'uColor')}
        for name in ('_ensure_line_capacity', 'line_seg_buffer', 'coverage_fbo',
                     'line_fbo', 'line_vao', 'coverage_tex', 'line_resolve_vao'):
            setattr(renderer, name, Mock())
        points = np.array([[10, 20], [30, 40]], dtype=np.float32)
        renderer.draw_line(points, (1, 0, 0), scissor=(10, 20, 100, 50))
        segments = np.frombuffer(renderer.line_seg_buffer.write.call_args.args[0], np.float32)
        np.testing.assert_array_equal(segments, [20, 40, 60, 80])
        np.testing.assert_array_equal(points, [[10, 20], [30, 40]])
        self.assertEqual(renderer.line_prog['uWidth'].value, 6.0)
        self.assertEqual(renderer.line_prog['uViewport'].value, (1280, 960))
        self.assertEqual(renderer._pixel_scissor((10, 20, 100, 50)), (20, 820, 200, 100))

    def test_backing_change_rebuilds_targets_with_same_logical_size(self):
        renderer = self.renderer()
        target = Mock()
        renderer._targets = [target]
        renderer._create_targets = Mock()
        renderer.clear = Mock()
        renderer.resize(640, 480, drawable_size=(640, 480))
        target.release.assert_called_once()
        renderer._create_targets.assert_called_once()
        self.assertEqual(renderer.logical_size, (640, 480))
        np.testing.assert_array_equal(renderer.pixel_scale, [1, 1])

    def test_unchanged_backing_keeps_phosphor_and_targets(self):
        renderer = self.renderer()
        renderer._create_targets = Mock()
        renderer.resize(640, 480, drawable_size=(1280, 960))
        renderer._create_targets.assert_not_called()

    def test_ui_and_trace_targets_share_the_selected_render_resolution(self):
        renderer = self.renderer()
        renderer._targets = []
        renderer.ctx.texture.side_effect = lambda size, *args, **kwargs: Mock(size=size)
        renderer._create_targets()
        self.assertEqual(renderer.ui_tex.size, (1280, 960))
        self.assertEqual(renderer.hdr_tex.size, (1280, 960))
        self.assertEqual(renderer.line_tex.size, (1280, 960))
