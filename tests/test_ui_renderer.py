"""UI geometry and command recording work without opening a display."""
import unittest

from window_support import configure_windowing
configure_windowing()
import ui_renderer as ui


class UIRendererTests(unittest.TestCase):
    def test_hit_testing_matches_top_left_layout_and_excludes_far_edges(self):
        rect = ui.Rect(10, 20, 100, 30)
        self.assertTrue(rect.collidepoint((10, 20)))
        self.assertTrue(rect.collidepoint((109, 49)))
        self.assertFalse(rect.collidepoint((110, 49)))
        self.assertFalse(rect.collidepoint((109, 50)))
        self.assertEqual(rect.center, (60, 35))
        self.assertTrue(rect.contains(ui.Rect(15, 25, 20, 10)))

    def test_clipping_is_captured_with_each_command_and_reset_for_next_frame(self):
        canvas = ui.Canvas((640, 480))
        canvas.fill((0, 0, 0))
        clip = ui.Rect(10, 20, 100, 200)
        canvas.set_clip(clip)
        ui.line(canvas, (255, 0, 0), (0, 0), (640, 480))
        self.assertEqual(canvas.commands[0][-1], clip)
        canvas.set_clip(None)
        ui.line(canvas, (0, 255, 0), (0, 0), (640, 480))
        self.assertIsNone(canvas.commands[1][-1])
        canvas.fill((5, 10, 15))
        self.assertEqual(canvas.commands, [])
        self.assertIsNone(canvas.clip)
        self.assertEqual(canvas.background, (5, 10, 15, 255))

    def test_border_stays_inside_rectangle_bounds(self):
        canvas = ui.Canvas((640, 480))
        ui.rect(canvas, (255, 255, 255), (10, 20, 100, 50), 2)
        self.assertEqual(len(canvas.commands), 4)
        for command in canvas.commands:
            for x, y in command[2]:
                self.assertTrue(10 <= x <= 110 and 20 <= y <= 70)

    def test_image_changes_are_versioned_without_replacing_pixel_storage(self):
        image = ui.Image(4, 3)
        pixels = image.pixels
        image.pixels[0] = (255, 0, 0)
        image.changed()
        self.assertIs(image.pixels, pixels)
        self.assertEqual(image.version, 1)
        self.assertEqual(image.pixels.shape, (3, 4, 3))
