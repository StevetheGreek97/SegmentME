"""mask_render.blend_color: the mask fill used by both the full render and the
fast partial redraw. It must give a pixel the same result wherever it sits --
cv2.addWeighted did not on Apple Silicon (SIMD vs row-end rounding), which
made partial redraws differ from full ones by 1."""
import numpy as np
import pytest

from ui.elements.image_display.mask_render import blend_color


@pytest.mark.parametrize("alpha", [0.0, 0.4, 0.85, 1.0])
def test_blend_matches_the_formula(alpha):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    rgb = (200, 30, 120)
    want = np.clip(np.rint(img * (1 - alpha) + np.array(rgb) * alpha), 0, 255)
    assert np.array_equal(blend_color(img, rgb, alpha), want.astype(np.uint8))


def test_blend_does_not_depend_on_position_or_box_width():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (64, 301, 3), dtype=np.uint8)
    full = blend_color(img, (10, 250, 90), 0.85)
    for x0, x1 in [(0, 7), (3, 64), (17, 301), (100, 133), (250, 301)]:
        part = blend_color(img[5:40, x0:x1], (10, 250, 90), 0.85)   # non-contiguous view
        assert np.array_equal(part, full[5:40, x0:x1])
