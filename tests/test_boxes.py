from boxes import clamp_box


def test_box_fully_inside_image_unchanged():
    assert clamp_box(10, 10, 50, 40, width=100, height=100) == (10, 10, 50, 40)


def test_box_clamped_to_image_bounds():
    assert clamp_box(-5, -5, 150, 150, width=100, height=100) == (0, 0, 100, 100)


def test_box_partially_negative_clamped():
    assert clamp_box(-20, 5, 30, 40, width=100, height=100) == (0, 5, 30, 40)


def test_box_beyond_right_bottom_clamped():
    assert clamp_box(80, 80, 200, 200, width=100, height=100) == (80, 80, 100, 100)


def test_zero_width_box_returns_none():
    assert clamp_box(10, 10, 10, 40, width=100, height=100) is None


def test_zero_height_box_returns_none():
    assert clamp_box(10, 10, 40, 10, width=100, height=100) is None


def test_inverted_box_returns_none():
    assert clamp_box(40, 40, 10, 10, width=100, height=100) is None


def test_box_entirely_outside_image_returns_none():
    assert clamp_box(150, 150, 200, 200, width=100, height=100) is None
