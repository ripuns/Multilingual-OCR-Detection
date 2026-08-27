"""Regression tests for grouping.text_grouping.group_text.

Covers the min/max bounding-box bug: pre-fix code only ever grew a group's
x2/y2 via max() while x1/y1 stayed pinned to the first box in the group, so
any later box that drifted left or up was absorbed into the group without
the crop rectangle expanding to include it (silent clipping).
"""

from grouping.text_grouping import group_text


def test_empty_input_returns_empty_list():
    assert group_text([]) == []


def test_single_box_returned_unchanged():
    boxes = [(10, 10, 50, 30)]
    assert group_text(boxes) == [(10, 10, 50, 30)]


def test_colinear_boxes_merge_to_exact_bounds():
    # Three boxes on the same line, all drifting rightward -> classic case,
    # already worked pre-fix, kept as a baseline sanity check. y1 values are
    # kept non-decreasing to match the ascending y1 sort group_text performs
    # internally before accumulation.
    boxes = [
        (10, 10, 40, 30),
        (45, 11, 80, 28),
        (85, 12, 120, 29),
    ]
    result = group_text(boxes)
    assert result == [(10, 10, 120, 30)]


def test_left_drifting_box_expands_group_x1():
    """Adversarial case: a later box starts to the LEFT of the group's
    initial x1. Pre-fix, current[0] was never updated, so the merged
    rectangle's left edge stayed at the first box's x1 and clipped the
    left-drifting box's content on crop. Post-fix, x1 must be the min
    across the whole group.
    """
    boxes = [
        (100, 10, 150, 20),  # group seed
        (40, 12, 95, 22),    # drifts left of the seed's x1=100
    ]
    result = group_text(boxes)
    assert len(result) == 1
    x1, y1, x2, y2 = result[0]
    assert x1 == 40  # must include the left-drifting box, not stay at 100
    assert x2 == 150
    assert y1 == 10
    assert y2 == 22


def test_group_y1_matches_topmost_box():
    """group_text sorts all boxes by y1 ascending before accumulating, so a
    group's y1 is always contributed by the first (topmost) box it absorbs
    -- the min() on y1 is a defensive no-op given that invariant, matching
    the master-plan's explicit min/max spec while staying reachable in
    practice only through the topmost box.
    """
    boxes = [
        (10, 20, 60, 40),
        (65, 22, 110, 38),
    ]
    result = group_text(boxes)
    assert len(result) == 1
    assert result[0][1] == 20


def test_left_drift_across_multiple_boxes_table_like_layout():
    """Adversarial case: several boxes on the same row (table-like layout)
    arrive with a left-drifting box in the middle of the group, modeled on
    the master-plan's overlapping/left-drifting scenario. y1 stays
    non-decreasing to respect group_text's internal sort.
    """
    boxes = [
        (200, 50, 260, 62),  # seed
        (150, 52, 195, 64),  # left-drift, would clip pre-fix
        (205, 55, 250, 68),  # further right box
    ]
    result = group_text(boxes)
    assert len(result) == 1
    x1, y1, x2, y2 = result[0]
    assert x1 == 150  # must include the left-drifting box, not stay at 200
    assert y1 == 50
    assert x2 == 260
    assert y2 == 68


def test_far_apart_boxes_do_not_merge():
    boxes = [
        (10, 10, 40, 30),
        (500, 10, 540, 30),  # far beyond h_gap
    ]
    result = group_text(boxes)
    assert len(result) == 2


def test_reading_order_preserved_across_lines():
    boxes = [
        (10, 100, 40, 120),  # second line
        (10, 10, 40, 30),    # first line
        (200, 10, 240, 30),  # first line, second word (out of scan order)
    ]
    result = group_text(boxes)
    # Groups must be ordered top-to-bottom, then left-to-right.
    assert result[0][1] <= result[1][1]
    if result[0][1] == result[1][1]:
        assert result[0][0] <= result[1][0]
