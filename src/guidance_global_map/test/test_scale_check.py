import pytest

from guidance_global_map.scale_check import Seen, find_mover, scale_correction, size_matches

BOX = Seen(1.0, 1.0, 0.27, 0.36)
PACK = Seen(0.4, 0.8, 0.13, 0.15)


def test_the_object_that_moved_is_found_among_the_still_ones() -> None:
    before = [BOX, PACK, Seen(0.60, 0.37, 0.20, 0.23)]
    after = [Seen(1.01, 0.99, 0.27, 0.36), PACK, Seen(0.60, 0.52, 0.21, 0.23)]

    was, now = find_mover(before, after)

    assert (was.x, was.y) == (0.60, 0.37)
    assert (now.x, now.y) == (0.60, 0.52)


def test_no_mover_when_nothing_moved_or_when_it_is_not_clear() -> None:
    rover = Seen(0.60, 0.37, 0.20, 0.23)
    assert find_mover([BOX, rover], [BOX, rover]) is None
    # two things moved
    assert (
        find_mover([BOX, rover], [Seen(1.3, 1.0, 0.27, 0.36), Seen(0.6, 0.6, 0.2, 0.23)]) is None
    )
    # the rover left the view
    assert find_mover([BOX, rover], [BOX]) is None


def test_correction_is_driven_over_seen() -> None:
    # the camera's scale is 1.55 times too large: a 0.15 m move looks like 0.2325 m
    assert scale_correction(0.2325, 0.15) == pytest.approx(1 / 1.55)
    assert scale_correction(0.15, 0.15) == 1.0
    with pytest.raises(ValueError, match='no move'):
        scale_correction(0.0, 0.15)


def test_size_check_at_the_corrected_scale() -> None:
    # the pack looks like the rover at the wrong scale, and like itself once corrected
    pack_seen_too_large = Seen(0.6, 1.2, 0.13 * 1.55, 0.15 * 1.55)
    assert size_matches(pack_seen_too_large, 1.0, (0.20, 0.23))
    assert not size_matches(pack_seen_too_large, 1 / 1.55, (0.20, 0.23))
    assert size_matches(Seen(0, 0, 0.31, 0.36), 1 / 1.55, (0.23, 0.20))
