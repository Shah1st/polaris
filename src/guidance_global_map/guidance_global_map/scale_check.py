"""Check the overhead camera's scale against a known move of the rover. No ROS, no I/O.

The scale (metres per pixel) comes from a number someone typed: a camera height, or a measured
value. A wrong number gives a map of the wrong size, and the rover's size cannot catch it by
itself: with a scale 1.55 times too large, a 13 x 15 cm box measures 20 x 23 cm, the rover's
own size.

A move of the rover can. The wheels say how far it drove; the camera sees how far the thing
that moved went. The ratio of the two corrects the scale, and the thing that moved is the
rover, whatever else in view has its size.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Seen:
    """One object in the camera's view: its centre and its two sides, in the camera's metres
    (which are wrong by the factor this module looks for)."""

    x: float
    y: float
    short: float
    long: float


def find_mover(
    before: list[Seen], after: list[Seen], still: float = 0.04
) -> tuple[Seen, Seen] | None:
    """The one object that is at another place in `after` than in `before`.

    An object is the same, and has not moved, when the two views have it within `still`
    metres. Returns (where it was, where it is), or None when nothing moved or when it is not
    clear what did (an object appeared or vanished as well, two things moved).
    """
    gone = [b for b in before if all(math.dist((b.x, b.y), (a.x, a.y)) > still for a in after)]
    new = [a for a in after if all(math.dist((a.x, a.y), (b.x, b.y)) > still for b in before)]
    if len(gone) != 1 or len(new) != 1:
        return None
    return gone[0], new[0]


def scale_correction(seen_distance: float, driven_distance: float) -> float:
    """Factor to multiply the camera's scale by: the rover drove `driven_distance` metres
    (odometry) and the camera saw it move `seen_distance` of its own metres."""
    if seen_distance <= 0:
        raise ValueError('the camera saw no move')
    return driven_distance / seen_distance


def size_matches(seen: Seen, correction: float, rover_size, tolerance: float = 0.25) -> bool:
    """True when the object, at the corrected scale, has the rover's two sides within
    `tolerance` (a fraction)."""
    short, long = sorted(rover_size)
    return (
        abs(seen.short * correction - short) <= tolerance * short
        and abs(seen.long * correction - long) <= tolerance * long
    )


def frame_from_move(seen_from, seen_to, driven_from, driven_to) -> tuple[float, float]:
    """What one move of the rover says about the camera: (rotation, scale correction).

    The rover went from `seen_from` to `seen_to` in the camera's frame and from `driven_from`
    to `driven_to` in the frame of its odometry (all (x, y), metres). The same displacement
    seen in both frames gives the angle to turn the camera's frame by to line it up with the
    odometry's (radians, in (-pi, pi]), and the factor to multiply the camera's scale by. It
    works whichever way the rover faces, forwards or backwards: only the displacement counts.
    """
    seen = (seen_to[0] - seen_from[0], seen_to[1] - seen_from[1])
    driven = (driven_to[0] - driven_from[0], driven_to[1] - driven_from[1])
    correction = scale_correction(math.hypot(*seen), math.hypot(*driven))
    rotation = math.atan2(driven[1], driven[0]) - math.atan2(seen[1], seen[0])
    return math.atan2(math.sin(rotation), math.cos(rotation)), correction
