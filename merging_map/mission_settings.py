"""
mission_settings.py - The distances and delays of a mission, in two sets: FIELD and ARENA.

The brain (cerveau.mission) and the map's inspection (fusion_carte.CarteFusion) used to keep
these as constants sized for the field: a 25 m forest, where a lost rover link comes from a
place (behind a ridge, a radio dead zone). In the 2.4 m arena of the demo several of them
are absurd: a 1.5 m no-go zone after a link loss blocks the whole floor, while a Wi-Fi glitch
in a room has nothing to do with where the rover is.

Only the values that depend on the size of the place differ between the two sets. Those that
depend on the rover or its sensors are the same in both (careful step, wait for the drone),
and the rover radius and the planning cell size stay with the map
(CarteFusion(rayon_robot=..., niveau_plan=...), lancer_overhead.REGLAGES_PETITE_ZONE).

    from mission_settings import ARENA
    mission(robot, carte, start, victim, settings=ARENA)

FIELD holds the values used so far, and stays the default. The ARENA values are orders of
magnitude, to check on the first test drive with the real robot.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MissionSettings:
    # ---- the same everywhere (rover and sensors) ------------------------------------------
    careful_step: float = 0.15  # m per step without the drone
    drone_wait_max: float = 300.0  # s waiting for the drone at the return point, then stop

    # ---- depends on the size of the place -------------------------------------------------
    # m driven per step while the drone sees the rover. The rover drives straight to the end
    # of the step, so a corner of the path inside it is cut: by about 11 cm at 0.3 m on a
    # right-angle turn, 7 cm at 0.2 m. Compare with the margin around obstacles.
    step: float = 0.3
    # m of possible position error without the drone before going back to the last point it
    # saw (unless the victim is closer). Compare with the width of the passages.
    uncertainty_budget: float = 0.6
    # Rover link lost: the rover goes back this far before the last point where the link
    # worked. 0: back to that point.
    link_backoff: float = 0.5
    # Rover link lost: radius of the no-go zone marked just past the point where the link was
    # lost. 0: no zone at all, the link loss is not caused by a place (Wi-Fi in a room).
    no_go_radius: float = 1.5
    # m of path ahead checked against newly seen obstacles after every step
    check_horizon: float = 3.0
    # Without the drone the rover goes to look at doubtful obstacles itself. It stops at least
    # inspect_reserve m outside the safety margin of the obstacle, and at most
    # inspect_max_distance m from it (or inspect_max_distance / 2 past the nearest allowed point).
    inspect_reserve: float = 0.35
    inspect_max_distance: float = 1.2

    def apply_to_map(self, carte):
        """Give the map the settings it uses itself (the rover's own inspection)."""
        carte.inspect_reserve = self.inspect_reserve
        carte.inspect_max_distance = self.inspect_max_distance


FIELD = MissionSettings()

ARENA = MissionSettings(
    step=0.2,  # to try: less corner cutting between obstacles 30 to 40 cm apart
    uncertainty_budget=0.15,  # passages of 30 to 40 cm
    link_backoff=0.0,
    no_go_radius=0.0,  # a Wi-Fi glitch, not a dead zone: wait for the link and go on
    check_horizon=1.0,
    inspect_reserve=0.15,
    inspect_max_distance=0.6,
)
