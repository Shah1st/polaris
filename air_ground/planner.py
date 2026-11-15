"""The route around obstacles, with the team's planner (merging_map/fusion_carte.py).

The obstacles come from the floor map seen from the air (floor_map.py: heights, and the
team's colour detection); the planning is the team's: obstacles grown by the rover's radius
with the exact Minkowski sum (`gonfler`), then `CarteFusion.astar` (8-connected, unknown cells
cost 3 times more, no corner cutting, and a way out when the rover starts inside a grown
zone). The rover's radius is the one the team tested in the small arena (0.15 m, commit
a7d09cc).

    planner = RoutePlanner(radius=0.15)
    route = planner.route(grid, start=(x, y), goal=(gx, gy))   # floor points, or None
    planner.blocked(grid, route)                               # first point now blocked

`grid` is FloorMap.grid(): {'res', 'x0', 'y0', 'cells' (row 0 = south, -1/0/1)}.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'merging_map')
)
from fusion_carte import OBSTACLE, CarteFusion, GrillePlan, gonfler  # noqa: E402


class RoutePlanner:
    def __init__(
        self,
        radius: float = 0.15,
        margin: float = 0.0,
        unknown_cost: float = 3.0,
        tolerance: float = 0.05,
    ) -> None:
        """`radius`: the circle around the whole rover seen from above (m). `margin`: more
        room for what the map does not know (m). `tolerance`: the goal counts as reached in
        any free cell this close (m)."""
        self.carte = CarteFusion(rayon_robot=radius, cout_inconnu=unknown_cost,
                                 marge_securite=margin)  # fmt: skip
        self.tolerance = tolerance
        self.last_grid = None
        self._plan = None

    @property
    def radius(self) -> float:
        return self.carte.rayon_effectif

    def _plan_of(self, grid: dict) -> GrillePlan:
        """The grid as the team's planner wants it: its own frame starts at the map's corner
        (x0, y0), so that the cells line up whatever the map's place on the floor."""
        if grid is not self.last_grid:
            cells = np.asarray(grid['cells'], np.int8)
            blocked = gonfler(cells == OBSTACLE, self.radius, grid['res'])
            self._plan = GrillePlan(cells, blocked, 0, 0, grid['res'])
            self.last_grid = grid
        return self._plan

    def blocked_cells(self, grid: dict) -> np.ndarray:
        """The cells the rover's centre may not enter (obstacles grown by its radius)."""
        return self._plan_of(grid).bloque

    def route(self, grid: dict, start, goal) -> list[tuple[float, float]] | None:
        """The way from `start` to `goal` (floor frame, m) around the obstacles, as few points
        as possible, ending on the goal itself when its cell is free; None when there is none
        (the goal is in an obstacle or walled in)."""
        x0, y0 = grid['x0'], grid['y0']
        plan = self._plan_of(grid)
        s = (start[0] - x0, start[1] - y0)
        g = (goal[0] - x0, goal[1] - y0)
        cells = self.carte.astar(plan, s, g, self.tolerance)
        if cells is None:
            return None
        blocked = self.carte._liberer_depart(plan, *plan.ij(*s))
        gi, gj = plan.ij(*g)
        if plan.dedans(gi, gj) and not blocked[gi, gj]:
            cells = [*cells[:-1], g]  # the goal itself, not its cell's centre
        points = self._shortcut([s, *cells[1:]], plan, blocked)
        return [(x + x0, y + y0) for x, y in points[1:]]

    @staticmethod
    def _clear(a, b, plan: GrillePlan, blocked: np.ndarray) -> bool:
        """The straight line from a to b crosses no blocked cell (sampled every 1/3 cell)."""
        n = max(1, math.ceil(math.dist(a, b) / (plan.res / 3)))
        for k in range(1, n + 1):
            x = a[0] + (b[0] - a[0]) * k / n
            y = a[1] + (b[1] - a[1]) * k / n
            i, j = plan.ij(x, y)
            if not plan.dedans(i, j) or blocked[i, j]:
                return False
        return True

    def _shortcut(self, points, plan, blocked):
        """The A*'s staircase of cells pulled straight wherever the way is clear: the rover's
        path follower gets a few corners instead of a point every 5 cm."""
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._clear(points[i], points[j], plan, blocked):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def blocked(self, grid: dict, route, start=None) -> tuple[float, float] | None:
        """The first point of `route` (from `start`) that the map now blocks, or None."""
        if not route:
            return None
        x0, y0 = grid['x0'], grid['y0']
        plan = self._plan_of(grid)
        local = [(x - x0, y - y0) for x, y in ([start] if start else []) + list(route)]
        free = self.carte._liberer_depart(plan, *plan.ij(*local[0]))
        for a, b in zip(local, local[1:], strict=False):
            if not self._clear(a, b, plan, free):
                return b[0] + x0, b[1] + y0
        return None
