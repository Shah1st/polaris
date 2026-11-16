"""
suivi_continu.py - Driving without stopping: what the planner sends to the rover while it
moves (sim/fusion_reelle.py --continu, with avancer_esquive.RoverSegments.lancer / suivi).

The rover follows a path at 20 Hz by itself and stops by itself when its depth camera sees an
obstacle on it (rover/rover_core.py, local_guard). Here, about once a second:

- `lisser` rounds the corners of the planned segment, so that the rover's follower drives them
  as arcs instead of stopping to pivot (it stops on corners sharper than 45 degrees);
- `a_remplacer` says whether the new plan is worth sending in place of the path being driven
  (a path sent again takes over at the rover's speed, see POST /path "id");
- `prolonger` starts the new plan a little ahead of the rover, on the path it is driving: the
  rover has moved on by the time the plan arrives;
- `grille_locale` places the rover's own obstacle grid (GET /local_grid, odometry frame) on
  the maps.

No HTTP and no thread here: plain functions, tested in test_suivi_continu.py.
"""

import math

from perception_rover import vers_carte

ANGLE_MAX = math.radians(40.0)  # under the rover's path_corner_deg (45): no stop on a corner
COUPE = 0.15  # m cut off each side of a corner, at most
PAS = 0.05  # m between two samples of a path


def longueur(chemin):
    return sum(math.dist(a, b) for a, b in zip(chemin, chemin[1:], strict=False))


def virage(a, b, c):
    """Change of direction at b on the way a -> b -> c, in radians (0: straight on)."""
    d1 = math.atan2(b[1] - a[1], b[0] - a[0])
    d2 = math.atan2(c[1] - b[1], c[0] - b[0])
    return abs(math.atan2(math.sin(d2 - d1), math.cos(d2 - d1)))


def lisser(points, libre, angle_max=ANGLE_MAX, coupe=COUPE, tours=3):
    """`points` with its sharp corners cut: a corner turning more than `angle_max` is replaced
    by two points, up to `coupe` m before and after it, when `libre(p, q)` says the short cut
    between them is clear; otherwise the corner stays (the rover will pivot on it). Repeated
    `tours` times: a right angle becomes four bends of 22 degrees."""
    pts = [tuple(p) for p in points]
    if len(pts) < 3:
        return pts
    for _ in range(tours):
        out, change = [pts[0]], False
        for k in range(1, len(pts) - 1):
            a, b, c = out[-1], pts[k], pts[k + 1]
            la, lc = math.dist(a, b), math.dist(b, c)
            d = min(coupe, la / 2, lc / 2)
            if d < 0.02 or virage(a, b, c) <= angle_max:
                out.append(b)
                continue
            p = (b[0] + (a[0] - b[0]) * d / la, b[1] + (a[1] - b[1]) * d / la)
            q = (b[0] + (c[0] - b[0]) * d / lc, b[1] + (c[1] - b[1]) * d / lc)
            if libre(p, q):
                out += [p, q]
                change = True
            else:
                out.append(b)
        out.append(pts[-1])
        pts = out
        if not change:
            break
    return pts


def projeter(chemin, pos):
    """(distance from the start of `chemin` to the point of it closest to `pos`, that point,
    its distance to `pos`)."""
    meilleur = (0.0, tuple(chemin[0]), math.dist(pos, chemin[0]))
    s = 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        ux, uy = b[0] - a[0], b[1] - a[1]
        n2 = ux * ux + uy * uy
        u = 0.0 if n2 == 0 else ((pos[0] - a[0]) * ux + (pos[1] - a[1]) * uy) / n2
        u = max(0.0, min(1.0, u))
        q = (a[0] + u * ux, a[1] + u * uy)
        d = math.dist(pos, q)
        if d < meilleur[2]:
            meilleur = (s + u * math.sqrt(n2), q, d)
        s += math.sqrt(n2)
    return meilleur


def point_a(chemin, s):
    """The point of `chemin` at distance `s` from its start (its ends beyond them)."""
    fait = 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        d = math.dist(a, b)
        if d > 0 and s <= fait + d:
            u = max(0.0, (s - fait) / d)
            return (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
        fait += d
    return tuple(chemin[-1])


def entre(chemin, s0, s1):
    """The part of `chemin` between the distances s0 and s1 from its start."""
    total = longueur(chemin)
    s0, s1 = max(0.0, min(s0, total)), max(0.0, min(s1, total))
    out, s = [point_a(chemin, s0)], 0.0
    for a, b in zip(chemin, chemin[1:], strict=False):
        s += math.dist(a, b)
        if s0 < s < s1:
            out.append(tuple(b))
    out.append(point_a(chemin, s1))
    return [p for k, p in enumerate(out) if k == 0 or math.dist(p, out[k - 1]) > 1e-9]


def corde_libre(plan, grille, vu=True, depart=None, rayon=0.0):
    """libre(a, b) for `lisser`: True when the straight line a -> b stays out of the obstacles
    grown by the rover's radius (`plan`, CarteFusion.grille_plan) and, with `vu`, only crosses
    cells seen free (`grille`). Within `rayon` of `depart` nothing is checked: the cameras do
    not see under the rover."""

    def libre(a, b):
        n = max(1, math.ceil(math.dist(a, b) / (grille.res / 2)))
        for k in range(n + 1):
            p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            if depart is not None and math.dist(p, depart) <= rayon:
                continue
            i, j = plan.ij(*p)
            if not plan.dedans(i, j) or plan.bloque[i, j]:
                return False
            if vu:
                gi = math.floor(p[1] / grille.res) - grille.i0
                gj = math.floor(p[0] / grille.res) - grille.j0
                h, w = grille.val.shape
                if not (0 <= gi < h and 0 <= gj < w) or grille.val[gi, gj] != 0:
                    return False
        return True

    return libre


def reste(chemin, pos):
    """`chemin` from the point closest to `pos` to its end."""
    s, _, _ = projeter(chemin, pos)
    return entre(chemin, s, math.inf)


def prolonger(courant, pos, avance):
    """Where to plan from while the rover drives `courant`: (the piece of `courant` from the
    rover to `avance` m further, its last point). The new plan starts at that last point and
    the piece is put in front of it, so the path sent passes where the rover is."""
    s, _, _ = projeter(courant, pos)
    morceau = entre(courant, s, s + avance)
    return morceau, morceau[-1]


def a_remplacer(courant, nouveau, pos, objectif, gain=0.2, tolerance=None):
    """True when `nouveau` is worth sending in place of `courant`, the path the rover (at
    `pos`) is driving towards `objectif`: it ends `gain` m nearer to the goal (more floor has
    been seen), or it ends as near and is `gain` m shorter, or it reaches the goal (within
    `tolerance`) when the one being driven stops short of it. A path that merely differs is
    not sent: the planner's route flickers from one plan to the next, the rover should not.
    (A path the map now bars is the caller's business: CarteFusion.premier_point_bloque.)"""
    vieux, neuf = reste(courant, pos), reste(nouveau, pos)
    fin_v, fin_n = math.dist(vieux[-1], objectif), math.dist(neuf[-1], objectif)
    if fin_n < fin_v - gain:
        return True
    if tolerance is not None and fin_n <= tolerance + 0.02 and fin_v > tolerance + 0.05:
        return True  # the last stretch: without it the rover stops just short, then hops
    return fin_n <= fin_v + 0.05 and longueur(neuf) < longueur(vieux) - gain


def grille_locale(reponse, origine, angle, age_max=2.0):
    """The answer of the rover's GET /local_grid (cells in its odometry frame) as
    (matrix, x0, y0) on the grid of the maps, or None when it is missing or too old.
    `origine`, `angle`: where the origin of the odometry frame is on the maps, and the
    direction of its x axis there."""
    if not reponse or not reponse.get('cells'):
        return None
    age = reponse.get('age_s')
    if age is None or float(age) > age_max:
        return None
    return vers_carte(
        reponse['cells'],
        float(reponse['x0']),
        float(reponse['y0']),
        float(reponse['res']),
        origine,
        angle,
    )
