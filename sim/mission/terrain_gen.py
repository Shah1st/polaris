"""Stand-in terrain: a stylized low-poly forest island, written as ONE textured GLB.

    python sim/mission/terrain_gen.py out/standin_forest.glb [--size 40] [--seed 7]

It only exists so that the mission simulation can be built and shown before the real terrain
model is there. It goes through exactly the same door as any other model (`terrain.prepare`
reads a GLB and derives every map from the mesh by ray casting): nothing below is given to the
simulation directly, so swapping the terrain is a path change.

Look: flat-shaded, autumn palette (gold / orange / red crowns, pale trunks, green grass), a
rock step across the island with two ramps, groves, clearings, rocks, logs and bushes.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

# name -> RGB. One 8x8 px block each in a 64x64 texture (flat colours, no bleeding).
PALETTE = {
    'grass_a': (104, 168, 72), 'grass_b': (122, 182, 78), 'grass_c': (88, 150, 66),
    'grass_d': (140, 190, 92), 'moss': (96, 138, 70), 'dirt': (176, 142, 98),
    'dirt_b': (160, 126, 86), 'rock_a': (128, 126, 122), 'rock_b': (150, 146, 138),
    'rock_c': (104, 104, 104), 'cliff': (138, 122, 104), 'earth': (112, 86, 62),
    'trunk_a': (226, 214, 196), 'trunk_b': (204, 190, 170), 'trunk_c': (120, 88, 60),
    'gold_a': (240, 190, 60), 'gold_b': (250, 208, 84), 'gold_c': (226, 168, 44),
    'orange_a': (236, 138, 44), 'orange_b': (244, 160, 62), 'red_a': (206, 78, 52),
    'red_b': (224, 100, 60), 'pine_a': (52, 110, 78), 'pine_b': (64, 128, 86),
    'pine_c': (44, 94, 70), 'bush_a': (84, 152, 82), 'bush_b': (188, 96, 60),
    'log': (134, 98, 66), 'log_end': (206, 178, 132),
}  # fmt: skip
NAMES = list(PALETTE)
BLOCK, SIDE = 8, 8  # px per colour block, blocks per texture side


class Builder:
    """Collects triangles with one palette colour per face (flat shading: no shared vertex)."""

    def __init__(self) -> None:
        self.tris: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []

    def add(self, verts: np.ndarray, faces: np.ndarray, colour) -> None:
        tri = np.asarray(verts, float)[np.asarray(faces, int)]
        self.tris.append(tri)
        if isinstance(colour, str):
            col = np.full(len(tri), NAMES.index(colour))
        else:
            col = np.asarray([NAMES.index(c) for c in colour])
        self.cols.append(col)

    def mesh(self):
        import trimesh
        from PIL import Image

        tri = np.concatenate(self.tris)
        col = np.concatenate(self.cols)
        v = tri.reshape(-1, 3)
        f = np.arange(len(v)).reshape(-1, 3)
        img = np.zeros((SIDE * BLOCK, SIDE * BLOCK, 3), np.uint8)
        for k, name in enumerate(NAMES):
            r, c = divmod(k, SIDE)
            img[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK] = PALETTE[name]
        r, c = np.divmod(col, SIDE)
        u = (c + 0.5) / SIDE
        w = 1.0 - (r + 0.5) / SIDE  # image row 0 is the top of the texture
        uv = np.repeat(np.stack([u, w], 1), 3, axis=0)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(img), metallicFactor=0.0, roughnessFactor=0.9
        )
        return trimesh.Trimesh(
            v, f, visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False
        )


def frustum(r0: float, r1: float, h: float, n: int = 7, caps: bool = True):
    """Closed cone frustum along z from 0 to h (r1 = 0: a cone)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring0 = np.stack([r0 * np.cos(a), r0 * np.sin(a), np.zeros(n)], 1)
    ring1 = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full(n, h)], 1)
    v = np.concatenate([ring0, ring1, [[0, 0, 0], [0, 0, h]]])
    f = []
    for k in range(n):
        j = (k + 1) % n
        f += [(k, j, n + j), (k, n + j, n + k)]
        if caps:
            f += [(2 * n, j, k), (2 * n + 1, n + k, n + j)]
    return v, np.array(f)


def icosphere(rng, radius=1.0, jitter=0.0, squash=(1.0, 1.0, 1.0), sub=1):
    import trimesh

    s = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    v = np.array(s.vertices)
    if jitter:
        v *= 1.0 + rng.uniform(-jitter, jitter, (len(v), 1))
    return v * radius * np.asarray(squash), np.array(s.faces)


def rot_z(v, a):
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


class Island:
    def __init__(self, size: float, seed: int) -> None:
        self.size, self.rng = size, np.random.default_rng(seed)
        self.r = size / 2
        self.ph = self.rng.uniform(0, 2 * math.pi, 8)

    # -- shape ---------------------------------------------------------------------------
    def ridge_y(self, x):
        """The rock step runs roughly west-east along this curve."""
        return 0.08 * self.size * np.sin(x * 5.0 / self.size + self.ph[0]) + 0.06 * self.size

    def ramp(self, x):
        """0 on the cliff, 1 in the middle of the two ramps where the step becomes a slope."""
        c = (-0.26 * self.size, 0.22 * self.size)
        return np.maximum(np.exp(-(((x - c[0]) / 2.3) ** 2)), np.exp(-(((x - c[1]) / 2.3) ** 2)))

    def path_d(self, x, y):
        """Distance to the trail (a curve that goes up the west ramp)."""
        px = -0.26 * self.size + 0.05 * self.size * np.sin(y * 6.0 / self.size + self.ph[1])
        return np.abs(x - px)

    def rim(self, x, y):
        a = np.arctan2(y, x)
        r = self.r * (0.93 + 0.035 * np.sin(3 * a + self.ph[2]) + 0.025 * np.sin(5 * a + self.ph[3]))
        return np.hypot(x, y) - r  # > 0 outside the island

    def height(self, x, y):
        s = 40.0 / self.size
        h = 0.55 * np.sin(x * 0.16 * s + self.ph[4]) * np.cos(y * 0.13 * s + self.ph[5])
        h += 0.35 * np.sin((x * 0.07 + y * 0.11) * s + self.ph[6])
        h += 0.12 * np.sin(x * 0.5 * s) * np.sin(y * 0.45 * s + self.ph[7])
        width = 0.12 + 2.6 * self.ramp(x)  # narrow = cliff, wide = ramp
        h += 1.5 / (1.0 + np.exp(-(y - self.ridge_y(x)) / width))
        d = self.rim(x, y)
        edge = np.clip(d / 1.2, 0, 1)
        return h + 3.0 - 7.0 * edge * edge * (3 - 2 * edge)

    def slope_deg(self, x, y, e=0.15):
        gx = (self.height(x + e, y) - self.height(x - e, y)) / (2 * e)
        gy = (self.height(x, y + e) - self.height(x, y - e)) / (2 * e)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    # -- ground --------------------------------------------------------------------------
    def ground(self, b: Builder, cell: float = 0.5) -> None:
        n = int(round(self.size / cell)) + 5
        xs = (np.arange(n) - (n - 1) / 2) * cell
        gx, gy = np.meshgrid(xs, xs)
        jit = self.rng.uniform(-0.12, 0.12, (2, n, n)) * cell
        gx, gy = gx + jit[0], gy + jit[1]
        gz = self.height(gx, gy)
        v = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        idx = np.arange(n * n).reshape(n, n)
        a, bb, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
        flip = self.rng.random(a.shape) < 0.5
        t1 = np.where(flip[..., None], np.stack([a, bb, d], -1), np.stack([a, bb, c], -1))
        t2 = np.where(flip[..., None], np.stack([bb, c, d], -1), np.stack([a, c, d], -1))
        f = np.concatenate([t1.reshape(-1, 3), t2.reshape(-1, 3)])
        tri = v[f]
        centre = tri.mean(1)
        keep = self.rim(centre[:, 0], centre[:, 1]) < 1.6
        f, tri, centre = f[keep], tri[keep], centre[keep]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
        steep = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
        noise = self.rng.random(len(f))
        grass = np.array(['grass_a', 'grass_b', 'grass_c', 'grass_d', 'moss'])
        col = grass[(noise * 4.4).astype(int)]
        on_path = self.path_d(centre[:, 0], centre[:, 1]) < 0.75
        col = np.where(on_path, np.where(noise < 0.5, 'dirt', 'dirt_b'), col)
        col = np.where(steep > 32, np.where(noise < 0.5, 'cliff', 'rock_a'), col)
        outside = self.rim(centre[:, 0], centre[:, 1]) > 0.25
        col = np.where(outside, np.where(noise < 0.6, 'earth', 'rock_c'), col)
        b.add(v, f, list(col))

    # -- things --------------------------------------------------------------------------
    def spots(self, n: int, min_d: float, taken: list, ok) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        tries = 0
        while len(out) < n and tries < n * 400:
            tries += 1
            x, y = self.rng.uniform(-self.r, self.r, 2)
            if self.rim(x, y) > -1.6 or self.slope_deg(x, y) > 14 or not ok(x, y):
                continue
            if any(math.hypot(x - a, y - bb) < max(min_d, d) for a, bb, d in taken):
                continue
            taken.append((x, y, min_d))
            out.append((x, y))
        return out

    def round_tree(self, b: Builder, x: float, y: float) -> None:
        rng = self.rng
        z = float(self.height(x, y))
        h = rng.uniform(1.9, 2.8)
        r = rng.uniform(0.11, 0.17)
        v, f = frustum(r * 1.25, r * 0.8, h + 0.5, 6)
        b.add(v + [x, y, z - 0.25], f, rng.choice(['trunk_a', 'trunk_b']))
        fam = [['gold_a', 'gold_b', 'gold_c'], ['orange_a', 'orange_b', 'gold_a'],
               ['red_a', 'red_b', 'orange_a']][rng.choice(3, p=[0.5, 0.3, 0.2])]  # fmt: skip
        for k in range(rng.integers(2, 4)):
            rad = rng.uniform(0.95, 1.45) * (1.0 if k == 0 else 0.75)
            off = rng.uniform(-0.6, 0.6, 2) * (0 if k == 0 else 1)
            cv, cf = icosphere(rng, rad, 0.12, (1.0, 1.0, rng.uniform(0.62, 0.8)))
            cz = z + h + rad * 0.45 + (0.0 if k == 0 else rng.uniform(-0.2, 0.5))
            cols = [fam[int(c)] for c in rng.integers(0, 3, len(cf))]
            b.add(rot_z(cv, rng.uniform(0, 6.28)) + [x + off[0], y + off[1], cz], cf, cols)

    def pine(self, b: Builder, x: float, y: float) -> None:
        rng = self.rng
        z = float(self.height(x, y))
        top = rng.uniform(4.2, 6.0)
        v, f = frustum(0.16, 0.07, top * 0.6, 6)
        b.add(v + [x, y, z - 0.25], f, 'trunk_c')
        base, levels = rng.uniform(1.15, 1.5), 4
        for k in range(levels):
            z0 = base + (top - base) * k / levels * 0.92
            hh = (top - base) / levels * 1.7
            rad = (1.35 - 0.26 * k) * rng.uniform(0.9, 1.05)
            v, f = frustum(rad, 0.0, hh, 8)
            cols = [['pine_a', 'pine_b', 'pine_c'][int(c)] for c in rng.integers(0, 3, len(f))]
            b.add(rot_z(v, rng.uniform(0, 6.28)) + [x, y, z + z0], f, cols)

    def rock(self, b: Builder, x: float, y: float, big: bool) -> None:
        rng = self.rng
        rad = rng.uniform(0.7, 1.3) if big else rng.uniform(0.22, 0.5)
        v, f = icosphere(rng, rad, 0.22, (1.0, rng.uniform(0.7, 1.0), rng.uniform(0.55, 0.85)))
        cols = [['rock_a', 'rock_b', 'rock_c'][int(c)] for c in rng.integers(0, 3, len(f))]
        z = float(self.height(x, y)) + rad * 0.18
        b.add(rot_z(v, rng.uniform(0, 6.28)) + [x, y, z], f, cols)

    def bush(self, b: Builder, x: float, y: float) -> None:
        rng = self.rng
        rad = rng.uniform(0.3, 0.5)
        fam = ['bush_a', 'bush_a', 'bush_b', 'gold_c'][rng.integers(0, 4)]
        v, f = icosphere(rng, rad, 0.18, (1.0, 1.0, 0.8))
        b.add(v + [x, y, float(self.height(x, y)) + rad * 0.45], f, fam)

    def log(self, b: Builder, x: float, y: float) -> None:
        rng = self.rng
        length, r = rng.uniform(1.4, 2.4), rng.uniform(0.11, 0.16)
        v, f = frustum(r, r * 0.9, length, 7)
        cols = ['log_end' if k % 4 >= 2 else 'log' for k in range(len(f))]
        v = rot_z(rot_y(v - [0, 0, length / 2], math.pi / 2), rng.uniform(0, 3.14))
        b.add(v + [x, y, float(self.height(x, y)) + r * 0.8], f, cols)

    def build(self):
        b = Builder()
        self.ground(b)
        taken: list = []
        s = self.size

        def clear_of_path(x, y):
            return self.path_d(x, y) > 1.3

        def grove(x, y):  # trees gather in groves and leave clearings
            v = math.sin(x * 9.5 / s + self.ph[1]) * math.cos(y * 8.5 / s + self.ph[2])
            return clear_of_path(x, y) and v > -0.25 and math.hypot(x - 0.2 * s, y + 0.25 * s) > 3.5

        area = (s / 40.0) ** 2
        for x, y in self.spots(int(72 * area), 2.3, taken, grove):
            self.round_tree(b, x, y)
        for x, y in self.spots(int(34 * area), 2.6, taken, grove):
            self.pine(b, x, y)
        for x, y in self.spots(int(9 * area), 3.0, taken, clear_of_path):
            self.rock(b, x, y, big=True)
        for x, y in self.spots(int(36 * area), 1.1, taken, clear_of_path):
            self.rock(b, x, y, big=False)
        for x, y in self.spots(int(26 * area), 1.2, taken, clear_of_path):
            self.bush(b, x, y)
        for x, y in self.spots(int(10 * area), 2.2, taken, clear_of_path):
            self.log(b, x, y)
        return b.mesh()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out')
    ap.add_argument('--size', type=float, default=40.0, help='island side, metres')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    mesh = Island(args.size, args.seed).build()
    # glTF is Y-up: z-up (x, y, z) -> (x, z, -y), undone by terrain.prepare(up='y').
    mesh.apply_transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    mesh.export(args.out)
    print(f'{args.out}: {len(mesh.faces)} triangles, {args.size:.0f} m island')


if __name__ == '__main__':
    main()
