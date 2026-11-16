"""From a 3D terrain model (GLB / OBJ) to what the mission simulation needs.

    python sim/mission/terrain.py MODEL.glb OUT_DIR [--size 40] [--up y] [--res 0.05]

Writes into OUT_DIR:
  mesh_*.obj, tex_*.png   the model, re-centred, scaled and z-up, as MuJoCo reads it
  terrain.json            what was done (scale, extent, files)
  maps.npz                maps derived from the mesh by vertical ray casting (see `derive`)
  maps.png                a picture of them, with the safe start points

Nothing but the mesh is used: the same code handles the stand-in island (`terrain_gen.py`) and
a model made elsewhere. `--size` is the side (m) the model's larger horizontal extent is scaled
to: it decides how big the rover is compared with the trees, so look at maps.png.

Maps (cells of `res` m, array[i, j] is the cell at x = x0 + j*res, y = y0 + i*res: row 0 is
south, the convention of merging_map):
  top      height of the first surface seen from above (what a lidar on a drone returns)
  ground   height of the ground the rover drives on (also under crowns), NaN outside
  blocked  True where the rover cannot be: something between 7 and 50 cm above the ground,
           inside a solid, too steep, no ground, or near the edge of the model
  canopy   True where a crown hides drivable ground from above
  safe     True where a robot or the victim may be placed: free, with a margin, and connected
           to the largest free area
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

ROVER_RADIUS = 0.16  # m, circle around the RaspRover's body
BAND = (0.07, 0.50)  # m above the ground: what the rover would hit
MAX_SLOPE_DEG = 24.0
SAFE_MARGIN = 0.45  # m of free ground around a start point, on top of the rover's radius
EDGE_MARGIN = 0.9  # m kept free along the edge of the model


# ----------------------------------------------------------------------------------------------
# Model -> MuJoCo meshes
# ----------------------------------------------------------------------------------------------
def _texture_of(mesh):
    """(PIL image or None, rgba) of a trimesh geometry."""
    vis = mesh.visual
    mat = getattr(vis, 'material', None)
    img = None
    if mat is not None:
        img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
    rgba = (0.6, 0.6, 0.6, 1.0)
    try:
        c = np.asarray(vis.to_color().main_color if img is None else (255, 255, 255, 255), float)
        rgba = tuple(float(v) / 255.0 for v in c[:4])
    except Exception:  # noqa: BLE001 - colour is cosmetic
        pass
    has_uv = getattr(vis, 'uv', None) is not None and len(vis.uv) == len(mesh.vertices)
    return (img if has_uv else None), rgba


def _write_obj(path: str, v: np.ndarray, f: np.ndarray, uv) -> None:
    with open(path, 'w') as out:
        np.savetxt(out, v, fmt='v %.5f %.5f %.5f')
        if uv is not None:
            np.savetxt(out, uv, fmt='vt %.6f %.6f')
            a = f + 1
            np.savetxt(out, np.stack([a[:, 0]] * 2 + [a[:, 1]] * 2 + [a[:, 2]] * 2, 1),
                       fmt='f %d/%d %d/%d %d/%d')  # fmt: skip
        else:
            np.savetxt(out, f + 1, fmt='f %d %d %d')


def convert(
    model_path: str, out_dir: str, size: float, up: str = 'y', max_tex: int = 4096
) -> dict:
    """Re-centre, scale and write the model as OBJ + PNG; returns the manifest."""
    import trimesh

    os.makedirs(out_dir, exist_ok=True)
    loaded = trimesh.load(model_path, process=False)
    meshes = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise ValueError(f'no triangle mesh in {model_path}')
    rot = np.eye(3) if up == 'z' else np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    verts = [np.asarray(m.vertices, float) @ rot.T for m in meshes]
    lo = np.min([v.min(0) for v in verts], 0)
    hi = np.max([v.max(0) for v in verts], 0)
    scale = size / max(hi[0] - lo[0], hi[1] - lo[1])
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    parts = []
    for k, (m, v) in enumerate(zip(meshes, verts)):
        v = (v - centre) * scale
        img, rgba = _texture_of(m)
        part = {'obj': f'mesh_{k}.obj', 'faces': int(len(m.faces)), 'rgba': rgba, 'texture': None}
        _write_obj(os.path.join(out_dir, part['obj']), v, np.asarray(m.faces),
                   np.asarray(m.visual.uv, float) if img is not None else None)  # fmt: skip
        if img is not None:
            img = img.convert('RGB')
            if max(img.size) > max_tex:
                img.thumbnail((max_tex, max_tex))
            part['texture'] = f'tex_{k}.png'
            img.save(os.path.join(out_dir, part['texture']))
        parts.append(part)
    ext = (hi - lo) * scale
    man = {
        'source': os.path.basename(model_path), 'size': size, 'scale': float(scale), 'up': up,
        'extent': [float(e) for e in ext], 'parts': parts,
        'x_range': [-float(ext[0]) / 2, float(ext[0]) / 2],
        'y_range': [-float(ext[1]) / 2, float(ext[1]) / 2],
    }  # fmt: skip
    json.dump(man, open(os.path.join(out_dir, 'terrain.json'), 'w'), indent=1)
    return man


def mjcf_assets(terrain_dir: str, man: dict) -> tuple[str, str]:
    """(<asset> children, <worldbody> geoms) of the terrain: visual only, never collides."""
    assets, geoms = [], []
    for k, p in enumerate(man['parts']):
        obj = os.path.join(terrain_dir, p['obj'])
        assets.append(f'<mesh name="terrain{k}" file="{obj}" inertia="shell"/>')
        if p['texture']:
            tex = os.path.join(terrain_dir, p['texture'])
            assets.append(f'<texture name="terrain{k}" type="2d" file="{tex}"/>')
            assets.append(f'<material name="terrain{k}" texture="terrain{k}" specular="0.08" '
                          'shininess="0.1" reflectance="0"/>')  # fmt: skip
        else:
            r, g, b, a = p['rgba']
            assets.append(f'<material name="terrain{k}" rgba="{r} {g} {b} {a}" specular="0.08"/>')
        geoms.append(f'<geom name="terrain{k}" type="mesh" mesh="terrain{k}" '
                     f'material="terrain{k}" contype="0" conaffinity="0" group="0"/>')  # fmt: skip
    return '\n'.join(assets), '\n'.join(geoms)


# ----------------------------------------------------------------------------------------------
# Mesh -> maps
# ----------------------------------------------------------------------------------------------
def cast_columns(terrain_dir: str, man: dict, res: float, max_hits: int = 8, progress=None):
    """Every surface crossed by a vertical line through each cell centre, from above.

    Returns (hits, x0, y0): hits[i, j, k] = height of the k-th surface met going down, NaN after
    the last one. MuJoCo's ray casting against the triangles of the mesh (BVH).
    """
    import mujoco

    assets, geoms = mjcf_assets(terrain_dir, man)
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset>{assets}</asset><worldbody>{geoms}</worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nx = int(math.ceil(man['extent'][0] / res))
    ny = int(math.ceil(man['extent'][1] / res))
    x0 = -nx * res / 2
    y0 = -ny * res / 2
    z_top = man['extent'][2] + 1.0
    hits = np.full((ny, nx, max_hits), np.nan, np.float32)
    down = np.array([0.0, 0.0, -1.0])
    geomid = np.zeros(1, np.int32)
    pnt = np.zeros(3)
    for i in range(ny):
        pnt[1] = y0 + (i + 0.5) * res
        for j in range(nx):
            pnt[0] = x0 + (j + 0.5) * res
            z = z_top
            for k in range(max_hits):
                pnt[2] = z
                d = mujoco.mj_ray(model, data, pnt, down, None, 1, -1, geomid)
                if d < 0:
                    break
                z -= d
                hits[i, j, k] = z
                z -= 1e-4
        if progress and i % 50 == 0:
            progress(i / ny)
    return hits, x0, y0


def _grey(arr: np.ndarray, radius_cells: int, op: str) -> np.ndarray:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_cells + 1,) * 2)
    return cv2.morphologyEx(arr, {'close': cv2.MORPH_CLOSE, 'open': cv2.MORPH_OPEN}[op], k)


def derive(hits: np.ndarray, res: float) -> dict:
    """The maps of the module's header, from the surfaces crossed by each vertical line."""
    import cv2

    n = np.isfinite(hits).sum(-1)
    inside = n > 0
    top = hits[..., 0].copy()
    # A closed model (a diorama with an underside) ends every line on its underside: the ground
    # is then the surface before the last. An open sheet ends on the ground itself.
    closed = bool((n[inside] % 2 == 0).mean() > 0.6)
    idx = np.clip(n - (2 if closed else 1), 0, hits.shape[-1] - 1)
    low = np.take_along_axis(hits, idx[..., None], -1)[..., 0]
    low[n < (2 if closed else 1)] = np.nan
    fill = np.nanmin(low) - 5.0
    lowf = np.where(np.isfinite(low), low, fill).astype(np.float32)
    # Ground: pits under sunk rocks and trunks closed, then peaks (lines that never met ground
    # because a trunk is fused with it) opened away.
    ground = _grey(_grey(lowf, int(round(1.5 / res)), 'close'), int(round(0.6 / res)), 'open')
    has_ground = inside & (ground > fill + 1.0)
    ground = np.where(has_ground, ground, np.nan)
    g = np.nan_to_num(ground, nan=fill)

    above = hits - g[..., None]
    in_band = ((above > BAND[0]) & (above < BAND[1])).any(-1)
    mid = (above > (BAND[0] + BAND[1]) / 2).sum(-1)  # surfaces above the middle of the band
    solid = mid % 2 == 1  # an odd number: that point is inside a closed shape
    no_floor = np.abs(np.nan_to_num(low, nan=fill) - g) > BAND[0]
    smooth = cv2.GaussianBlur(g, (0, 0), 0.10 / res)
    gy, gx = np.gradient(smooth, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    edge = cv2.dilate((~has_ground).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * int(round(EDGE_MARGIN / res)) + 1,) * 2)) > 0  # fmt: skip
    blocked = ~has_ground | in_band | solid | no_floor | (slope > MAX_SLOPE_DEG) | edge
    canopy = has_ground & ~blocked & (top - g > 1.0)

    r = int(round((ROVER_RADIUS + SAFE_MARGIN) / res))
    roomy = cv2.erode((~blocked).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))  # fmt: skip
    # Reachability is decided with the rover's own radius; the margin only picks the points.
    rr = int(round(ROVER_RADIUS / res))
    drivable = cv2.erode((~blocked).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1,) * 2))  # fmt: skip
    count, labels, stats, _ = cv2.connectedComponentsWithStats(drivable, connectivity=8)
    if count < 2:
        raise ValueError('no drivable ground found: check --size and --up')
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    safe = (roomy > 0) & (labels == main)
    return {
        'top': np.where(inside, top, np.nan).astype(np.float32), 'ground': ground.astype(np.float32),
        'blocked': blocked, 'canopy': canopy, 'safe': safe, 'drivable': labels == main,
        'slope': slope.astype(np.float32), 'closed': closed,
    }  # fmt: skip


def picture(maps: dict, points=None, scale: int = 1) -> np.ndarray:
    """BGR picture of the maps (north up)."""
    import cv2

    g = maps['ground']
    lo, hi = np.nanmin(g), np.nanmax(g)
    shade = np.nan_to_num((g - lo) / max(hi - lo, 1e-6), nan=0.0)
    img = np.zeros((*g.shape, 3), np.uint8)
    img[...] = (30, 26, 22)
    free = ~maps['blocked']
    img[free] = (np.stack([70 + 60 * shade, 120 + 80 * shade, 70 + 50 * shade], -1)[free]).astype(
        np.uint8
    )
    img[maps['blocked'] & np.isfinite(g)] = (70, 80, 150)
    img[maps['canopy']] = (img[maps['canopy']] * 0.55 + np.array([20, 110, 150]) * 0.45).astype(
        np.uint8
    )
    safe_edge = maps['safe'] & ~cv2.erode(
        maps['safe'].astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    img[safe_edge] = (255, 255, 255)
    img = img[::-1].copy()
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for (i, j), colour in points or []:
        cv2.circle(
            img, (int(j * scale), int((g.shape[0] - 1 - i) * scale)), 6, colour, -1, cv2.LINE_AA
        )
    return img


def prepare(model_path: str, out_dir: str, size: float = 40.0, up: str = 'y', res: float = 0.05,
            progress=None) -> dict:  # fmt: skip
    import cv2

    man = convert(model_path, out_dir, size, up)
    hits, x0, y0 = cast_columns(out_dir, man, res, progress=progress)
    maps = derive(hits, res)
    np.savez_compressed(os.path.join(out_dir, 'maps.npz'), res=res, x0=x0, y0=y0,
                        **{k: v for k, v in maps.items() if k != 'closed'})  # fmt: skip
    man.update(res=res, x0=x0, y0=y0, closed=maps['closed'],
               free_m2=float((~maps['blocked']).sum() * res * res),
               safe_m2=float(maps['safe'].sum() * res * res),
               canopy_m2=float(maps['canopy'].sum() * res * res))  # fmt: skip
    json.dump(man, open(os.path.join(out_dir, 'terrain.json'), 'w'), indent=1)
    cv2.imwrite(os.path.join(out_dir, 'maps.png'), picture(maps))
    return man


class Terrain:
    """The prepared terrain: maps and world <-> cell helpers."""

    def __init__(self, terrain_dir: str) -> None:
        self.dir = terrain_dir
        self.man = json.load(open(os.path.join(terrain_dir, 'terrain.json')))
        z = np.load(os.path.join(terrain_dir, 'maps.npz'))
        self.res, self.x0, self.y0 = float(z['res']), float(z['x0']), float(z['y0'])
        self.top, self.ground = z['top'], z['ground']
        self.blocked, self.canopy = z['blocked'], z['canopy']
        self.safe, self.drivable = z['safe'], z['drivable']
        self.shape = self.ground.shape
        self.floor = float(np.nanmin(self.ground))
        self.top_filled = np.nan_to_num(self.top, nan=self.floor - 20.0)
        self.ground_filled = np.nan_to_num(self.ground, nan=self.floor)
        self.x1 = self.x0 + self.shape[1] * self.res
        self.y1 = self.y0 + self.shape[0] * self.res

    def cell(self, x, y):
        j = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.shape[1] - 1)
        i = np.clip(((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.shape[0] - 1)
        return i, j

    def xy(self, i, j):
        return self.x0 + (np.asarray(j) + 0.5) * self.res, self.y0 + (
            np.asarray(i) + 0.5
        ) * self.res

    def ground_at(self, x, y):
        return self.ground_filled[self.cell(x, y)]

    def top_at(self, x, y):
        return self.top_filled[self.cell(x, y)]

    def is_blocked(self, x, y):
        inside = (self.x0 <= np.asarray(x)) & (np.asarray(x) < self.x1)
        inside &= (self.y0 <= np.asarray(y)) & (np.asarray(y) < self.y1)
        return ~inside | self.blocked[self.cell(x, y)]

    def normal_at(self, x: float, y: float, d: float = 0.12) -> np.ndarray:
        gx = (self.ground_at(x + d, y) - self.ground_at(x - d, y)) / (2 * d)
        gy = (self.ground_at(x, y + d) - self.ground_at(x, y - d)) / (2 * d)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    def path_distances(self, x: float, y: float, step: int = 4) -> np.ndarray:
        """Driving distance (m) from (x, y) to every cell of a grid `step` times coarser,
        through drivable cells only (inf elsewhere). Breadth-first with 8 neighbours."""
        import heapq

        free = self.drivable[::step, ::step]
        dist = np.full(free.shape, np.inf)
        i, j = self.cell(x, y)
        start = (int(i) // step, int(j) // step)
        if not free[start]:
            return dist
        dist[start] = 0.0
        heap = [(0.0, start)]
        r = self.res * step
        while heap:
            d, (a, b) = heapq.heappop(heap)
            if d > dist[a, b]:
                continue
            for da, db, w in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                              (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)):  # fmt: skip
                u, v = a + da, b + db
                if 0 <= u < free.shape[0] and 0 <= v < free.shape[1] and free[u, v]:
                    nd = d + w * r
                    if nd < dist[u, v]:
                        dist[u, v] = nd
                        heapq.heappush(heap, (nd, (u, v)))
        return dist

    def safe_points(self, rng, n: int, min_gap: float, avoid=(), avoid_gap: float = 0.0,
                    where=None) -> list[tuple[float, float]]:  # fmt: skip
        """`n` safe points at least `min_gap` m apart (and `avoid_gap` m from `avoid`)."""
        mask = self.safe if where is None else (self.safe & where)
        ii, jj = np.nonzero(mask)
        if not len(ii):
            return []
        order = rng.permutation(len(ii))
        out: list[tuple[float, float]] = []
        for k in order:
            x, y = (float(v) for v in self.xy(ii[k], jj[k]))
            if any(math.hypot(x - a, y - b) < min_gap for a, b in out):
                continue
            if any(math.hypot(x - a, y - b) < avoid_gap for a, b in avoid):
                continue
            out.append((x, y))
            if len(out) == n:
                break
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('model')
    ap.add_argument('out_dir')
    ap.add_argument('--size', type=float, default=40.0)
    ap.add_argument('--up', choices=('y', 'z'), default='y', help='up axis of the model (glTF: y)')
    ap.add_argument('--res', type=float, default=0.05)
    args = ap.parse_args()
    man = prepare(args.model, args.out_dir, args.size, args.up, args.res,
                  progress=lambda f: print(f'  ray casting {100 * f:.0f} %', flush=True))  # fmt: skip
    print(json.dumps({k: man[k] for k in ('source', 'scale', 'extent', 'closed', 'free_m2',
                                          'safe_m2', 'canopy_m2')}, indent=1))  # fmt: skip


if __name__ == '__main__':
    main()
