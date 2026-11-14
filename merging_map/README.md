# merging_map: drone + rover map fusion

Merges the obstacle tiles sent by the drone and the rover into one map, plans a path with A*
and decides what to do when the rover is blocked.

| File | What it is |
|---|---|
| `fusion_carte.py` | The module: tile storage, fusion, A*, blocking decisions |
| `demo_fusion.py` | Demo on a simulated world (needs `numpy` and `matplotlib`) |
| `demo_fusion.png` | Output of the demo |
| `test_fusion_carte.py` | pytest tests (fusion rules, tile cutting, planning, decisions) |
| `conftest.py` | Lets pytest import `fusion_carte` from this folder |

```bash
cd merging_map
python demo_fusion.py
```

![demo](demo_fusion.png)

## Instructions for the rover team: tile format

* **N×N** matrix (N = 32), finest cell = **5 cm** (level z = 4).
* Values:
   * `1` = obstacle
   * `0` = free, **actually seen**
   * `-1` = not seen
* ⚠️ **"not seen" ≠ "free"**. Use `-1` for cells:
   * **behind any detected obstacle** (the camera can't see through it);
   * **beyond the camera's reliable range** (about 2 m);
   * anywhere you didn't look.

  If a hidden cell is sent as `0`, it **erases obstacles detected by the drone**.
* Tiles must be **aligned to the global grid** (drone / ArUco frame), not to the robot's frame.
* **Row 0 = south** (not the top, as in an image). Row i → +Y, column j → +X.
* Message:

  ```json
  {"source": "rover", "t": <seconds>, "z": 4, "x": <tx>, "y": <ty>, "data": [[...]]}
  ```

  Tile `(z, x, y)` covers `x*T <= X < (x+1)*T` and `y*T <= Y < (y+1)*T`, with
  `T = 32 × cell size` (1.6 m at z = 4).
* To split a large matrix into tiles, use `decouper()` in `fusion_carte.py`:

  ```python
  from fusion_carte import CarteFusion, decouper

  carte = CarteFusion()   # N = 32, finest cell 5 cm, zmax = 4
  # local: numpy matrix of -1/0/1 at 5 cm, whose cell [0][0] is at (x0, y0) in the global frame
  for msg in decouper(local, x0, y0, 4, carte, 'rover', t=now):
      send(msg)           # or carte.recevoir(msg) when testing locally
  ```
* For the pan-tilt camera, first send `{"T":900,"main":1,"module":2}` to the ESP32. The OLED
  should then show "RaspRover PT".

## How the fusion works

* The drone and rover tiles are stored separately: nothing the drone sends is ever deleted.
* In the merged view, a cell **seen by the rover** (`0` or `1`) wins over the drone; elsewhere the
  drone value stands. `-1` never changes anything.
* Within one source, the finest level and the most recent tile win.

## Using the module

```python
from fusion_carte import CarteFusion

carte = CarteFusion(n=32, res_fine=0.05, zmax=4)
carte.recevoir(msg)            # every tile received from the drone or the rover
carte.signaler_obstacle(x, y)  # wheel slip / stuck / slope too steep, felt by the rover

dec = carte.decider(rover_position, victim_position)
# dec['action'] is one of:
#   'SUIVRE'           -> dec['chemin']: list of (x, y) in metres
#   'DEMANDER_DRONE'   -> dec['tuiles']: [(z, x, y), ...] for the drone to fly over lower
#   'OBJECTIF_DEGRADE' -> dec['point']: closest reachable point to the victim, plus dec['chemin']
```

When no path exists, the module asks: "if the obstacles the drone only saw from far away were
actually passable, where would I go?" It then asks the drone to zoom on exactly those tiles. If
the drone confirms they are blocked, the rover goes to the closest reachable point and raises
the alert.

## Tests and lint

```bash
pytest merging_map                     # needs numpy (also run by a plain `pytest` from the root)
ruff check merging_map && ruff format --check merging_map
```

Code comments and identifiers are in French.
