"""Two simulated drones and two simulated rovers, declared in one JSON file, feeding ONE map.

    python devices/examples/run_fleet_sim.py            # survey, look, plan for each rover
    python devices/examples/run_fleet_sim.py --drive    # then drive each rover, one at a time

No hardware, numpy only (merging_map's simulation).
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polaris_devices import build  # noqa: E402

CONFIG = Path(__file__).with_name('fleet_sim.json')
VICTIM = (8.0, 20.0)


def length(path):
    return sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--victim', type=float, nargs=2, default=VICTIM, metavar=('X', 'Y'))
    p.add_argument('--drive', action='store_true', help='run the mission of each rover')
    args = p.parse_args(argv)

    fleet = build(args.config)
    for device_id, n in fleet.survey().items():
        print(f'{device_id}: {n} tiles published')
    for rover_id, rover in fleet.mobiles().items():
        rover.observer(tour_complet=True)  # what each rover sees around its start
        print(f'{rover_id}: located at {fleet.pose_of(rover_id)}')
    results = {}
    for rover_id in fleet.mobiles():
        dec = fleet.plan(rover_id, args.victim)
        results[rover_id] = dec
        print(f'{rover_id}: {dec["action"]}, path {length(dec.get("chemin") or []):.1f} m')
    if args.drive:
        for rover_id in fleet.mobiles():  # one rover per mission: no traffic management
            print(f'--- mission of {rover_id}')
            fin = fleet.run_mission(rover_id, args.victim, journal=lambda s: None)['fin']
            end = fin['traj'][-1]
            print(f'{rover_id}: {length(fin["traj"]):.1f} m driven, ends {end}, '
                  f'{math.dist(end, args.victim):.2f} m from the victim')  # fmt: skip
    return fleet, results


if __name__ == '__main__':
    main()
