"""A record of each drive, to compare one test with the next (and with the team's chain).

guide_rover.py writes one JSON line per event and four per second while a goal is set, in
air_ground/runs/<date>-<time>.jsonl (not ignored by git: commit a run worth keeping).

    python air_ground/run_log.py air_ground/runs/20261116-101500.jsonl

prints, for each goal: how long it took, the distance driven, the paths sent to the rover,
the pauses (the rover standing still for 0.5 s or more before arriving), the time spent
without seeing the rover from the air, and how far from the goal it ended.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time


class RunLog:
    def __init__(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, time.strftime('%Y%m%d-%H%M%S') + '.jsonl')
        self._lock = threading.Lock()
        self._t0: float | None = None

    def write(self, kind: str, at: float | None = None, **fields) -> None:
        """One line; `at`: the time of the event (time.monotonic() by default), written in
        seconds from the first line."""
        at = time.monotonic() if at is None else at
        if self._t0 is None:
            self._t0 = at
        line = {'t': round(at - self._t0, 3), 'kind': kind, **fields}
        with self._lock, open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line, separators=(',', ':')) + '\n')


def summarise(lines: list[dict]) -> list[dict]:
    """One summary per goal of a log."""
    out, cur = [], None
    for e in lines:
        if e['kind'] == 'goal':
            cur = {'goal': e['goal'], 't0': e['t'], 'ticks': [], 'sends': 0, 'routes': 0}
            out.append(cur)
        elif cur is None:
            continue
        elif e['kind'] == 'tick':
            cur['ticks'].append(e)
        elif e['kind'] == 'send':
            cur['sends'] += 1
        elif e['kind'] == 'route':
            cur['routes'] += 1
        elif e['kind'] == 'end':
            cur['end'] = e
            cur = None
    result = []
    for g in out:
        ticks = [t for t in g['ticks'] if t.get('rover')]
        steps = zip(ticks, ticks[1:], strict=False)
        dist = sum(math.dist(a['rover'][:2], b['rover'][:2]) for a, b in steps)
        pauses, still_since = 0, None
        for a, b in zip(ticks, ticks[1:], strict=False):
            speed = math.dist(a['rover'][:2], b['rover'][:2]) / max(b['t'] - a['t'], 1e-3)
            if speed < 0.02 and b.get('mode') in ('idle', 'settle', 'turn'):
                still_since = a['t'] if still_since is None else still_since
            else:
                if still_since is not None and a['t'] - still_since >= 0.5:
                    pauses += 1
                still_since = None
        end = g.get('end', {})
        last = ticks[-1]['rover'][:2] if ticks else None
        t_end = end.get('t') or (ticks[-1]['t'] if ticks else g['t0'])
        result.append({
            'goal': g['goal'],
            'status': end.get('status', 'unfinished'),
            'duration_s': round(t_end - g['t0'], 1),
            'driven_m': round(dist, 2),
            'paths_sent': g['sends'],
            'routes': g['routes'],
            'pauses': pauses,
            'blind_s': round(sum(0.25 for t in ticks if (t.get('seen_age') or 0) > 1.0), 1),
            'end_error_m': None if last is None else round(math.dist(last, g['goal']), 3),
        })  # fmt: skip
    return result


def main() -> None:
    for path in sys.argv[1:]:
        with open(path, encoding='utf-8') as f:
            lines = [json.loads(x) for x in f if x.strip()]
        print(path)
        for s in summarise(lines):
            print('  ' + ', '.join(f'{k} {v}' for k, v in s.items()))


if __name__ == '__main__':
    main()
