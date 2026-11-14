"""Measure the gyro scale of the ESP32 board: turn the rover by hand, full turns.

    python3 calibrate_gyro.py [--port /dev/ttyACM0]

Opening the port reboots the ESP32 (about 7 s, IMU calibration): keep the rover still.
Then the bias is measured (3 s, still), and for each try you turn the rover by hand exactly
N full turns to the LEFT (counter-clockwise seen from above), back to a reference mark, and
press Enter. The script prints the raw gz integral and the `gyro_scale` (raw -> rad/s) for
rover_config.json. `yaw_sign` is -1 if the integral of a left turn comes out negative.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time

from rover_core import find_esp32_port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None)
    ap.add_argument('--turns', type=float, default=1.0, help='full turns per try (default 1)')
    args = ap.parse_args()

    import serial

    port = args.port or find_esp32_port()
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 115200, 0.2
    s.dtr = s.rts = False
    s.open()
    print(f'{port} open. ESP32 rebooting: KEEP THE ROVER STILL...', flush=True)
    t0 = time.time()
    while time.time() - t0 < 20:
        if 'UGV started' in s.readline().decode(errors='replace'):
            break
    time.sleep(1.0)
    s.reset_input_buffer()

    def samples():
        while True:
            line = s.readline().decode(errors='replace').strip()
            if line.startswith('{"T":1001'):
                try:
                    yield time.time(), json.loads(line)
                except ValueError:
                    pass

    feed = samples()
    print('Measuring the bias, 3 s, keep still...', flush=True)
    vals = []
    t0 = time.time()
    for t, m in feed:
        vals.append(m['gz'])
        if t - t0 > 3.0:
            break
    bias = sum(vals) / len(vals)
    noise = math.sqrt(sum((v - bias) ** 2 for v in vals) / len(vals))
    print(f'bias = {bias:.2f} raw, noise = {noise:.1f} raw ({len(vals)} samples)', flush=True)

    enter = threading.Event()

    def wait_enter():
        while True:
            sys.stdin.readline()
            enter.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    target = 2 * math.pi * args.turns
    results = []
    while True:
        print(
            f'\n>>> Turn the rover {args.turns:g} full turn(s) to the LEFT '
            '(counter-clockwise from above), back to the mark, then press Enter. '
            '(Ctrl+C to finish)',
            flush=True,
        )
        enter.clear()
        integral = 0.0
        last = None
        peak = 0.0
        for t, m in feed:
            if last is not None:
                integral += (m['gz'] - bias) * (t - last)
            last = t
            peak = max(peak, abs(m['gz'] - bias))
            if enter.is_set():
                break
        scale = target / integral if integral else float('nan')
        results.append(scale)
        print(
            f'integral = {integral:.1f} raw*s, peak rate = {peak:.0f} raw -> '
            f'gyro_scale = {abs(scale):.6f} rad/s per raw unit, '
            f'yaw_sign = {1 if integral > 0 else -1}',
            flush=True,
        )
        if abs(scale) > 0:
            print(f'   i.e. {1 / math.degrees(abs(scale)):.2f} raw per deg/s', flush=True)
        good = [abs(r) for r in results if r == r]
        if len(good) > 1:
            mean = sum(good) / len(good)
            spread = (max(good) - min(good)) / mean * 100
            print(f'   mean of {len(good)} tries: {mean:.6f} (spread {spread:.1f} %)', flush=True)
        if peak * abs(scale) > math.radians(300):
            print('   warning: turned fast, the gyro may have saturated: turn slower', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nDone.')
