"""The command log: what the rover received, and how fast it acted on it."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rover_server  # noqa: E402
from rover_core import CommandLog, Config  # noqa: E402

MISSION = '192.168.1.5 python-urllib'


def test_step_latencies_measured_on_the_robot() -> None:
    log = CommandLog()
    log.add(10.0, MISSION, '/path', {'points': [[0.2, 0]], 'frame': 'odom', 'speed': 0.2}, 200)
    log.on_tick(10.04, 0.0, 0.0, 0.0, 0.0, 'path', None)  # not yet out
    log.on_tick(10.05, 0.2, 0.0, 0.0, 0.0, 'path', None)  # sent to the ESP32
    log.on_tick(10.15, 0.2, 0.0, 0.01, 0.0, 'path', None)  # wheels barely
    log.on_tick(10.25, 0.2, 0.0, 0.06, 0.06, 'path', None)  # wheels turn
    log.on_tick(11.40, 0.0, 0.0, 0.0, 0.0, 'idle', 'done')
    e = log.snapshot(12.0)['entries'][0]
    assert e['text'] == 'path 1 pt, odom, 0.20 m/s'
    assert (e['out_ms'], e['wheels_ms'], e['done_ms']) == (50, 250, 1400)
    assert e['status'] == 'done' and e['client'] == MISSION and e['ago_s'] == 2.0


def test_held_key_is_one_line_with_its_largest_gap() -> None:
    log = CommandLog()
    page = CommandLog.client_name('192.168.1.5', 'Mozilla/5.0 (Windows)')
    for t in (0.0, 0.1, 0.2, 0.3, 1.0, 1.1):  # 0.7 s without a repeat: the rover stopped
        log.add(t, page, '/cmd', {'v': 0.15, 'omega': 0.0}, 200)
    log.add(1.2, page, '/cmd', {'v': 0.0, 'omega': 0.0}, 200)  # key released
    entries = log.snapshot(2.0)['entries']
    assert [e['text'] for e in entries] == ['manual v +0.00 m/s, w +0.00 rad/s',
                                            'manual v +0.15 m/s, w +0.00 rad/s']  # fmt: skip
    held = entries[1]
    assert held['count'] == 6 and held['max_gap_ms'] == 700 and held['held_s'] == 1.1
    assert entries[0]['expect'] == 'stop'


def test_refused_stop_and_wheels_that_do_not_follow() -> None:
    log = CommandLog()
    log.add(0.0, MISSION, '/move', {'distance': 0.1}, 409)  # before "Gyro OK"
    log.add(1.0, MISSION, '/stop', {}, 200)
    log.on_tick(1.05, 0.0, 0.0, 0.08, 0.08, 'idle', None)
    log.on_tick(1.30, 0.0, 0.0, 0.0, 0.005, 'idle', None)
    log.add(2.0, MISSION, '/turn', {'angle': 90}, 200)
    for k in range(1, 130):  # commanded, but the wheels never turn (lifted? unpowered?)
        log.on_tick(2.0 + k * 0.05, 0.0, 1.0, 0.0, 0.0, 'turn', None)
    turn, stop, move = log.snapshot(9.0)['entries']
    assert move['status'] == 'refused' and move['out_ms'] is None
    assert stop['status'] == 'done' and (stop['out_ms'], stop['wheels_ms']) == (50, 300)
    assert turn['status'] == 'wheels did not follow' and turn['out_ms'] == 50


def test_a_newer_order_closes_the_measure_of_the_older_one() -> None:
    log = CommandLog()
    log.add(0.0, MISSION, '/stop', {}, 200)
    log.add(0.02, MISSION, '/cmd', {'v': 0.15, 'omega': 0.0}, 200)  # right after the stop
    log.on_tick(0.05, 0.15, 0.0, 0.0, 0.0, 'manual', None)
    log.on_tick(0.20, 0.15, 0.0, 0.08, 0.08, 'manual', None)
    manual, stop = log.snapshot(1.0)['entries']
    assert stop['status'] == 'superseded' and stop['out_ms'] is None
    assert manual['status'] == 'done' and (manual['out_ms'], manual['wheels_ms']) == (30, 180)


def test_clients_rate_and_gaps() -> None:
    log = CommandLog()
    for k in range(50):  # the mission polls /state 5 times a second, with one 1.2 s hole
        log.note_request(k * 0.2 + (1.0 if k >= 25 else 0.0), MISSION)
    c = log.snapshot(11.0)['clients'][0]
    assert c['client'] == MISSION and c['max_gap_ms'] == 1200 and c['median_gap_ms'] == 200
    assert CommandLog.client_name('10.0.0.2', 'python-requests/2.31') == '10.0.0.2 python-requests'
    assert CommandLog.client_name('10.0.0.2', None, 'mission') == '10.0.0.2 mission'


def test_server_logs_posts_and_serves_commands(tmp_path) -> None:
    args = SimpleNamespace(sim_drift=0.0, width=64, height=48, fps=5.0, quality=50, camera='0',
                           no_camera=True, d405='off')  # fmt: skip
    app = rover_server.App(Config(), str(tmp_path / 'cfg.json'), args, serial_port=None)
    app.set_mode('sim')
    srv = ThreadingHTTPServer(('127.0.0.1', 0), rover_server.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_address[1]}'

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(), method='POST')
        req.add_header('X-Rover-Client', 'test-mission')
        try:
            return urllib.request.urlopen(req, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:  # steps are refused before "Gyro OK"
            s = json.loads(urllib.request.urlopen(base + '/state', timeout=5).read())
            if s['odom']['gyro_ready']:
                break
            time.sleep(0.1)
        assert post('/move', {'distance': 0.05}) == 200
        assert post('/move', {'distance': 'far'}) == 400
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            snap = json.loads(urllib.request.urlopen(base + '/commands', timeout=5).read())
            move = [e for e in snap['entries'] if e['code'] == 200 and e['route'] == '/move']
            if move and move[0]['status'] != 'pending':
                break
            time.sleep(0.1)
        assert move[0]['status'] == 'done' and move[0]['client'].endswith('test-mission')
        assert 0 <= move[0]['out_ms'] <= 120  # the 20 Hz loop: within one or two ticks
        assert move[0]['wheels_ms'] >= move[0]['out_ms'] and move[0]['done_ms'] is not None
        bad = [e for e in snap['entries'] if e['code'] == 400]
        assert bad and bad[0]['status'] == 'refused'
        names = [c['client'] for c in snap['clients']]
        assert any(n.endswith('test-mission') for n in names)
    finally:
        srv.shutdown()
        srv.server_close()
        app.close()


if __name__ == '__main__':
    pytest.main([__file__])
