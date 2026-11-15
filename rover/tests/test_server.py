"""HTTP API of rover_server.py, in process, against the simulated ESP32 (127.0.0.1:8120-8121)."""

from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROVER_DIR = os.path.dirname(HERE)
sys.path.insert(0, ROVER_DIR)

import rover_server  # noqa: E402
from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from rover_core import Config, Rover  # noqa: E402
from rover_server import App, Camera, make_handler  # noqa: E402

PORT = 8120
SIGTERM_PORT = 8121


def strict_json(data: bytes):
    def refuse(name):
        raise ValueError(f'{name} is not JSON')

    return json.loads(data, parse_constant=refuse)


class Bench:
    def __init__(self, tmp) -> None:
        self.board = FakeEsp32()
        self.rover = Rover(
            Config(bias_calib_time=0.5), lambda: FakeSerial(self.board, boot_time=0.3)
        )
        self.camera = Camera('test', width=64, height=48, fps=15, quality=70)
        self.camera.request = lambda: None  # frames are published by the tests
        self.cfg_path = os.path.join(tmp, 'rover_config.json')
        self.app = App(self.rover, self.camera, self.cfg_path, None)
        self.server = rover_server.ThreadingHTTPServer(('127.0.0.1', PORT), make_handler(self.app))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.rover.start()
        assert wait_for(lambda: self.rover.odom.bias_ready, 8.0)

    def request(self, method: str, path: str, body: bytes | str | None = None):
        conn = http.client.HTTPConnection('127.0.0.1', PORT, timeout=10)
        if isinstance(body, str):
            body = body.encode()
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, dict(r.getheaders()), data

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.rover.close()


def wait_for(cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@pytest.fixture(scope='module')
def bench(tmp_path_factory):
    b = Bench(str(tmp_path_factory.mktemp('rover')))
    yield b
    b.close()


@pytest.fixture(autouse=True)
def stopped(request):
    yield
    if 'bench' in request.fixturenames:
        b = request.getfixturevalue('bench')
        b.rover.stop()
        time.sleep(0.1)


def test_state_has_identity_and_strict_json(bench: Bench) -> None:
    code, _, data = bench.request('GET', '/state')
    assert code == 200
    s1 = strict_json(data)
    assert isinstance(s1['boot_id'], str) and len(s1['boot_id']) >= 8
    assert s1['uptime_s'] >= 0
    time.sleep(0.2)
    s2 = strict_json(bench.request('GET', '/state')[2])
    assert s2['boot_id'] == s1['boot_id'] and s2['uptime_s'] >= s1['uptime_s']


def test_nan_inside_the_state_becomes_null(bench: Bench) -> None:
    with bench.rover.lock:
        bench.rover.ctrl.history.appendleft({'kind': 'move', 'target': float('nan')})
    try:
        code, _, data = bench.request('GET', '/state')
        assert code == 200
        assert strict_json(data)['ctrl']['history'][0]['target'] is None
    finally:
        with bench.rover.lock:
            bench.rover.ctrl.history.popleft()


@pytest.mark.parametrize(
    'path, body',
    [
        ('/move', '{"distance": NaN}'),
        ('/move', '{"distance": "nan"}'),
        ('/move', '{"distance": 1000}'),
        ('/move', '{"distance": true}'),
        ('/turn', '{"angle": "inf"}'),
        ('/path', '{"points": [["inf", 0]]}'),
        ('/path', '{"points": [[1]]}'),
        ('/path', '{"points": [[20000, 0]]}'),
        ('/path', '{"points": [[1, 0]], "speed": Infinity}'),
        ('/sequence', '{"steps": [{"move": NaN}]}'),
        ('/sequence', '{"steps": [[1]]}'),
        ('/sequence', '{"steps": "abc"}'),
        ('/cmd', '{"v": "nan"}'),
        ('/cmd', '{"v": true}'),
        ('/cmd', '{"v": 50}'),
        ('/cmd', '{"T": 13, "X": Infinity}'),
        ('/cmd', '{"T": 1, "L": 5, "R": 5}'),
        ('/gimbal', '{"pan": "nan"}'),
        ('/gimbal', '{"pan_rate": NaN}'),
    ],
)
def test_bad_numbers_are_refused_and_nothing_moves(bench: Bench, path, body) -> None:
    code, _, data = bench.request('POST', path, body)
    assert code == 400, data
    assert 'error' in strict_json(data)
    time.sleep(0.1)
    ctrl = bench.rover.ctrl
    assert ctrl.mode == 'idle' and not ctrl.queue
    assert ctrl.v_cmd == 0.0 and ctrl.w_cmd == 0.0


def test_a_legitimate_move_still_works(bench: Bench) -> None:
    n = len(bench.rover.ctrl.history)
    code, _, _ = bench.request('POST', '/move', '{"distance": 0.05}')
    assert code == 200
    assert wait_for(lambda: len(bench.rover.ctrl.history) > n, 8.0)
    assert bench.rover.ctrl.history[0]['status'] == 'done'


def test_steps_refused_with_409_before_the_gyro_is_ready(bench: Bench) -> None:
    bench.rover.recalibrate_gyro()
    code, _, data = bench.request('POST', '/turn', '{"angle": 90}')
    assert code == 409 and 'gyro' in strict_json(data)['error']
    assert bench.rover.ctrl.mode == 'idle'
    assert bench.request('POST', '/cmd', '{"v": 0, "omega": 0}')[0] == 200
    assert wait_for(lambda: bench.rover.odom.bias_ready, 5.0)


def test_stop_answers_after_a_loop_error(bench: Bench) -> None:
    ctrl = bench.rover.ctrl
    real = ctrl.step
    calls = []

    def broken(t):
        calls.append(t)
        if len(calls) == 1:
            raise OverflowError('cannot convert float infinity to integer')
        return real(t)

    ctrl.step = broken
    try:
        assert wait_for(lambda: len(calls) > 2, 2.0)
        assert bench.request('POST', '/stop')[0] == 200
        state = strict_json(bench.request('GET', '/state')[2])
        assert state['loop_errors'] >= 1 and 'OverflowError' in state['last_loop_error']
        assert state['ctrl']['fault'].startswith('controller error')
    finally:
        del ctrl.step


def test_config_validated_all_or_nothing_and_saved(bench: Bench) -> None:
    cfg = bench.rover.cfg
    before = dict(cfg.__dict__)
    for body in (
        '{"v_manual": 0.25, "manual_timeout": NaN}',
        '{"v_manual": 0.25, "manual_timeout": 30}',
        '{"v_manual": 0.25, "track": 0}',
        '{"v_manual": 0.25, "w_min": null}',
        '[1, 2]',
    ):
        code, _, _ = bench.request('POST', '/config', body)
        assert code == 400, body
        assert cfg.__dict__ == before, body
    assert not os.path.exists(bench.cfg_path)
    code, _, data = bench.request('POST', '/config', '{"v_manual": 0.2}')
    assert code == 200 and strict_json(data)['changed'] == ['v_manual']
    with open(bench.cfg_path, encoding='utf-8') as f:
        assert strict_json(f.read().encode())['v_manual'] == 0.2
    assert strict_json(bench.request('GET', '/config')[2])['v_manual'] == 0.2
    bench.request('POST', '/config', json.dumps({'v_manual': before['v_manual']}))


def test_snapshot_fresh_old_and_refreshed(bench: Bench, monkeypatch) -> None:
    cam = bench.camera
    monkeypatch.setattr(rover_server, 'SNAPSHOT_WAIT', 0.4)
    cam._publish(b'\xff\xd8first')
    code, headers, data = bench.request('GET', '/snapshot.jpg')
    assert code == 200 and data == b'\xff\xd8first'
    assert 0 <= int(headers['X-Age-Ms']) < 1000

    cam.frame_t -= 10.0  # the camera stopped 10 s ago (idle release, or broken)
    code, headers, data = bench.request('GET', '/snapshot.jpg')
    assert code == 503 and int(headers['X-Age-Ms']) >= 10000
    assert 'error' in strict_json(data)

    threading.Timer(0.1, cam._publish, args=(b'\xff\xd8second',)).start()
    code, headers, data = bench.request('GET', '/snapshot.jpg')
    assert code == 200 and data == b'\xff\xd8second'  # waited for a picture taken now


def test_sigterm_stops_the_rover_like_ctrl_c(tmp_path) -> None:
    proc = subprocess.Popen(
        [
            sys.executable,
            '-u',
            os.path.join(ROVER_DIR, 'rover_server.py'),
            '--sim',
            '--no-camera',
            '--host',
            '127.0.0.1',
            '--port',
            str(SIGTERM_PORT),
            '--config',
            str(tmp_path / 'cfg.json'),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        lines = []
        end = time.monotonic() + 10
        while time.monotonic() < end:
            line = proc.stdout.readline()
            lines.append(line)
            if 'Open http' in line:
                break
        assert any('Open http' in x for x in lines), lines
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=10)
        assert 'Stopping the rover.' in out
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
