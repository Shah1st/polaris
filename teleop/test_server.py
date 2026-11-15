"""Tests of teleop/server.py against a stand-in rover (mock_rover.py), never the real one.

    python -m pytest teleop/test_server.py -q -p no:cacheprovider

Needs requests, OpenCV and numpy (the dashboard's own environment). Ports 8151-8161 on
127.0.0.1. The JavaScript of the pages is parsed with esprima when it is importable.
"""

import json
import math
import os
import re
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import pytest
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mock_rover  # noqa: E402
import server  # noqa: E402

ROVER_PORT, SERVER_PORT, PERC_PORT, DEAD_PORT = 8151, 8152, 8153, 8154


@pytest.fixture(scope='module')
def rover():
    m = mock_rover.MockRover(ROVER_PORT)
    yield m
    m.close()


def now_ms(offset: float = 0.0) -> float:
    return (time.time() + offset) * 1000.0


def serve(handler, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(('127.0.0.1', port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --- driving logic -------------------------------------------------------------------------


def test_nothing_sent_before_a_key(rover):
    t0 = time.monotonic()
    server.Robot(rover.url)
    time.sleep(0.5)
    assert [e for e in rover.log if e[0] >= t0] == []


def test_clamp_and_non_finite_refused(rover):
    r = server.Robot(rover.url)
    r.set(1e9, -1e9)
    assert r.want == (server.V_MAX, -server.W_MAX)
    r.set('0.1', '-0.5')
    assert r.want == (0.1, -0.5)
    for v, w in [(math.nan, 0), (0, math.nan), (math.inf, 0), (0, -math.inf), (10**400, 0)]:
        with pytest.raises(ValueError):
            r.set(v, w)
    with pytest.raises(ValueError):
        r.set(0.1, 0, t_client=math.nan)  # a NaN stamp used to disable the delay check for good
    with pytest.raises(ValueError):
        r.set(True, 0)
    assert r.want == (0.1, -0.5)  # unchanged by the refused ones
    r.stop()


def test_expiry_after_hold(rover):
    r = server.Robot(rover.url)
    t0 = time.monotonic()
    r.set(0.1, 0.0, now_ms(), 'a', 1)
    time.sleep(1.0)
    moving = rover.moving(t0)
    assert moving, 'the wanted speed never reached the robot'
    assert moving[-1][0] - t0 < server.HOLD + 0.1
    zeros = [e for e in rover.log if e[0] >= t0 and e[1] == '/cmd' and not e[2]['v']]
    assert zeros and zeros[0][0] > moving[-1][0]
    assert r.want == (0.0, 0.0)


def test_stale_per_browser():
    r = server.Robot(f'http://127.0.0.1:{DEAD_PORT}')  # nothing listens: no robot at all
    for _ in range(20):
        assert r.set(0.1, 0, now_ms(), 'A', 1)
    # Another computer whose clock is 1 s behind: it has its own reference, it can drive.
    assert all(r.set(0.1, 0, now_ms(-1.0), 'B', 1) for _ in range(20))
    # One of A's commands 2 s late is still refused.
    assert not r.set(0.1, 0, now_ms(-2.0), 'A', 1)
    assert r.n_stale == 1
    # A stamp from the future blocks that browser only, and only until it pauses.
    assert r.set(0.1, 0, now_ms(+100.0), 'C', 1)
    assert not r.set(0.1, 0, now_ms(), 'C', 1)
    assert r.set(0.1, 0, now_ms(), 'A', 1)
    r._sessions['C']['seen'] -= server.SESSION_IDLE + 1  # as after a pause
    assert r.set(0.1, 0, now_ms(), 'C', 1)


def test_stop_latches_until_a_new_press(rover):
    r = server.Robot(rover.url)
    for _ in range(3):
        assert r.set(0.1, 0, now_ms(), 'a', 1)
        time.sleep(0.05)
    late = now_ms(-0.05)  # left the browser 50 ms before the stop, arrives after it
    r.stop('a', 1)
    t_stop = time.monotonic()
    assert not r.set(0.1, 0, late, 'a', 1)
    assert not r.set(0.1, 0, now_ms(), 'a', 1)  # same press: keys not released
    time.sleep(0.6)
    stops = [e for e in rover.log if e[1] == '/stop' and e[0] >= t_stop - 0.01]
    assert stops, 'the stop never reached the robot'
    assert rover.moving(stops[0][0]) == [], 'the robot moved after the stop'
    assert r.n_latched == 2
    # Released and pressed again: drives.
    assert r.set(0.1, 0, now_ms(), 'a', 2)
    assert r.want == (0.1, 0)
    r.stop()


def test_stop_latches_a_press_the_server_never_saw():
    r = server.Robot(f'http://127.0.0.1:{DEAD_PORT}')
    r.stop('a', 4)  # the stop overtook the first command of press 4
    assert not r.set(0.1, 0, now_ms(), 'a', 4)
    assert r.set(0.1, 0, now_ms(), 'a', 5)


def test_stop_from_elsewhere_latches_every_browser():
    r = server.Robot(f'http://127.0.0.1:{DEAD_PORT}')
    assert r.set(0.1, 0, now_ms(), 'a', 3)
    r.stop()  # e.g. Space in another tab, or curl
    assert not r.set(0.1, 0, now_ms(), 'a', 3)
    assert r.set(0.1, 0, now_ms(), 'a', 4)


def test_stop_latch_for_clients_without_press_numbers():
    r = server.Robot(f'http://127.0.0.1:{DEAD_PORT}')
    assert r.set(0.1, 0)
    r.stop()
    assert not r.set(0.1, 0)
    assert r.set(0, 0)  # released
    assert r.set(0.1, 0)


# --- HTTP ----------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def http(rover, tmp_path_factory):
    suivi = tmp_path_factory.mktemp('suivi') / 'suivi.json'
    robot = server.Robot(rover.url)
    hub = server.CameraHub(rover.url + '/video_feed')
    handler = server.make_handler(
        robot,
        hub,
        8081,
        '',
        {},
        f'http://127.0.0.1:{DEAD_PORT}',
        str(suivi),
        f'http://127.0.0.1:{PERC_PORT}/debug.jpg',
    )
    srv = serve(handler, SERVER_PORT)
    yield {'url': f'http://127.0.0.1:{SERVER_PORT}', 'robot': robot, 'suivi': suivi}
    srv.shutdown()
    srv.server_close()


def raw_post(port: int, path: str, body: bytes, length: str | None = None) -> int:
    with socket.create_connection(('127.0.0.1', port), timeout=5) as s:
        s.sendall(
            f'POST {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: '
            f'{len(body) if length is None else length}\r\n\r\n'.encode()
            + body
        )
        return int(s.recv(1024).split(b' ')[1])


@pytest.mark.parametrize(
    'body',
    [
        b'{"v": NaN, "omega": 0}',
        b'{"v": "nan", "omega": 0}',  # float('nan') would pass the clamp as the full speed
        b'{"v": 0.1, "omega": Infinity}',
        b'{"v": 1e999}',
        b'[1, 2]',
        b'"go"',
        b'{"v": ' + b'9' * 400 + b'}',
        b'{"v": 0.1, "t": NaN}',
        b'{"v": 0.1, "press": "x"}',
        b'not json',
    ],
)
def test_bad_commands_are_400_and_never_move(http, rover, body):
    t0 = time.monotonic()
    assert raw_post(SERVER_PORT, '/api/cmd', body) == 400
    time.sleep(0.15)
    assert rover.moving(t0) == []
    assert http['robot'].want == (0.0, 0.0)


def test_bad_content_length_is_400(http):
    assert raw_post(SERVER_PORT, '/api/cmd', b'{}', length='abc') == 400


def test_stop_is_obeyed_whatever_its_body(http, rover):
    for body in (b'', b'[]', b'not json', b'{"press": NaN}', b'{"c": "a", "press": 1e999}'):
        n = http['robot'].n_stop
        assert raw_post(SERVER_PORT, '/api/stop', body) == 200
        assert http['robot'].n_stop == n + 1


def test_http_drive_then_stop_latch(http):
    url = http['url']

    def go():
        cmd = {'v': 0.1, 'omega': 0, 't': now_ms(), 'c': 'h', 'press': 7}
        return requests.post(url + '/api/cmd', json=cmd)

    r = go()
    assert r.json()['accepted'] is True
    requests.post(url + '/api/stop', json={'c': 'h', 'press': 7})
    r = go()
    assert r.status_code == 200 and r.json()['accepted'] is False
    s = requests.get(url + '/api/state').json()
    assert s['refused'] == 'latched' and s['n_latched'] >= 1 and s['boot'] == server.BOOT


def test_frame_carries_the_process_id(http):
    deadline = time.time() + 6
    while time.time() < deadline:
        r = requests.get(http['url'] + '/frame.jpg?c=t&p=ld&f=0', timeout=5)
        if r.status_code == 200:
            break
    assert r.status_code == 200 and r.content[:2] == b'\xff\xd8'
    assert r.headers['X-Boot'] == server.BOOT
    assert 'X-Boot' in r.headers['Access-Control-Expose-Headers']
    assert int(r.headers['X-Seq']) >= 1 and int(r.headers['X-Age-Ms']) >= 0


# --- camera pan-tilt -----------------------------------------------------------------------


def gimbal_posts(rover, after: float) -> list:
    with rover.lock:
        return [e[2] for e in rover.log if e[0] >= after and e[1] == '/gimbal']


def test_gimbal_held_key_is_repeated_then_stopped(rover):
    r = server.Robot(rover.url)
    t0 = time.monotonic()
    r.gimbal({'pan_rate': 1, 'tilt_rate': 0})
    time.sleep(server.GIMBAL_HOLD + 0.4)  # the page stopped repeating it
    posts = gimbal_posts(rover, t0)
    assert posts[0] == {'pan_rate': 1.0, 'tilt_rate': 0.0}
    assert posts.count(posts[0]) >= 2, 'a held key is repeated before the robot forgets it'
    assert posts[-1] == {'pan_rate': 0.0, 'tilt_rate': 0.0}
    assert rover.moving(t0) == []  # never a wheel command


def test_gimbal_angles_and_center_are_sent_once(rover):
    r = server.Robot(rover.url)
    t0 = time.monotonic()
    for order in ({'pan': 45, 'tilt': '-10.04'}, {'tilt': 30}, {'center': True}):
        r.gimbal(order)
        time.sleep(0.2)
    time.sleep(0.3)
    sent = [{'pan': 45.0, 'tilt': -10.0}, {'tilt': 30.0}, {'center': True}]
    assert gimbal_posts(rover, t0) == sent


def test_gimbal_angle_or_stop_ends_a_held_key(rover):
    r = server.Robot(rover.url)
    t0 = time.monotonic()
    r.gimbal({'pan_rate': -1})
    time.sleep(0.15)
    r.gimbal({'pan': 0})
    time.sleep(0.5)
    zero = {'pan_rate': 0.0, 'tilt_rate': 0.0}
    assert gimbal_posts(rover, t0) == [{'pan_rate': -1.0, 'tilt_rate': 0.0}, zero, {'pan': 0.0}]
    t1 = time.monotonic()
    r.gimbal({'tilt_rate': 1})
    time.sleep(0.1)
    r.stop()
    time.sleep(0.3)
    assert gimbal_posts(rover, t1)[-1] == zero


def test_gimbal_error_is_reported():
    r = server.Robot(f'http://127.0.0.1:{DEAD_PORT}')  # nothing listens
    r.gimbal({'center': True})
    deadline = time.time() + 6  # a refused connection takes about 2 s on Windows
    while time.time() < deadline and not r.g_err:
        time.sleep(0.1)
    assert r.g_err and 'ConnectionError' in r.g_err


@pytest.mark.parametrize(
    'body',
    [
        b'{"pan": NaN}',
        b'{"tilt": "x"}',
        b'{"pan": 500}',
        b'{"tilt": -91}',
        b'{"pan_rate": Infinity}',
        b'{"tilt_rate": true}',
        b'[1]',
        b'not json',
    ],
)
def test_bad_gimbal_orders_are_400_and_never_sent(http, rover, body):
    t0 = time.monotonic()
    assert raw_post(SERVER_PORT, '/api/gimbal', body) == 400
    time.sleep(0.15)
    assert gimbal_posts(rover, t0) == []


def test_http_gimbal_and_its_state(http, rover):
    t0 = time.monotonic()
    assert requests.post(http['url'] + '/api/gimbal', json={'pan': -30}).status_code == 200
    time.sleep(0.6)  # also a new read of the robot's state
    assert gimbal_posts(rover, t0) == [{'pan': -30.0}]
    s = requests.get(http['url'] + '/api/state').json()
    assert s['robot']['gimbal'] == {'pan': 0.0, 'tilt': 0.0, 'moving': False}  # the mock's
    assert s['gimbal_err'] is None


# --- depth camera (D405) -------------------------------------------------------------------


def test_d405_pictures_and_status_are_relayed(http, rover):
    url = http['url']
    r = requests.get(url + '/d405/color.jpg', timeout=5)
    assert r.status_code == 200 and r.content[:2] == b'\xff\xd8'
    r = requests.get(url + '/d405/depth.jpg?max=2&t=123', timeout=5)
    assert r.status_code == 200 and r.headers['Content-Type'] == 'image/jpeg'
    with rover.lock:
        assert rover.gets[-1] == '/d405/depth.jpg?max=2'  # only the range reaches the robot
    s = requests.get(url + '/d405/status', timeout=5).json()
    assert s['ok'] is True and s['camera']['name'] == 'RealSense D405'


@pytest.mark.parametrize('q', ['max=abc', 'max=50', 'max=0.01', 'max=nan', 'max=inf'])
def test_d405_bad_range_is_400_and_never_sent(http, rover, q):
    with rover.lock:
        n = len(rover.gets)
    assert requests.get(http['url'] + '/d405/depth.jpg?' + q, timeout=5).status_code == 400
    with rover.lock:
        assert not any(p.startswith('/d405/') for p in rover.gets[n:])


def test_d405_without_robot_is_503():
    handler = server.make_handler(
        server.Robot(f'http://127.0.0.1:{DEAD_PORT}'), None, 8081, '', {}
    )
    srv = serve(handler, 8161)
    try:
        r = requests.get('http://127.0.0.1:8161/d405/color.jpg', timeout=10)
        assert r.status_code == 503 and 'robot unreachable' in r.json()['error']
    finally:
        srv.shutdown()
        srv.server_close()


# --- mission page --------------------------------------------------------------------------


def test_mission_page_has_no_external_resource(http):
    r = requests.get(http['url'] + '/mission')
    assert r.status_code == 200 and 'text/html' in r.headers['Content-Type']
    assert not re.search(r'(src|href)\s*=\s*["\']?(https?:)?//', r.text)
    assert '@import' not in r.text and 'url(' not in r.text


def test_api_mission_waits_then_serves_the_document(http):
    suivi = http['suivi']
    if suivi.exists():
        suivi.unlink()
    r = requests.get(http['url'] + '/api/mission')
    assert r.status_code == 503 and r.json()['perception'] is True
    with open(os.path.join(HERE, 'mission_exemple.json')) as f:
        doc = json.load(f)
    doc['t'] = time.time()
    doc['rover'][2] = math.nan  # heading not measured yet: must not break the page
    suivi.write_text(json.dumps(doc))
    j = requests.get(http['url'] + '/api/mission').json()
    assert j['doc']['decision']['action'] == 'SUIVRE'
    assert j['doc']['rover'][2] is None
    assert 0 <= j['age_s'] < 5 and 0 <= j['age_doc_s'] < 5
    assert len(j['doc']['grille']['lignes']) == 48
    # Unchanged file: a small answer, not the map again.
    j2 = requests.get(http['url'] + '/api/mission?v=' + j['version']).json()
    assert j2['inchange'] is True and 'doc' not in j2
    suivi.write_text('{"t": 1, "etat": "en_c')  # not written atomically
    r = requests.get(http['url'] + '/api/mission')
    assert r.status_code == 503 and 'illisible' in r.json()['erreur']


def test_api_mission_without_suivi(rover):
    handler = server.make_handler(server.Robot(rover.url), None, 8081, '', {})
    srv = serve(handler, 8155)
    try:
        r = requests.get('http://127.0.0.1:8155/api/mission')
        assert r.status_code == 404 and '--suivi' in r.json()['erreur']
        assert r.json()['perception'] is False
        assert requests.get('http://127.0.0.1:8155/perception/debug.jpg').status_code == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_perception_debug_relay(http):
    url = http['url'] + '/perception/debug.jpg'
    assert requests.get(url).status_code == 503  # nothing listens yet
    import demo_mission

    demo_mission.perception_debug(PERC_PORT)
    r = requests.get(url)
    assert r.status_code == 200 and r.content[:2] == b'\xff\xd8'


# --- MJPEG readers -------------------------------------------------------------------------


def test_mjpeg_reader_that_left_frees_its_thread(rover):
    dead = mock_rover.MockRover(8156, snapshots=False)  # a robot with no camera frame
    hub = server.CameraHub(dead.url + '/video_feed')
    srv = serve(server.make_handler(server.Robot(dead.url), hub, 8081, '', {}), 8157)
    try:
        time.sleep(0.5)
        before = threading.active_count()
        socks = []
        for _ in range(5):
            s = socket.create_connection(('127.0.0.1', 8157), timeout=5)
            s.sendall(b'GET /video_rect HTTP/1.1\r\nHost: x\r\n\r\n')
            socks.append(s)
        time.sleep(0.5)
        assert threading.active_count() >= before + 5
        # A reader still there gets keep-alive bytes even without any frame.
        s = socks[0]
        s.settimeout(server.KEEPALIVE + 2)
        got = b''
        while b'\r\n\r\n' not in got:
            got += s.recv(4096)
        head, _, rest = got.partition(b'\r\n\r\n')
        assert b'multipart/x-mixed-replace' in head
        assert (rest or s.recv(16)).strip(b'\r\n') == b''
        for s in socks:
            s.close()
        deadline = time.time() + 3 * server.KEEPALIVE + 2
        while time.time() < deadline and threading.active_count() > before:
            time.sleep(0.2)
        assert threading.active_count() <= before, 'MJPEG threads alive after the readers left'
    finally:
        srv.shutdown()
        srv.server_close()
        dead.close()


# --- drone camera (--drone) ----------------------------------------------------------------


def test_camera_label_hides_credentials():
    label = server.camera_label('http://admin:secret@192.168.1.50:8081/video')
    assert label == '192.168.1.50:8081'
    assert server.camera_label('http://phone.local/shot.jpg') == 'phone.local'
    assert server.camera_label(None) is None


def first_frame(url: str, timeout: float = 8.0) -> requests.Response:
    deadline = time.time() + timeout
    while True:
        r = requests.get(url, timeout=5)
        if r.status_code == 200 or time.time() > deadline:
            return r
        time.sleep(0.1)


@pytest.mark.parametrize('path', ['/video', '/shot.jpg'])  # MJPEG stream, snapshot URL
def test_drone_camera_is_served(rover, path):
    phone = mock_rover.MockPhone(8158)
    drone = server.CameraHub(phone.url + path, snapshots=False)
    hub = server.CameraHub(rover.url + '/video_feed')
    handler = server.make_handler(server.Robot(rover.url), hub, 8081, '', {}, overhead=drone)
    srv = serve(handler, 8159)
    url = 'http://127.0.0.1:8159'
    try:
        r = first_frame(url + '/frame.jpg?s=overhead&c=t&p=ld')
        assert r.status_code == 200 and r.content[:2] == b'\xff\xd8'
        assert r.headers['Access-Control-Allow-Origin'] == '*'
        r2 = first_frame(url + '/frame.jpg?s=overhead&c=t&p=ld')
        assert int(r2.headers['X-Seq']) > int(r.headers['X-Seq'])  # a new picture each time
        d = requests.get(url + '/api/state').json()['drone']
        assert d['configured'] and d['camera'] == '127.0.0.1:8158' and d['error'] is None
        assert d['mode'] == ('stream' if path == '/video' else 'snapshot URL')
        # Re-served as MJPEG for local readers (lancer_overhead.py --camera).
        with requests.get(url + '/drone_feed', stream=True, timeout=8) as feed:
            assert feed.headers['Content-Type'].startswith('multipart/x-mixed-replace')
            got = b''
            for chunk in feed.iter_content(4096):
                got += chunk
                if b'\xff\xd9' in got:
                    break
        assert b'--frame' in got and b'\xff\xd8' in got
        if path == '/video':
            assert phone.readers == 1, 'the phone was read more than once'
    finally:
        srv.shutdown()
        srv.server_close()
        phone.close()


@pytest.mark.parametrize('path, expected', [('/locked', '401'), ('/clip.mp4', 'video/mp4')])
def test_wrong_drone_url_says_why_without_the_password(path, expected):
    phone = mock_rover.MockPhone(8160)
    drone = server.CameraHub(f'http://admin:s3cret@127.0.0.1:8160{path}', snapshots=False)
    try:
        deadline = time.time() + 8
        while drone.error is None and time.time() < deadline:
            drone.t_viewer = time.time()  # as if the page showed it
            time.sleep(0.1)
        assert drone.error is not None and expected in drone.error
        assert 's3cret' not in drone.error and 'admin' not in drone.error
        assert drone.seq == 0  # nothing taken for a picture
    finally:
        drone.t_viewer = 0.0
        phone.close()


def test_no_drone_camera_never_shows_the_rover_camera(http):
    url = http['url']
    first_frame(url + '/frame.jpg?c=t&p=ld&f=0')  # the rover camera works
    for s in ('overhead', 'nothing'):
        assert requests.get(url + f'/frame.jpg?s={s}&c=t&p=ld', timeout=5).status_code == 404
    assert requests.get(url + '/drone_feed', timeout=5).status_code == 404
    assert requests.get(url + '/api/state').json()['drone'] == {'configured': False}


# --- JavaScript syntax ---------------------------------------------------------------------


@pytest.mark.parametrize('page', ['index.html', 'sim.html', 'mission.html'])
def test_javascript_parses(page):
    esprima = pytest.importorskip('esprima')
    with open(os.path.join(HERE, page)) as f:
        html = f.read()
    scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
    assert scripts
    for js in scripts:
        # esprima 4 predates `??`; same precedence class for a syntax check
        esprima.parseScript(js.replace('??', '||'))
    assert len(re.findall(r'id="([^"]+)"', html)) == len(set(re.findall(r'id="([^"]+)"', html)))


def test_snapshot_wait_leaves_the_link_to_the_rover() -> None:
    """A drone camera asked by snapshots with a share of the link: the slower the picture, the
    longer the pause, so the camera never takes more than its share of the air time."""
    assert server.snapshot_wait(0.03, None) == pytest.approx(0.07)  # as before: every 0.1 s
    assert server.snapshot_wait(0.5, None) == 0.0
    assert server.snapshot_wait(0.05, 0.25) == pytest.approx(0.15)  # fast link: about 5 a second
    assert server.snapshot_wait(0.4, 0.25) == pytest.approx(1.2)  # slow link: 3 times the transfer
    assert server.snapshot_wait(3.0, 0.25) == server.SNAPSHOT_WAIT_MAX  # never longer than that
