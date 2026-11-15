"""Run the shared simulation as if it were the real robots, over the network.

    python sim/run_world.py [scene dir] [--auto] [--port 5050] [--state-port 7410]

HTTP on --port (same shape as the RaspRover's own web app):
  GET  /video_feed   rover camera, MJPEG (read it like the real robot's stream)
  GET  /drone_feed   drone camera (looking down), MJPEG
  GET  /state        JSON: true rover pose, drone pose, both cameras' intrinsics
                     and camera->world poses, room size, sim time
  GET  /grid/<rover|overhead>.png   true occupancy maps (white = blocked)
  GET  /snapshot     .npz of ONE simulation step taken after the request: rover
                     camera (rgb, exact depth, K, T), simulated D405 (d405_rgb,
                     d405_depth = sensor model, d405_depth_true, d405_K, d405_T),
                     true rover pose (t, rover = x, y, yaw). T = camera->world,
                     OpenCV axes. The D405 is only rendered while snapshots are
                     being requested.
  POST /cmd          RaspRover JSON command, e.g. {"T":1,"L":0.2,"R":0.2}
                     (wheel speeds m/s), or {"v":0.2,"omega":0.3}
  POST /drone        {"x":..,"y":..,"z":..,"yaw":..}
TCP on --state-port: true rover state in the polaris guidance_rover_vo format
(pose + points from the exact depth), for testing ROS nodes without the mapper.

--auto: the rover drives the planned route (trajectory.json waypoints) on its
own, back and forth, and the drone follows above it.
"""

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from world import RoverWorld  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('scene', nargs='?', default=os.path.join(HERE, 'scenes', 'apartment'))
ap.add_argument('--auto', action='store_true')
ap.add_argument('--port', type=int, default=5050)  # 5000 is taken by AirPlay on macOS
ap.add_argument('--state-port', type=int, default=7410)
ap.add_argument('--hz', type=float, default=10.0)
ap.add_argument(
    '--d405-mount',
    type=float,
    nargs=3,
    default=None,
    metavar=('X', 'Y', 'Z'),
    help='D405 position on the rover, body frame, metres (default 0.09 0 0.06: ASSUMED)',
)
ap.add_argument('--d405-max', type=float, default=1.0, help='D405 max valid depth, metres')
args = ap.parse_args()

wkw = {'d405_max_range': args.d405_max}
if args.d405_mount:
    wkw['d405_mount'] = tuple(args.d405_mount)
world = RoverWorld(args.scene, **wkw)
# The overhead map takes a few seconds to compute: start it now so /grid/overhead.png is ready.
threading.Thread(target=world.grid, args=('overhead',), daemon=True).start()
lock = threading.Lock()
latest = {'rover': b'', 'drone': b'', 'state': {}}
cmd = {'v': 0.0, 'omega': 0.0, 't': 0.0}
drone_target = None
# /snapshot: a request flags `want`; the main loop's next step renders every
# stream of that one step into `snap["arrays"]` (the D405 is only rendered
# then); the waiting handlers send it.
snap_cv = threading.Condition()
snap = {'seq': 0, 'arrays': None, 'want': False}


def jpg(rgb, q=85):
    return cv2.imencode(
        '.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q]
    )[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _mjpeg(self, key, hz):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with lock:
                    data = latest[key]
                if data:
                    self.wfile.write(
                        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                    )
                time.sleep(1.0 / hz)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/video_feed'):
            return self._mjpeg('rover', args.hz)
        if self.path.startswith('/drone_feed'):
            return self._mjpeg('drone', 5)
        if self.path.startswith('/state'):
            with lock:
                return self._json(latest['state'])
        if self.path.startswith('/snapshot'):
            with snap_cv:
                snap['want'] = True
                seq0 = snap['seq']
                fresh = snap_cv.wait_for(lambda: snap['seq'] > seq0, timeout=5.0)
                arrays = snap['arrays']
            if not fresh or arrays is None:
                return self._json({'error': 'no snapshot (sim stalled?)'}, 503)
            buf = io.BytesIO()
            np.savez(buf, **arrays)
            body = buf.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/grid/'):
            kind = 'overhead' if 'overhead' in self.path else 'rover'
            g = (world.grid(kind).T[::-1] * 255).astype(np.uint8)  # image: x right, y up
            data = cv2.imencode('.png', g)[1].tobytes()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        global drone_target
        n = int(self.headers.get('Content-Length', 0))
        try:
            msg = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._json({'error': 'bad json'}, 400)
        if self.path.startswith('/cmd'):
            if msg.get('T') == 1:
                left, right = float(msg.get('L', 0)), float(msg.get('R', 0))
                v, om = (left + right) / 2, (right - left) / 0.13
            else:
                v, om = float(msg.get('v', 0)), float(msg.get('omega', 0))
            with lock:
                cmd.update(v=v, omega=om, t=time.time())
            return self._json({'ok': True})
        if self.path.startswith('/drone'):
            drone_target = (
                float(msg['x']),
                float(msg['y']),
                float(msg.get('z', 2.2)),
                float(msg.get('yaw', 0)),
            )
            return self._json({'ok': True})
        self.send_error(404)


server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

state_server = None
if args.state_port:
    from state_stream import RoverState, StateServer

    state_server = StateServer(args.state_port)


def true_points(depth, K, stride=8, max_range=4.0):
    """Exact points in base_link from the rover camera depth (people would be here too)."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    ok = (z > 0.05) & (z < max_range)
    xc = (us - K[0, 2]) / K[0, 0] * z
    yc = (vs - K[1, 2]) / K[1, 1] * z
    # camera (x right, y down, z fwd) -> base_link (x fwd, y left, z up), camera ~ above centre
    from world import BASE_Z, CAM_IN_BASE

    pts = np.stack([z + CAM_IN_BASE[0], -xc + CAM_IN_BASE[1], -yc + BASE_Z + CAM_IN_BASE[2]], -1)
    return pts[ok].astype(np.float32)


def move_is_free(v, om, dt):
    """Would world.step(v, om, dt) go through (same integration, no side effect)?"""
    x, y, yaw = world.rover_pose
    n = max(1, int(round(dt / 0.02)))
    for _ in range(n):
        yaw += om * dt / n
        x += v * math.cos(yaw) * dt / n
        y += v * math.sin(yaw) * dt / n
        if not world.footprint_free(x, y, yaw):
            return False
    return True


# Auto mode: pure pursuit along the planned route, back and forth.
route = None
escape = None  # [(v, omega), steps left] while backing off an obstacle
if args.auto:
    tj = json.load(open(os.path.join(args.scene, 'trajectory.json')))
    route = np.array(tj['waypoints'])
    k_wp, direction = 1, 1

print(
    f'sim running: http://localhost:{args.port}/video_feed (rover), /drone_feed, /state'
    + (f', state stream tcp {args.state_port}' if state_server else ''),
    flush=True,
)
period = 1.0 / args.hz
next_t = time.time()
i = 0
while True:
    if route is not None:
        x, y, yaw = world.rover_pose
        tx, ty = route[k_wp]
        if math.hypot(tx - x, ty - y) < 0.3:
            if not 0 < k_wp + direction < len(route):
                direction = -direction
            k_wp += direction
            tx, ty = route[k_wp]
        a = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        v, om = (
            (0.0, max(-0.8, min(0.8, 2 * a)))
            if abs(a) > 0.8
            else (0.22 * math.cos(a), max(-0.8, min(0.8, 1.5 * a)))
        )
        if world.collided or escape:
            # Touched something: back off while turning, on whichever side is
            # free (always turning towards the target could stay blocked for
            # good), and keep at it for a few steps before pursuing again.
            side = 1.0 if a >= 0 else -1.0
            if not escape or not move_is_free(*escape[0], period):
                cands = (
                    (-0.12, 0.8 * side),
                    (-0.12, -0.8 * side),
                    (0.0, 0.8 * side),
                    (0.0, -0.8 * side),
                    (-0.12, 0.0),
                    (0.12, 0.8 * side),
                    (0.12, -0.8 * side),
                )
                free = [c for c in cands if move_is_free(*c, period)]
                escape = [free[0], 6] if free else None
            if escape:
                (v, om), escape[1] = escape[0], escape[1] - 1
                if escape[1] <= 0:
                    escape = None
            else:
                v, om = 0.0, 0.0
    else:
        with lock:
            v, om = (
                (cmd['v'], cmd['omega']) if time.time() - cmd['t'] < 1.0 else (0.0, 0.0)
            )  # 1 s watchdog
    world.step(v, om, period)
    x, y, yaw = world.rover_pose
    if drone_target is not None:
        world.set_drone(*drone_target)
    elif route is not None:
        world.set_drone(x, y, 2.2, 0.0)
    rgb, depth, K, T = world.rover_camera()
    snapshot = {
        't': world.t,
        'rover': {'x': x, 'y': y, 'yaw': yaw, 'collided': world.collided},
        'rover_camera': {
            'K': K.tolist(),
            'T_world_cam': T.tolist(),
            'size': list(world.rover_size),
        },
        'room': list(world.room),
        # /grid/<kind>.png: cell (i, j) covers x = origin_x + i * res, y = origin_y + j * res
        'grid': {
            'res': world.res,
            'origin': [-world.room[0] / 2, -world.room[1] / 2],
            'shape': list(world.grid('rover').shape),
        },
    }
    Kd4 = world._K(*world.d405_size, world.d405_fovy)
    snapshot['d405_camera'] = {
        'K': Kd4.tolist(),
        'T_world_cam': world._cam_pose_cv('d405_cam').tolist(),
        'size': list(world.d405_size),
        'mount': list(world.d405_mount),
        'mount_assumed': True,
        'range': list(world.d405_range),
    }
    if snap['want']:
        d = world.d405_camera()
        arrays = {
            't': np.float64(world.t),
            'rover': np.array([x, y, yaw]),
            'rgb': rgb,
            'depth': depth.astype(np.float32),
            'K': K,
            'T': T,
            'd405_rgb': d['rgb'],
            'd405_depth': d['depth'],
            'd405_depth_true': d['depth_true'],
            'd405_K': d['K'],
            'd405_T': d['T'],
            'd405_mount': np.array(world.d405_mount),
            'd405_range': np.array(world.d405_range),
        }
        with snap_cv:
            snap['arrays'] = arrays
            snap['seq'] += 1
            snap['want'] = False
            snap_cv.notify_all()
    if i % 2 == 0:
        drone, Kd, Td = world.drone_camera()
        snapshot['drone_camera'] = {
            'K': Kd.tolist(),
            'T_world_cam': Td.tolist(),
            'size': list(world.drone_size),
        }
        with lock:
            latest['drone'] = jpg(drone, 80)
    with lock:
        latest['rover'] = jpg(rgb)
        if 'drone_camera' not in snapshot and 'drone_camera' in latest['state']:
            snapshot['drone_camera'] = latest['state']['drone_camera']
        latest['state'] = snapshot
    if state_server is not None and state_server.n_clients:
        state_server.publish(
            RoverState(time.time(), x, y, yaw, False, 1.0, 0, true_points(depth, K))
        )
    i += 1
    next_t += period
    time.sleep(max(0.0, next_t - time.time()))
