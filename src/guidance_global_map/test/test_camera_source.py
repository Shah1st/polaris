import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from guidance_global_map.camera_source import (
    CameraReader,
    StillImageReader,
    camera_reader,
    iter_mjpeg,
)
from guidance_global_map.drone import OverheadDrone
from guidance_global_map.drone import camera_reader as exported_reader
from guidance_global_map.measure_scale import main as measure_main
from guidance_global_map.measure_scale import metres_per_pixel_from_points

# Ports reserved for these tests, on 127.0.0.1 only.
MJPEG_PORT = 8111
SNAPSHOT_PORT = 8112


def numbered_frame(index: int) -> np.ndarray:
    """A frame that carries its index in two flat halves, readable after JPEG compression."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :80] = (index // 16) * 16 + 8
    frame[:, 80:] = (index % 16) * 16 + 8
    return frame


def frame_index(frame: np.ndarray) -> int:
    high = int(frame[:, 20:60].mean() // 16)
    low = int(frame[:, 100:140].mean() // 16)
    return high * 16 + low


def jpeg(frame: np.ndarray) -> bytes:
    return cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


class FrameServer:
    """A local camera server. Records when each numbered frame was made (time.monotonic)."""

    def __init__(self, port: int, mjpeg: bool, fps: float = 20.0, length: bool = True) -> None:
        self.made: dict[int, float] = {}
        self.paused = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - name required by http.server
                if mjpeg:
                    self.stream()
                else:
                    index = len(server.made) % 256
                    server.made[index] = time.monotonic()
                    body = jpeg(numbered_frame(index))
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def stream(self) -> None:
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                index = 0
                try:
                    while True:
                        if server.paused.is_set():
                            time.sleep(0.02)
                            continue
                        server.made[index % 256] = time.monotonic()
                        body = jpeg(numbered_frame(index % 256))
                        header = b'--frame\r\nContent-Type: image/jpeg\r\n'
                        if length:
                            header += f'Content-Length: {len(body)}\r\n'.encode()
                        self.wfile.write(header + b'\r\n' + body + b'\r\n')
                        self.wfile.flush()
                        index += 1
                        time.sleep(1.0 / fps)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.httpd.daemon_threads = True
        self.url = f'http://127.0.0.1:{port}/video'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def mjpeg_server():
    server = FrameServer(MJPEG_PORT, mjpeg=True)
    yield server
    server.close()


@pytest.fixture
def snapshot_server():
    server = FrameServer(SNAPSHOT_PORT, mjpeg=False)
    yield server
    server.close()


def test_mjpeg_parts_are_split_with_and_without_content_length() -> None:
    frames = [jpeg(numbered_frame(i)) for i in (3, 200)]
    with_length = b''.join(
        b'--b\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(f) + f + b'\r\n'
        for f in frames
    )
    without = b''.join(b'--b\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n' for f in frames)

    for body in (with_length, without):
        parts = list(iter_mjpeg(io.BytesIO(body), chunk=1000))
        assert parts == frames


def test_mjpeg_stream_gives_the_newest_frame_and_a_frame_after_the_call(mjpeg_server) -> None:
    read = camera_reader(mjpeg_server.url)
    try:
        assert isinstance(read, CameraReader)
        first = read.read_new(timeout=5.0)
        assert first is not None and first.shape == (120, 160, 3)

        for _ in range(3):
            asked = time.monotonic()
            frame = read.read_new(timeout=3.0)
            # Made after the call: the latency allowance skips frames already on their way.
            assert mjpeg_server.made[frame_index(frame)] > asked

        newest = max(mjpeg_server.made, key=mjpeg_server.made.get)
        assert abs(frame_index(read()) - newest) <= 2  # never a backlog of old frames
    finally:
        read.close()


def test_a_stalled_stream_reads_as_none_after_max_age(mjpeg_server) -> None:
    read = camera_reader(mjpeg_server.url, max_age=0.5)
    try:
        assert read.read_new(timeout=5.0) is not None
        mjpeg_server.paused.set()
        time.sleep(1.0)
        assert read() is None
        assert read.read_new(timeout=0.3) is None
        mjpeg_server.paused.clear()
        assert read.read_new(timeout=3.0) is not None
    finally:
        read.close()


def test_snapshot_url_is_polled(snapshot_server) -> None:
    read = camera_reader(snapshot_server.url.replace('/video', '/shot.jpg'))
    try:
        assert read.read_new(timeout=5.0) is not None
        asked = time.monotonic()
        frame = read.read_new(timeout=3.0)
        assert snapshot_server.made[frame_index(frame)] > asked
        assert read.frames >= 2
    finally:
        read.close()
    count = read.frames
    time.sleep(0.3)
    assert read.frames == count  # closed: no more requests


def test_still_file_and_unreachable_sources(tmp_path) -> None:
    path = tmp_path / 'photo.png'
    cv2.imwrite(str(path), numbered_frame(42))

    still = camera_reader(str(path))
    assert isinstance(still, StillImageReader)
    assert frame_index(still()) == 42
    assert frame_index(still.read_new()) == 42
    still.close()

    dead = camera_reader('http://127.0.0.1:8119/nothing', max_age=0.5)
    try:
        assert dead() is None
        assert dead.read_new(timeout=0.3) is None
    finally:
        dead.close()
    assert exported_reader is camera_reader  # the name other parts import from drone


class LiveCamera:
    """Plays the reader of a live camera: read() still has the old frame, read_new() a new one."""

    def __init__(self, old: np.ndarray, new: np.ndarray) -> None:
        self.old, self.new, self.calls = old, new, []

    def __call__(self) -> np.ndarray:
        self.calls.append('read')
        return self.old

    def read_new(self, timeout: float) -> np.ndarray:
        self.calls.append(('read_new', timeout))
        return self.new


def test_forced_refresh_waits_for_a_frame_taken_after_the_call() -> None:
    from guidance_calib.synthetic import Box, Scene, render_top_down
    from guidance_global_map.pipeline import OverheadSettings

    def scene(x: float) -> np.ndarray:
        rover = Box(x, 0.50, 0.20, 0.23, color=(30, 30, 30))
        return render_top_down(Scene(marker_positions={}, boxes=[rover]))

    camera = LiveCamera(scene(0.40), scene(1.00))
    settings = OverheadSettings(metres_per_pixel=1.0 / 300.0, floor_window=0.6)
    drone = OverheadDrone(camera, settings, clock=lambda: 0.0, fresh_timeout=1.5)

    assert drone.rover_pose()[0] == pytest.approx(0.80, abs=0.02)  # plain read: old frame
    drone.refresh(force=True)
    assert camera.calls == ['read', ('read_new', 1.5)]
    assert drone.rover_pose()[0] == pytest.approx(1.40, abs=0.02)


def test_scale_from_two_points(tmp_path, capsys) -> None:
    assert metres_per_pixel_from_points((100, 100), (400, 500), 1.0) == pytest.approx(0.002)
    with pytest.raises(ValueError):
        metres_per_pixel_from_points((1, 1), (1, 1), 1.0)

    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), numbered_frame(1))
    measure_main([str(path), '--points', '10', '20', '110', '20', '--distance', '0.5'])
    assert 'metres_per_pixel: 0.005000' in capsys.readouterr().out


def test_opencv_capture_path_reads_a_video(tmp_path) -> None:
    # The same loop serves camera indexes and rtsp URLs; a video file exercises it here.
    path = str(tmp_path / 'clip.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (160, 120))
    for index in range(20):
        writer.write(numbered_frame(index))
    writer.release()

    read = CameraReader(path)
    try:
        frame = read.read_new(timeout=5.0)
        assert frame is not None and frame.shape == (120, 160, 3)
    finally:
        read.close()
