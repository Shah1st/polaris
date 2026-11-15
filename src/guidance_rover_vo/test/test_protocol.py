"""Wire format tests (pure Python, no ROS needed)."""

import socket
import threading

import numpy as np
import pytest

from guidance_rover_vo.protocol import RoverState, StreamReader, decode, encode


def _state(n: int = 5, version: int = 3) -> RoverState:
    pts = np.arange(3 * n, dtype=np.float32).reshape(n, 3) / 10.0
    return RoverState(
        stamp=12.5,
        x=1.0,
        y=-2.0,
        yaw=0.3,
        lost=False,
        quality=0.9,
        map_version=version,
        points=pts,
    )


def test_roundtrip() -> None:
    s = _state()
    out, used = decode(encode(s))
    assert used == len(encode(s))
    assert (out.x, out.y, out.yaw, out.lost, out.map_version) == (1.0, -2.0, 0.3, False, 3)
    np.testing.assert_array_equal(out.points, s.points)


def test_partial_buffer_waits_for_more() -> None:
    msg = encode(_state())
    for cut in (0, 3, 10, len(msg) - 1):
        assert decode(msg[:cut]) == (None, 0)


def test_empty_cloud() -> None:
    out, _ = decode(encode(_state(n=0)))
    assert out.points.shape == (0, 3)


def test_rejects_foreign_stream() -> None:
    with pytest.raises(ValueError):
        decode(b'\xff\xff\xff\xff' + b'x' * 10)


def test_stream_reader_over_socket() -> None:
    a, b = socket.socketpair()
    msgs = [encode(_state(version=v)) for v in range(3)]

    def send() -> None:
        data = b''.join(msgs)
        for i in range(0, len(data), 7):  # arrive in small pieces
            a.sendall(data[i : i + 7])
        a.close()

    threading.Thread(target=send).start()
    reader = StreamReader(b)
    assert [reader.read().map_version for _ in range(3)] == [0, 1, 2]
    with pytest.raises(ConnectionError):
        reader.read()
    b.close()
