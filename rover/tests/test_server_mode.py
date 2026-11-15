"""Switching the server between the real rover and the simulated one."""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rover_server  # noqa: E402
from fake_esp32 import FakeEsp32, FakeSerial  # noqa: E402
from rover_core import Config  # noqa: E402

ARGS = SimpleNamespace(
    sim_drift=0.0, width=64, height=48, fps=5.0, quality=50, camera='0', no_camera=True
)


def wait(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def real_board(monkeypatch) -> FakeEsp32:
    """The 'real' rover: a fake board behind the serial port the server opens."""
    board = FakeEsp32()
    monkeypatch.setattr(rover_server, 'open_serial', lambda port: FakeSerial(board, boot_time=0.2))
    return board


def test_switch_real_to_sim_stops_the_real_rover_and_back(real_board, tmp_path) -> None:
    app = rover_server.App(Config(), str(tmp_path / 'cfg.json'), ARGS, serial_port='/dev/fake')
    try:
        app.set_mode('real')
        assert wait(lambda: app.rover.link.connected)
        app.rover.manual(0.2, 0.0)
        assert wait(lambda: real_board.wl > 0.1)  # the real rover drives

        app.set_mode('sim')
        state = app.state()
        assert state['mode'] == 'sim' and state['sim'] and state['real_available']
        assert real_board.target == (0.0, 0.0)  # told to stop before letting go
        assert wait(lambda: app.rover.link.connected)
        app.rover.manual(0.2, 0.0)
        assert wait(lambda: app.sim_board.wl > 0.1)  # the simulated one drives now
        time.sleep(0.3)
        assert real_board.target == (0.0, 0.0)  # no command reaches the real one any more

        app.set_mode('real')
        assert app.state()['mode'] == 'real' and not app.state()['sim']
        assert wait(lambda: app.rover.link.connected)
    finally:
        app.close()


def test_no_real_mode_without_a_serial_port(tmp_path) -> None:
    app = rover_server.App(Config(), str(tmp_path / 'cfg.json'), ARGS, serial_port=None)
    try:
        app.set_mode('sim')
        assert app.state()['real_available'] is False
        with pytest.raises(ValueError):
            app.set_mode('real')
        with pytest.raises(ValueError):
            app.set_mode('plane')
        assert app.mode == 'sim'
    finally:
        app.close()
