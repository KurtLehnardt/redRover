"""Firmware rover backend, driven against an in-process fake board.

The fake speaks the real wire protocol, so these tests exercise the same
framing, handshake, and streaming path that a physical Arduino would — without
one attached.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from src.config import load_config
from src.rover import wire
from src.rover.backends import create_rover
from src.rover.backends.base import RoverBackend
from src.rover.backends.firmware import FirmwareRover
from src.rover.controller import RoverController, RoverState, Waypoint


class FakeBoard:
    """A minimal redRover firmware node implemented in Python.

    Mirrors the C++ node's observable behaviour: it answers Hello, publishes
    descriptors, acknowledges commands, latches an e-stop, and refuses to drive
    while latched.
    """

    def __init__(
        self, sensors: list[wire.SensorDescriptor] | None = None, board: str = "fake-rover"
    ):
        self.board = board
        self.sensors = sensors or []
        self.reader = wire.FrameReader()
        self.outbox = bytearray()
        self.drive_commands: list[tuple[int, int, int]] = []
        self.estop = False
        self.stopped = 0
        self.stream_rate = 0
        self.closed = False

    # -- the serial surface FirmwareRover uses --

    @property
    def in_waiting(self) -> int:
        return len(self.outbox)

    def read(self, count: int) -> bytes:
        data = bytes(self.outbox[:count])
        del self.outbox[:count]
        return data

    def write(self, data: bytes) -> int:
        for frame in self.reader.feed(data):
            self._handle(frame)
        return len(data)

    def close(self) -> None:
        self.closed = True

    # -- behaviour --

    def _send(self, msg_type: wire.MsgType, payload: bytes = b"", seq: int = 0) -> None:
        self.outbox.extend(wire.Frame(type=msg_type, seq=seq, payload=payload).encode())

    def _ack(self, frame: wire.Frame, code: wire.AckCode = wire.AckCode.OK) -> None:
        self._send(wire.MsgType.ACK, bytes([int(frame.type), int(code)]), frame.seq)

    def _handle(self, frame: wire.Frame) -> None:
        if frame.type is wire.MsgType.HELLO:
            payload = (
                bytes([wire.PROTOCOL_VERSION, len(self.sensors), 0])
                + struct.pack("<HHH", 500, 5000, 500)
                + self.board.encode().ljust(16, b"\x00")
            )
            self._send(wire.MsgType.HELLO_ACK, payload, frame.seq)

        elif frame.type is wire.MsgType.DESCRIBE:
            for descriptor in self.sensors:
                self._send(wire.MsgType.DESCRIPTOR, encode_descriptor(descriptor))
            self._ack(frame)

        elif frame.type is wire.MsgType.DRIVE:
            if self.estop:
                self._ack(frame, wire.AckCode.ESTOP_ACTIVE)
                return
            self.drive_commands.append(struct.unpack("<hhh", frame.payload[:6]))
            self._ack(frame)

        elif frame.type is wire.MsgType.STOP:
            self.stopped += 1
            self._ack(frame)

        elif frame.type is wire.MsgType.ESTOP:
            self.estop = bool(frame.payload[0])
            if self.estop:
                self.stopped += 1
            self._ack(frame)

        elif frame.type is wire.MsgType.SET_STREAM:
            self.stream_rate = struct.unpack("<H", frame.payload[:2])[0]
            self._ack(frame)

        else:
            self._ack(frame)

    def emit_sample(self, sensor_id: int, timestamp_ms: int, values: list[int]) -> None:
        payload = (
            bytes([sensor_id])
            + struct.pack("<I", timestamp_ms)
            + bytes([len(values)])
            + struct.pack(f"<{len(values)}i", *values)
        )
        self._send(wire.MsgType.SAMPLE, payload)


def encode_descriptor(d: wire.SensorDescriptor) -> bytes:
    return (
        bytes([d.id, int(d.kind), int(d.unit), d.channels])
        + struct.pack("<b", d.scale_exp)
        + struct.pack("<H", d.rate_hz)
        + bytes([1 if d.failed else 0])
        + d.name.encode().ljust(16, b"\x00")
    )


def accel_descriptor(rate_hz: int = 1000, sensor_id: int = 0) -> wire.SensorDescriptor:
    return wire.SensorDescriptor(
        id=sensor_id,
        kind=wire.SensorKind.ACCELERATION,
        unit=wire.Unit.METRE_PER_SECOND2,
        channels=3,
        scale_exp=-3,
        rate_hz=rate_hz,
        failed=False,
        name="vibration",
    )


@pytest.fixture
async def rover_and_board():
    board = FakeBoard(sensors=[accel_descriptor()])
    rover = FirmwareRover(port="fake", speed=0.5, max_speed_mps=1.0)
    rover._serial = board
    rover._pump_task = asyncio.create_task(rover._pump())
    try:
        yield rover, board
    finally:
        rover._pump_task.cancel()
        rover._serial = None


# === Backend contract ===


def test_both_backends_satisfy_the_protocol():
    """A patrol must not care which robot it is driving."""
    assert isinstance(RoverController(simulate=True), RoverBackend)
    assert isinstance(FirmwareRover(simulate=True), RoverBackend)


def test_factory_selects_by_connection():
    config = load_config()

    config.rover.connection = "ble"
    assert isinstance(create_rover(config, simulate=True), RoverController)

    config.rover.connection = "serial"
    assert isinstance(create_rover(config, simulate=True), FirmwareRover)

    config.rover.connection = "carrier-pigeon"
    with pytest.raises(ValueError, match="unknown"):
        create_rover(config, simulate=True)


def test_capabilities_report_the_sampling_ceiling():
    """The RVR+ cannot sample fast enough for bearing work, and says so."""
    rvr = RoverController(simulate=True).capabilities
    assert rvr.max_sensor_rate_hz == 50.0
    assert rvr.has_leds


# === Handshake and discovery ===


@pytest.mark.asyncio
async def test_handshake_reads_board_identity(rover_and_board):
    rover, board = rover_and_board
    hello = await rover._handshake()
    assert hello is not None
    assert hello.board == "fake-rover"
    assert hello.sensor_count == 1
    assert hello.command_timeout_ms == 500


@pytest.mark.asyncio
async def test_descriptors_populate_the_sensor_table(rover_and_board):
    rover, board = rover_and_board
    await rover._handshake()
    await rover._request_descriptors(expected=1)

    assert len(rover.sensors) == 1
    sensor = rover.sensors[0]
    assert sensor.name == "vibration"
    assert sensor.kind is wire.SensorKind.ACCELERATION
    assert sensor.descriptor.rate_hz == 1000

    caps = rover.capabilities
    assert caps.name == "fake-rover"
    # A 1 kHz accelerometer is the whole point: it clears the bearing bar.
    assert caps.max_sensor_rate_hz == 1000.0


@pytest.mark.asyncio
async def test_samples_are_scaled_into_physical_units(rover_and_board):
    rover, board = rover_and_board
    await rover._handshake()
    await rover._request_descriptors(expected=1)

    board.emit_sample(0, 1234, [-1024, 0, 1000])  # milli-g
    await asyncio.sleep(0.05)

    sensor = rover.sensors[0]
    assert sensor.samples_received == 1
    assert sensor.last_values == pytest.approx([-1.024, 0.0, 1.0])
    assert rover.sensor_data["vibration"] == pytest.approx([-1.024, 0.0, 1.0])


# === Commands ===


@pytest.mark.asyncio
async def test_drive_with_heading_sends_a_velocity_command(rover_and_board):
    rover, board = rover_and_board
    await rover.drive_with_heading(255, 0)
    await asyncio.sleep(0.05)

    assert board.drive_commands
    linear, angular, lateral = board.drive_commands[-1]
    assert linear == 1000  # 255/255 * 1.0 m/s -> 1000 mm/s
    assert angular == 0
    assert lateral == 0


@pytest.mark.asyncio
async def test_stop_is_forwarded(rover_and_board):
    rover, board = rover_and_board
    await rover.stop()
    await asyncio.sleep(0.05)
    assert board.stopped >= 1


@pytest.mark.asyncio
async def test_estop_latches_on_the_board_and_refuses_drive(rover_and_board):
    rover, board = rover_and_board
    await rover.emergency_stop()
    await asyncio.sleep(0.05)

    assert rover.estopped
    assert board.estop
    assert rover.state is RoverState.ESTOP

    with pytest.raises(RuntimeError, match="emergency stop"):
        await rover.drive_to(Waypoint(station_id="A", x=1.0, y=0.0))

    rover.clear_estop()
    await asyncio.sleep(0.05)
    assert not rover.estopped
    assert not board.estop


@pytest.mark.asyncio
async def test_streaming_requests_the_matching_rate(rover_and_board):
    rover, board = rover_and_board
    await rover.start_sensor_streaming(period_ms=10)
    await asyncio.sleep(0.05)
    assert rover.streaming
    assert board.stream_rate == 100

    await rover.stop_sensor_streaming()
    await asyncio.sleep(0.05)
    assert not rover.streaming


# === Navigation honesty ===


@pytest.mark.asyncio
async def test_simulated_drive_marks_the_position_estimated():
    rover = FirmwareRover(simulate=True, speed=1.0, max_speed_mps=1.0, time_scale=0.0)
    await rover.connect()
    await rover.drive_to(Waypoint(station_id="A", x=2.0, y=0.0))
    assert rover.position == (2.0, 0.0)
    assert rover.position_is_estimated is True


@pytest.mark.asyncio
async def test_odometry_sample_overrides_dead_reckoning(rover_and_board):
    rover, board = rover_and_board
    odometry = wire.SensorDescriptor(
        id=1,
        kind=wire.SensorKind.ODOMETRY,
        unit=wire.Unit.METRE,
        channels=3,
        scale_exp=-3,
        rate_hz=20,
        failed=False,
        name="odom",
    )
    board.sensors.append(odometry)
    await rover._handshake()
    await rover._request_descriptors(expected=2)

    assert rover.position_is_estimated is True
    board.emit_sample(1, 500, [1500, -250, 90])  # mm, mm, milli-degrees
    await asyncio.sleep(0.05)

    assert rover.position == pytest.approx((1.5, -0.25))
    # A measured pose is not an estimate, and the flag has to say so.
    assert rover.position_is_estimated is False
    assert rover.capabilities.has_odometry


# === Sensor source integration ===


@pytest.mark.asyncio
async def test_firmware_vibration_source_reports_the_delivered_rate(rover_and_board):
    """The sample must carry the rate that actually arrived, not the advertised one."""
    from src.sensors.sources import FirmwareVibrationSource

    rover, board = rover_and_board
    await rover._handshake()
    await rover._request_descriptors(expected=1)

    async def feed():
        for i in range(200):
            board.emit_sample(0, i, [0, 0, 1000 + (50 if i % 2 else -50)])
            await asyncio.sleep(0.001)

    source = FirmwareVibrationSource(rover, duration=0.3)
    feeder = asyncio.create_task(feed())
    sample = await source.read("M-001")
    feeder.cancel()

    assert len(sample.raw_signal) >= 8
    assert sample.sample_rate > 0
    # Rate is derived from what arrived over the real elapsed window.
    assert sample.sample_rate == pytest.approx(len(sample.raw_signal) / sample.duration, rel=0.2)


@pytest.mark.asyncio
async def test_firmware_source_refuses_a_board_with_no_accelerometer(rover_and_board):
    from src.sensors.sources import FirmwareVibrationSource, SourceUnavailable

    rover, board = rover_and_board
    board.sensors.clear()
    await rover._handshake()
    rover.sensors.clear()

    source = FirmwareVibrationSource(rover, duration=0.05)
    with pytest.raises(SourceUnavailable, match="no accelerometer"):
        await source.read("M-001")


# === Regressions found reviewing this PR ===


@pytest.mark.asyncio
async def test_drive_with_heading_is_continuous_and_non_blocking(rover_and_board):
    """Regression: it used to turn in place, blocking for the whole rotation.

    RoomExplorer calls this at ~10 Hz with small heading corrections. A
    blocking turn per call meant a firmware rover stuttered in place and
    covered no ground, so mapping produced an empty map.
    """
    import time as _time

    rover, board = rover_and_board
    rover._heading = 0.0

    started = _time.monotonic()
    await rover.drive_with_heading(200, 90)  # a 90-degree error
    elapsed = _time.monotonic() - started
    await asyncio.sleep(0.05)

    # Returns immediately rather than blocking for a rotation.
    assert elapsed < 0.1
    # One command, carrying both a forward and a steering component.
    assert len(board.drive_commands) == 1
    linear, angular, _ = board.drive_commands[0]
    assert angular > 0, "should steer toward the target heading"
    assert board.stopped == 0, "must not stop the motors to turn"


@pytest.mark.asyncio
async def test_heading_error_throttles_the_forward_component(rover_and_board):
    """A large error turns nearly on the spot; a small one barely slows down."""
    rover, board = rover_and_board

    rover._heading = 0.0
    await rover.drive_with_heading(255, 180)  # straight backwards
    await asyncio.sleep(0.02)
    big_error_linear = board.drive_commands[-1][0]

    rover._heading = 0.0
    await rover.drive_with_heading(255, 5)  # nearly aligned
    await asyncio.sleep(0.02)
    small_error_linear = board.drive_commands[-1][0]

    assert big_error_linear < small_error_linear
    assert small_error_linear > 900  # ~full speed at 1.0 m/s


@pytest.mark.asyncio
async def test_repeated_heading_commands_keep_driving(rover_and_board):
    """The explorer's call pattern must produce continuous motion."""
    rover, board = rover_and_board
    for heading in (0, 12, 25, 31, 44):
        await rover.drive_with_heading(150, heading)
    await asyncio.sleep(0.05)

    assert len(board.drive_commands) == 5
    assert board.stopped == 0
    assert all(linear > 0 for linear, _, _ in board.drive_commands)


@pytest.mark.asyncio
async def test_measured_heading_stops_the_open_loop_integrator(rover_and_board):
    rover, board = rover_and_board
    odometry = wire.SensorDescriptor(
        id=1,
        kind=wire.SensorKind.ODOMETRY,
        unit=wire.Unit.METRE,
        channels=3,
        scale_exp=0,
        rate_hz=20,
        failed=False,
        name="odom",
    )
    board.sensors.append(odometry)
    await rover._handshake()
    await rover._request_descriptors(expected=2)

    board.emit_sample(1, 100, [0, 0, 137])  # measured heading of 137 degrees
    await asyncio.sleep(0.05)
    assert rover.heading == pytest.approx(137.0)

    # An open-loop command must not overwrite a measured heading.
    await rover.drive_with_heading(100, 200)
    await asyncio.sleep(0.02)
    assert rover.heading == pytest.approx(137.0)


def test_clear_estop_warns_when_it_cannot_reach_the_board(caplog):
    """Silently leaving the board latched is worse than saying so.

    Deliberately synchronous: with no running loop there is nothing to
    schedule the clear frame on, which is exactly the case that used to fail
    without a word.
    """
    import logging

    rover = FirmwareRover(port="fake")
    rover._estop = True
    with caplog.at_level(logging.WARNING):
        rover.clear_estop()
    assert any(
        "still" in record.message and "latched" in record.message for record in caplog.records
    )


# === The backend split must reach every call site ===


def test_no_module_constructs_a_rover_outside_the_factory():
    """Regression: the mapper, the dashboard, and live_patrol each built a
    RoverController directly, so connection = "serial" raised ValueError."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        if path.name == "factory.py":
            continue
        if re.search(r"^\s*rover\s*=\s*RoverController\(", path.read_text(), re.M):
            offenders.append(str(path.relative_to(root)))
    assert offenders == [], f"should use create_rover(): {offenders}"


@pytest.mark.asyncio
async def test_explorer_runs_against_a_firmware_rover(tmp_path):
    """Regression: RoomExplorer's simulated loop needs inject_sensor_data,
    which FirmwareRover did not implement, so `explore_map --simulate` with
    connection = "serial" raised AttributeError."""
    from src.mapping import OccupancyGrid, RoomExplorer

    config = load_config()
    config.rover.connection = "serial"
    config.simulation.time_scale = 0.0

    rover = create_rover(config, simulate=True)
    await rover.connect()
    grid = OccupancyGrid(width_m=4.0, height_m=4.0, cell_cm=10)
    explorer = RoomExplorer(
        rover=rover,
        grid=grid,
        speed=60,
        duration=0.4,
        room_bounds_m=2.0,
        simulate=True,
        seed=1,
    )
    await explorer.run()

    assert grid.stats()["free"] > 0


def test_injected_values_never_masquerade_as_board_readings():
    rover = FirmwareRover(simulate=True)
    rover.inject_sensor_data(accelerometer=(0.0, 0.0, 1.0))
    assert rover.sensor_data["accelerometer"] == [0.0, 0.0, 1.0]
    # The overlay is separate from anything a board reported.
    assert rover.sensors == {}
