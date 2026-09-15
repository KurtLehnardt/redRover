"""Rover backend for any controller running the firmware in ``firmware/``.

This is what makes redRover platform-independent: an Arduino Uno, an ESP32, a
Pico, or a Teensy speaking the framed protocol in :mod:`src.rover.wire` drops
into the same patrol the Sphero RVR+ runs.

It is also the answer to the bandwidth problem. A BLE toy rover streams its
IMU at tens of Hz, which cannot resolve bearing defect frequencies. A firmware
rover reading an analog accelerometer streams at kHz, and the sensor registry
reports that rate, so the fusion engine knows the band is genuinely observable.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

from .. import wire
from ..controller import (
    RoverState,
    Waypoint,
    heading_difference,
    heading_to_vector,
    vector_to_heading,
)
from .base import RoverCapabilities

logger = logging.getLogger(__name__)

# How long to wait for a HelloAck before giving up on a port.
HANDSHAKE_TIMEOUT_S = 3.0
# Keepalive cadence. Must be comfortably inside the firmware's command
# watchdog, or the node will stop the motors mid-leg.
KEEPALIVE_INTERVAL_S = 0.15


@dataclass
class FirmwareSensor:
    """A sensor the board advertised, plus its most recent reading."""

    descriptor: wire.SensorDescriptor
    last_values: list[float] = field(default_factory=list)
    last_timestamp_ms: int = 0
    samples_received: int = 0

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def kind(self) -> wire.SensorKind:
        return self.descriptor.kind


class FirmwareRover:
    """Drives a firmware rover over a serial link."""

    def __init__(
        self,
        port: str = "",
        baud: int = 115200,
        speed: float = 0.3,
        simulate: bool = False,
        max_speed_mps: float = 1.5,
        max_drive_seconds: float = 20.0,
        turn_rate_mrad_s: int = 1500,
        heading_gain: float = 2.0,
        time_scale: float = 1.0,
    ):
        self.port = port
        self.baud = baud
        self.speed = speed
        self.simulate = simulate
        self.max_speed_mps = max_speed_mps
        self.max_drive_seconds = max_drive_seconds
        self.turn_rate_mrad_s = turn_rate_mrad_s
        # Proportional gain from heading error (rad) to angular rate (rad/s).
        self.heading_gain = heading_gain
        self.time_scale = max(0.0, time_scale)
        self.state = RoverState.IDLE

        self._serial = None
        self._reader = wire.FrameReader()
        self._seq = 0
        self._estop = False
        self._streaming = False
        self._position = (0.0, 0.0)
        self._heading = 0.0
        self._position_estimated = True
        # True once an odometry sensor has reported a measured heading, after
        # which the open-loop integrator stops guessing.
        self._heading_measured = False
        self._last_heading_update = 0.0
        self._commanded_heading = 0.0
        self._hello: wire.HelloAck | None = None
        self._status: wire.Status | None = None

        self.sensors: dict[int, FirmwareSensor] = {}
        # Simulation/test overlay; see inject_sensor_data.
        self._injected: dict[str, list[float]] = {}
        self._sensor_callbacks: list = []
        self._callback_tasks: set[asyncio.Task] = set()
        self._pump_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._acks: dict[int, asyncio.Future] = {}

    # -- capabilities -------------------------------------------------------

    @property
    def capabilities(self) -> RoverCapabilities:
        max_rate = max(
            (s.descriptor.rate_hz for s in self.sensors.values()), default=0
        )
        return RoverCapabilities(
            name=self._hello.board if self._hello else "firmware-rover",
            holonomic=bool(self._hello.holonomic) if self._hello else False,
            has_odometry=any(
                s.kind in (wire.SensorKind.ODOMETRY, wire.SensorKind.ENCODER)
                for s in self.sensors.values()
            ),
            has_leds=False,
            has_battery=any(
                s.kind is wire.SensorKind.BATTERY for s in self.sensors.values()
            )
            or (self._status is not None and self._status.battery_mv > 0),
            max_sensor_rate_hz=float(max_rate),
            max_speed_mps=(
                self._hello.max_linear_mm_s / 1000.0 if self._hello else self.max_speed_mps
            ),
        )

    # -- state --------------------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def position_is_estimated(self) -> bool:
        return self._position_estimated

    @property
    def streaming(self) -> bool:
        return self._streaming

    @property
    def estopped(self) -> bool:
        return self._estop

    @property
    def connected(self) -> bool:
        return self.simulate or self._serial is not None

    @property
    def sensor_data(self) -> dict:
        """Latest readings keyed by sensor name, in physical units."""
        data = {
            sensor.name: list(sensor.last_values) for sensor in self.sensors.values()
        }
        data.update(self._injected)
        return data

    def inject_sensor_data(self, **values) -> None:
        """Set sensor values directly.

        Simulation and test seam only -- RoomExplorer's simulated loop feeds
        the backend through this. Injected values are kept in a separate
        overlay so they can never be confused with a reading that arrived from
        a board.
        """
        for key, value in values.items():
            self._injected[key] = (
                [float(v) for v in value]
                if isinstance(value, (tuple, list))
                else [float(value)]
            )
        if "locator" in values:
            locator = list(values["locator"])
            self._position = (float(locator[0]), float(locator[1]))
        self._dispatch_callbacks()

    # -- connection ---------------------------------------------------------

    async def connect(self) -> None:
        if self.simulate:
            logger.info("Firmware rover simulator connected")
            return

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for connection='serial' (pip install pyserial)"
            ) from exc

        port = self.port or self._autodetect_port()
        if not port:
            raise RuntimeError(
                "no serial port configured or detected; set [rover].serial_port"
            )

        logger.info("Opening firmware rover on %s @ %d baud", port, self.baud)
        self._serial = await asyncio.to_thread(
            serial.Serial, port, self.baud, timeout=0, write_timeout=1.0
        )
        # Many boards reset when the port opens; wait for the bootloader.
        await asyncio.sleep(2.0)

        self._pump_task = asyncio.create_task(self._pump())

        hello = await self._handshake()
        if hello is None:
            await self.disconnect()
            raise RuntimeError(
                f"no redRover firmware responded on {port}. Is the sketch flashed "
                f"and the baud rate {self.baud} correct?"
            )
        self._hello = hello
        logger.info(
            "Firmware rover '%s' ready: %d sensors, %.2f m/s max, %d ms watchdog",
            hello.board, hello.sensor_count, hello.max_linear_mm_s / 1000.0,
            hello.command_timeout_ms,
        )

        await self._request_descriptors(expected=hello.sensor_count)
        self._keepalive_task = asyncio.create_task(self._keepalive())

    @staticmethod
    def _autodetect_port() -> str:
        try:
            from serial.tools import list_ports
        except ImportError:
            return ""
        for candidate in list_ports.comports():
            device = candidate.device or ""
            # The usual suspects across macOS, Linux, and Windows.
            if any(token in device for token in ("usbmodem", "usbserial", "ttyACM",
                                                 "ttyUSB", "COM")):
                logger.info("Auto-detected serial port %s (%s)", device, candidate.description)
                return device
        return ""

    async def _handshake(self) -> wire.HelloAck | None:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_S
        while time.monotonic() < deadline:
            await self._send(wire.MsgType.HELLO, expect_ack=False)
            await asyncio.sleep(0.2)
            if self._hello is not None:
                return self._hello
        return self._hello

    async def _request_descriptors(self, expected: int, timeout: float = 2.0) -> None:
        await self._send(wire.MsgType.DESCRIBE)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(self.sensors) < expected:
            await asyncio.sleep(0.05)
        if len(self.sensors) < expected:
            logger.warning(
                "board advertised %d sensors but only %d descriptors arrived",
                expected, len(self.sensors),
            )
        for sensor in self.sensors.values():
            marker = " [FAILED TO START]" if sensor.descriptor.failed else ""
            logger.info(
                "  sensor %d: %-16s %-14s %d ch @ %d Hz%s",
                sensor.descriptor.id, sensor.name, sensor.kind.name,
                sensor.descriptor.channels, sensor.descriptor.rate_hz, marker,
            )

    async def disconnect(self) -> None:
        try:
            if self._serial is not None:
                await self.stop()
        except Exception as exc:
            logger.debug("firmware rover: stop during disconnect: %s", exc)

        for task in (self._keepalive_task, self._pump_task):
            if task is not None:
                task.cancel()
        self._keepalive_task = self._pump_task = None

        for task in list(self._callback_tasks):
            task.cancel()
        self._callback_tasks.clear()

        if self._serial is not None:
            try:
                await asyncio.to_thread(self._serial.close)
            except Exception as exc:
                logger.debug("firmware rover: close: %s", exc)
            self._serial = None

        self.state = RoverState.IDLE
        logger.info("Firmware rover disconnected")

    # -- transport ----------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFF
        return self._seq

    async def _send(
        self, msg_type: wire.MsgType, payload: bytes = b"", expect_ack: bool = True,
        timeout: float = 1.0,
    ) -> wire.Ack | None:
        seq = self._next_seq()
        frame = wire.Frame(type=msg_type, seq=seq, payload=payload)

        if self.simulate or self._serial is None:
            return wire.Ack(of_type=int(msg_type), code=wire.AckCode.OK)

        future: asyncio.Future | None = None
        if expect_ack:
            future = asyncio.get_running_loop().create_future()
            self._acks[seq] = future

        data = frame.encode()
        try:
            await asyncio.to_thread(self._serial.write, data)
        except Exception as exc:
            self._acks.pop(seq, None)
            logger.error("firmware rover: write failed: %s", exc)
            return None

        if future is None:
            return None
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._acks.pop(seq, None)
            logger.warning("firmware rover: no ack for %s (seq %d)", msg_type.name, seq)
            return None

    async def _pump(self) -> None:
        """Read the serial port and dispatch inbound frames."""
        while self._serial is not None:
            try:
                waiting = self._serial.in_waiting
                if waiting:
                    data = await asyncio.to_thread(self._serial.read, waiting)
                    for frame in self._reader.feed(data):
                        self._dispatch(frame)
                else:
                    await asyncio.sleep(0.002)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("firmware rover: read loop error: %s", exc)
                await asyncio.sleep(0.05)

    def _dispatch(self, frame: wire.Frame) -> None:
        try:
            if frame.type is wire.MsgType.HELLO_ACK:
                self._hello = wire.HelloAck.parse(frame.payload)
            elif frame.type is wire.MsgType.DESCRIPTOR:
                descriptor = wire.SensorDescriptor.parse(frame.payload)
                self.sensors[descriptor.id] = FirmwareSensor(descriptor=descriptor)
            elif frame.type is wire.MsgType.SAMPLE:
                self._on_sample(wire.Sample.parse(frame.payload))
            elif frame.type is wire.MsgType.STATUS:
                self._status = wire.Status.parse(frame.payload)
                self._estop = self._status.estop
            elif frame.type is wire.MsgType.ACK:
                ack = wire.Ack.parse(frame.payload)
                future = self._acks.pop(frame.seq, None)
                if future is not None and not future.done():
                    future.set_result(ack)
            elif frame.type is wire.MsgType.LOG:
                message = wire.LogMessage.parse(frame.payload)
                logger.log(
                    {
                        wire.LogLevel.DEBUG: logging.DEBUG,
                        wire.LogLevel.INFO: logging.INFO,
                        wire.LogLevel.WARN: logging.WARNING,
                        wire.LogLevel.ERROR: logging.ERROR,
                    }[message.level],
                    "[firmware] %s", message.text,
                )
        except wire.ProtocolError as exc:
            logger.debug("firmware rover: unparseable %s: %s", frame.type.name, exc)

    def _on_sample(self, sample: wire.Sample) -> None:
        sensor = self.sensors.get(sample.sensor_id)
        if sensor is None:
            return
        sensor.last_values = [sensor.descriptor.to_physical(v) for v in sample.values]
        sensor.last_timestamp_ms = sample.timestamp_ms
        sensor.samples_received += 1

        if sensor.kind is wire.SensorKind.ODOMETRY and len(sensor.last_values) >= 2:
            # A measured pose beats dead reckoning.
            self._position = (sensor.last_values[0], sensor.last_values[1])
            self._position_estimated = False
            if len(sensor.last_values) >= 3:
                self._heading = sensor.last_values[2] % 360.0
                self._heading_measured = True

        self._dispatch_callbacks()

    def _dispatch_callbacks(self) -> None:
        if not self._sensor_callbacks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        snapshot = self.sensor_data
        for callback in self._sensor_callbacks:
            task = loop.create_task(callback(snapshot))
            self._callback_tasks.add(task)
            task.add_done_callback(self._callback_tasks.discard)

    async def _keepalive(self) -> None:
        """Hold off the firmware's command watchdog while driving.

        The watchdog is what stops the robot when the host dies; this is the
        host proving it is still alive, and it must beat the timeout.
        """
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_S)
            if self.state is RoverState.NAVIGATING and not self._estop:
                await self._send(wire.MsgType.PING, expect_ack=False)

    # -- motion -------------------------------------------------------------

    def travel_time_for(self, distance_m: float, speed_fraction: float | None = None) -> float:
        fraction = self.speed if speed_fraction is None else speed_fraction
        ground_speed = max(0.05, fraction * self.max_speed_mps)
        return min(distance_m / ground_speed, self.max_drive_seconds)

    async def drive_with_heading(self, speed: int, heading: int) -> None:
        """Drive at `speed` (0-255) steering toward `heading` degrees.

        This is a **continuous, non-blocking** command, matching how the RVR+
        behaves and how the explorer uses it: called at ~10 Hz with small
        heading corrections. Implementing it as turn-in-place-then-go made
        every call block for the duration of a rotation, so a firmware rover
        stuttered in place and never covered any ground.

        The firmware has no heading loop of its own, so the heading error is
        converted into an angular velocity and the chassis curves onto the
        heading while still moving. With an odometry sensor the error is
        measured; without one it is integrated from the commanded rate, and
        the pose stays flagged as estimated.
        """
        if self._estop and speed:
            return

        target = float(int(heading) % 360)
        error = heading_difference(target, self._heading)

        # Proportional steering, clamped to the chassis limit. A large error
        # also throttles the forward component back, so the rover turns on the
        # spot rather than driving a wide arc away from the target.
        angular_mrad_s = int(
            max(-self.turn_rate_mrad_s,
                min(self.turn_rate_mrad_s, math.radians(error) * self.heading_gain * 1000))
        )
        forward_scale = max(0.0, math.cos(math.radians(min(abs(error), 90.0))))
        linear_mm_s = int((speed / 255.0) * self.max_speed_mps * 1000 * forward_scale)

        await self._send(
            wire.MsgType.DRIVE,
            wire.drive_payload(linear_mm_s, angular_mrad_s),
            expect_ack=False,
        )
        self._integrate_heading(angular_mrad_s)
        self._commanded_heading = target

    def _integrate_heading(self, angular_mrad_s: int) -> None:
        """Advance the open-loop heading estimate since the last command.

        Only used when no odometry sensor reports a measured heading; the
        result is an estimate and is never presented as a measurement.
        """
        now = time.monotonic()
        if self._heading_measured:
            self._last_heading_update = now
            return
        elapsed = now - self._last_heading_update if self._last_heading_update else 0.0
        self._last_heading_update = now
        if elapsed <= 0.0 or elapsed > 1.0:
            return  # first command, or a gap too long to integrate honestly
        self._heading = (
            self._heading + math.degrees(angular_mrad_s / 1000.0 * elapsed)
        ) % 360.0

    async def _turn_by(self, degrees: float) -> None:
        """Rotate in place by `degrees`, open loop."""
        radians = math.radians(abs(degrees))
        rate = max(1, self.turn_rate_mrad_s) / 1000.0  # rad/s
        duration = min(radians / rate, self.max_drive_seconds)
        signed_rate = self.turn_rate_mrad_s if degrees > 0 else -self.turn_rate_mrad_s

        await self._send(
            wire.MsgType.DRIVE, wire.drive_payload(0, signed_rate), expect_ack=False,
        )
        await self._drive_for(duration)
        await self.stop()

    async def _drive_for(self, seconds: float) -> None:
        """Hold a command for `seconds`, pinging so the watchdog stays happy."""
        if self.simulate:
            await asyncio.sleep(seconds * self.time_scale)
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._estop:
                return
            await asyncio.sleep(min(KEEPALIVE_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            await self._send(wire.MsgType.PING, expect_ack=False)

    async def drive_to(self, waypoint: Waypoint) -> None:
        if self._estop:
            raise RuntimeError("rover is in emergency stop; call clear_estop() first")

        self.state = RoverState.NAVIGATING
        dx = waypoint.x - self._position[0]
        dy = waypoint.y - self._position[1]
        distance = math.hypot(dx, dy)
        target_heading = vector_to_heading(dx, dy)
        travel_time = self.travel_time_for(distance)

        logger.info(
            "Navigating to %s (%s) at (%.2f, %.2f) — %.2f m on heading %.0f",
            waypoint.station_id, waypoint.name, waypoint.x, waypoint.y,
            distance, target_heading,
        )

        try:
            if self.simulate:
                await asyncio.sleep(min(travel_time, 2.0) * self.time_scale)
                self._position = (waypoint.x, waypoint.y)
                self._heading = target_heading
                self._position_estimated = True
            else:
                turn = heading_difference(target_heading, self._heading)
                if abs(turn) > 1.0:
                    await self._turn_by(turn)
                self._heading = target_heading

                linear_mm_s = int(self.speed * self.max_speed_mps * 1000)
                await self._send(
                    wire.MsgType.DRIVE, wire.drive_payload(linear_mm_s, 0),
                    expect_ack=False,
                )
                await self._drive_for(travel_time)
                await self.stop()
                self._settle_position(distance, target_heading, travel_time)
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception:
            self.state = RoverState.ERROR
            await self.stop()
            raise

        self.state = RoverState.DWELLING

    def _settle_position(self, distance: float, heading: float, travel_time: float) -> None:
        if not self._position_estimated:
            return  # odometry already told us where we are
        ground_speed = max(0.05, self.speed * self.max_speed_mps)
        travelled = min(distance, travel_time * ground_speed)
        ux, uy = heading_to_vector(heading)
        self._position = (
            self._position[0] + ux * travelled,
            self._position[1] + uy * travelled,
        )
        self._position_estimated = True

    async def return_home(self) -> None:
        self.state = RoverState.RETURNING
        await self.drive_to(Waypoint(station_id="HOME", x=0.0, y=0.0, name="Home Base"))
        self.state = RoverState.IDLE

    async def stop(self) -> None:
        if self.state is RoverState.NAVIGATING:
            self.state = RoverState.IDLE
        await self._send(wire.MsgType.STOP, expect_ack=False)

    # -- safety -------------------------------------------------------------

    async def emergency_stop(self) -> None:
        self._estop = True
        self.state = RoverState.ESTOP
        logger.warning("EMERGENCY STOP engaged (firmware rover)")
        await self._send(wire.MsgType.ESTOP, bytes([1]), expect_ack=False)

    def clear_estop(self) -> None:
        """Release the local latch and tell the board, if a loop is running.

        The board latches independently, so a clear that never reaches it
        leaves the robot refusing to move with no indication why. When there
        is no running loop to schedule the frame on, say so rather than
        failing silently; ``clear_estop_async`` is the reliable form.
        """
        self._estop = False
        if self.state is RoverState.ESTOP:
            self.state = RoverState.IDLE
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "clear_estop() called outside an event loop: the board is still "
                "latched. Await clear_estop_async() to clear it."
            )
            return
        task = loop.create_task(self.clear_estop_async())
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    async def clear_estop_async(self) -> None:
        """Release the latch locally and on the board."""
        self._estop = False
        if self.state is RoverState.ESTOP:
            self.state = RoverState.IDLE
        await self._send(wire.MsgType.ESTOP, bytes([0]), expect_ack=False)

    # -- sensors ------------------------------------------------------------

    async def start_sensor_streaming(self, period_ms: int = 100) -> None:
        rate_hz = 0 if period_ms <= 0 else max(1, int(1000 / period_ms))
        await self._send(wire.MsgType.SET_STREAM, wire.set_stream_payload(rate_hz))
        self._streaming = True
        logger.info("Firmware sensor streaming started (%d Hz requested)", rate_hz)

    async def stop_sensor_streaming(self) -> None:
        # Rate 0 with an empty mask disables every sensor.
        await self._send(wire.MsgType.SET_STREAM, wire.set_stream_payload(0, 0))
        self._streaming = False
        logger.info("Firmware sensor streaming stopped")

    def add_sensor_callback(self, callback) -> None:
        self._sensor_callbacks.append(callback)

    def remove_sensor_callback(self, callback) -> None:
        if callback in self._sensor_callbacks:
            self._sensor_callbacks.remove(callback)

    def find_sensor(self, kind: wire.SensorKind) -> FirmwareSensor | None:
        """First working sensor of a given kind, or None."""
        for sensor in self.sensors.values():
            if sensor.kind is kind and not sensor.descriptor.failed:
                return sensor
        return None

    async def read_once(self, sensor_id: int) -> bool:
        ack = await self._send(wire.MsgType.READ_ONCE, bytes([sensor_id & 0xFF]))
        return ack is not None and ack.ok

    # -- optional hardware --------------------------------------------------

    async def set_leds(self, r: int, g: int, b: int) -> None:
        """Set a status LED through aux channel 0, if the sketch wires one up."""
        brightness = max(0, min(255, (r + g + b) // 3))
        await self._send(
            wire.MsgType.SET_AUX, wire.set_aux_payload(0, brightness), expect_ack=False,
        )

    async def get_battery(self) -> int | None:
        """Battery percentage, or None when the board does not report one."""
        sensor = self.find_sensor(wire.SensorKind.BATTERY)
        if sensor is not None and sensor.last_values:
            volts = sensor.last_values[0]
            # A 3S lithium pack: 9.0 V empty, 12.6 V full.
            percent = (volts - 9.0) / (12.6 - 9.0) * 100.0
            return int(max(0.0, min(100.0, percent)))
        if self._status is not None and self._status.battery_mv > 0:
            percent = (self._status.battery_mv / 1000.0 - 9.0) / 3.6 * 100.0
            return int(max(0.0, min(100.0, percent)))
        return None
