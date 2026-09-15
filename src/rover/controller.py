"""RVR+ motor control and navigation.

Supports three connection modes:
- ble:      native BLE via bleak (Mac M1 / any platform with BLE)
- uart:     Sphero SDK serial DAL (Raspberry Pi with UART hat)
- simulate: logs movements for development without hardware

Heading convention
------------------
The RVR+ uses **0 deg = +Y (forward), increasing clockwise**, so 90 deg = +X.
This differs from the mathematical convention (0 deg = +X, counter-clockwise)
and mixing the two silently steers the robot 90 deg off and mirrored.  Use
:func:`heading_to_vector` and :func:`vector_to_heading` rather than calling
``atan2`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading helpers (RVR+ convention: 0 deg = +Y, clockwise)
# ---------------------------------------------------------------------------


def heading_to_vector(heading_deg: float) -> tuple[float, float]:
    """Unit (dx, dy) for an RVR+ heading in degrees."""
    rad = math.radians(heading_deg % 360.0)
    return math.sin(rad), math.cos(rad)


def vector_to_heading(dx: float, dy: float) -> float:
    """RVR+ heading in degrees [0, 360) pointing along (dx, dy)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def heading_difference(target_deg: float, current_deg: float) -> float:
    """Shortest signed turn from *current* to *target*, in [-180, 180)."""
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def _consume_exception(task: asyncio.Task) -> None:
    """Retrieve a detached task's exception so asyncio does not warn."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("detached rover task failed: %s", exc)


# ---------------------------------------------------------------------------
# Sphero V2 BLE protocol constants
# ---------------------------------------------------------------------------

_ANTIDOS_CHARACTERISTIC = "00020005-574f-4f20-5370-6865726f2121"
_ANTIDOS_CHARACTERISTIC_ALT = "00010003-574f-4f20-5370-6865726f2121"  # RVR+ uses this
_API_V2_CHARACTERISTIC = "00010002-574f-4f20-5370-6865726f2121"
_ANTIDOS_PAYLOAD = b"usetheforce...band"

_SOP = 0x8D
_EOP = 0xD8
_ESCAPE = 0xAB
_ESCAPED_ESCAPE = bytes([0xAB, 0x23])
_ESCAPED_SOP = bytes([0xAB, 0x05])
_ESCAPED_EOP = bytes([0xAB, 0x50])

_FLAGS_DEFAULT = 0x3A  # requests_response | is_activity | has_target | has_source

# Target IDs
_TID_NORDIC = 0x01  # power, LEDs, system info
_TID_ST = 0x02  # drive, motors, IMU, sensors

# Source ID when talking from a BLE host
_SID_BLE = 0x01

# Device / Command IDs
_DID_POWER = 0x13
_CID_WAKE = 0x0D
_CID_BATTERY_PCT = 0x10

_DID_DRIVE = 0x16
_CID_RAW_MOTORS = 0x01
_CID_RESET_YAW = 0x06
_CID_DRIVE_WITH_HEADING = 0x07

_DID_LEDS = 0x1A
_CID_SET_LEDS_32 = 0x1A  # 32-bit mask — too large for BLE on RVR+
_CID_SET_LEDS_8 = 0x1C  # 8-bit mask — fits in a single BLE write

# Sensor streaming (DID=0x18)
_DID_SENSOR = 0x18
_CID_CONFIGURE_STREAMING = 0x39
_CID_START_STREAMING = 0x3A
_CID_STOP_STREAMING = 0x3B
_CID_CLEAR_STREAMING = 0x3C
_CID_STREAMING_DATA = 0x3D
_CID_RESET_LOCATOR = 0x13

# Sensor service IDs (16-bit)
_SENSOR_IMU = 0x0001
_SENSOR_ACCELEROMETER = 0x0002
_SENSOR_COLOR = 0x0003
_SENSOR_GYROSCOPE = 0x0004
_SENSOR_LOCATOR = 0x0006
_SENSOR_VELOCITY = 0x0007
_SENSOR_SPEED = 0x0008
_SENSOR_AMBIENT_LIGHT = 0x000A
_SENSOR_ENCODERS = 0x000B

# Data size codes
_DATA_SIZE_8BIT = 0x00
_DATA_SIZE_16BIT = 0x01
_DATA_SIZE_32BIT = 0x02

# Per-component normalization ranges.  Streaming values arrive as unsigned
# integers spanning the configured data size; each component is mapped back
# onto its own (min, max) range.
_SENSOR_COMPONENT_RANGES: dict[int, tuple[tuple[float, float], ...]] = {
    _SENSOR_ACCELEROMETER: ((-16.0, 16.0),) * 3,  # x, y, z in g
    _SENSOR_GYROSCOPE: ((-2000.0, 2000.0),) * 3,  # x, y, z in deg/s
    _SENSOR_LOCATOR: ((-16000.0, 16000.0),) * 2,  # x, y in metres
    _SENSOR_VELOCITY: ((-5.0, 5.0),) * 2,  # vx, vy in m/s
    _SENSOR_SPEED: ((0.0, 5.0),),  # m/s
    # Pitch and yaw span +/-180, roll only +/-90.  Using one range for all
    # three doubles the reported roll.
    _SENSOR_IMU: ((-180.0, 180.0), (-90.0, 90.0), (-180.0, 180.0)),
    _SENSOR_ENCODERS: ((0.0, 4294967295.0),) * 2,  # left, right raw ticks
}

_SENSOR_NAMES = {
    _SENSOR_ACCELEROMETER: "accelerometer",
    _SENSOR_GYROSCOPE: "gyroscope",
    _SENSOR_LOCATOR: "locator",
    _SENSOR_VELOCITY: "velocity",
    _SENSOR_SPEED: "speed",
    _SENSOR_ENCODERS: "encoders",
    _SENSOR_IMU: "imu",
    _SENSOR_COLOR: "color",
    _SENSOR_AMBIENT_LIGHT: "ambient_light",
}


class _SpheroV2Protocol:
    """Low-level Sphero V2 BLE packet builder / parser."""

    def __init__(self):
        self._seq: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._buf = bytearray()

    # -- sequence tracking --------------------------------------------------

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return seq

    # -- escaping -----------------------------------------------------------

    @staticmethod
    def _escape(data: bytes) -> bytes:
        out = bytearray()
        for b in data:
            if b == _ESCAPE:
                out += _ESCAPED_ESCAPE
            elif b == _SOP:
                out += _ESCAPED_SOP
            elif b == _EOP:
                out += _ESCAPED_EOP
            else:
                out.append(b)
        return bytes(out)

    @staticmethod
    def _unescape(data: bytes) -> bytes:
        out = bytearray()
        i = 0
        while i < len(data):
            if data[i] == _ESCAPE and i + 1 < len(data):
                code = data[i + 1]
                if code == 0x23:
                    out.append(_ESCAPE)
                elif code == 0x05:
                    out.append(_SOP)
                elif code == 0x50:
                    out.append(_EOP)
                else:
                    out.append(data[i])
                    out.append(code)
                i += 2
            else:
                out.append(data[i])
                i += 1
        return bytes(out)

    # -- checksum -----------------------------------------------------------

    @staticmethod
    def _checksum(payload: bytes) -> int:
        return (0xFF - (sum(payload) & 0xFF)) & 0xFF

    # -- build packet -------------------------------------------------------

    def build_packet(
        self,
        did: int,
        cid: int,
        target_id: int,
        data: bytes = b"",
        seq: int | None = None,
    ) -> bytes:
        """Build a fully-framed Sphero V2 packet ready to write."""
        if seq is None:
            seq = self._next_seq()
        payload = bytes([_FLAGS_DEFAULT, target_id, _SID_BLE, did, cid, seq]) + data
        chk = self._checksum(payload)
        escaped = self._escape(payload + bytes([chk]))
        return bytes([_SOP]) + escaped + bytes([_EOP])

    # -- response parsing (fed from BLE notifications) ----------------------

    def resolve(self, payload: bytes) -> None:
        """Resolve a pending future from an already-unescaped payload.

        Layout: FLAGS TID SID DID CID SEQ [DATA...] CHK
        """
        if len(payload) < 7:
            logger.debug("BLE: short packet dropped (%d bytes)", len(payload))
            return
        seq = payload[5]
        resp_data = payload[6:-1]  # strip checksum
        fut = self._pending.pop(seq, None)
        if fut and not fut.done():
            fut.set_result(resp_data)

    def expect_response(self, seq: int, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        """Register a future for a given sequence number."""
        fut = loop.create_future()
        self._pending[seq] = fut
        return fut

    def cancel_pending(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()


class RoverState(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    DWELLING = "dwelling"  # Parked at station, measuring
    RETURNING = "returning"
    ERROR = "error"
    ESTOP = "estop"


@dataclass
class Waypoint:
    """A machine station location."""

    station_id: str
    x: float  # metres from origin
    y: float
    heading: float = 0.0  # degrees, RVR+ convention
    name: str = ""


@dataclass
class PatrolRoute:
    """Ordered list of waypoints to visit."""

    name: str
    waypoints: list[Waypoint] = field(default_factory=list)


class RoverController:
    """Controls RVR+ movement and navigation."""

    def __init__(
        self,
        connection: str = "ble",
        speed: float = 0.3,
        simulate: bool = False,
        max_speed_mps: float = 1.5,
        max_drive_seconds: float = 20.0,
        time_scale: float = 1.0,
    ):
        self.connection = connection
        self.speed = speed
        self.simulate = simulate
        self.max_speed_mps = max_speed_mps
        self.max_drive_seconds = max_drive_seconds
        # Multiplies artificial delays in simulate mode (0.0 = no waiting).
        self.time_scale = max(0.0, time_scale)
        self.state = RoverState.IDLE
        self._position = (0.0, 0.0)
        self._heading = 0.0
        # True when position comes from dead reckoning rather than the locator.
        self._position_estimated = True
        self._rvr = None  # sphero_sdk handle (uart mode)
        self._ble_client = None  # bleak BleakClient (ble mode)
        self._proto = None  # _SpheroV2Protocol (ble mode)
        self._api_char = None  # cached BLE characteristic object
        self._estop = False

        # Sensor streaming state
        self._sensor_data = {
            "locator": (0.0, 0.0),  # x, y in metres
            "velocity": (0.0, 0.0),  # vx, vy in m/s
            "accelerometer": (0.0, 0.0, 0.0),  # ax, ay, az in g
            "gyroscope": (0.0, 0.0, 0.0),  # gx, gy, gz in deg/s
        }
        self._streaming = False
        # Incremented whenever a sensor's values are actually written, so a
        # consumer can distinguish a newly delivered packet from a re-read of
        # the same cached tuple. Polling `sensor_data` on a timer cannot tell
        # the difference, and reports the poll rate as the sample rate.
        self._sensor_updates: dict[str, int] = {}
        self._sensor_callbacks = []  # list of async callbacks
        # asyncio keeps only weak references to tasks, so a fire-and-forget
        # callback task can be collected mid-execution.  Hold strong refs.
        self._callback_tasks: set[asyncio.Task] = set()
        # token -> list of sensor IDs, in order configured
        self._streaming_slots: dict[int, list[int]] = {}
        self._ble_rx_buffer = bytearray()  # reassembly buffer for fragments

    # -- public state -------------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def streaming(self) -> bool:
        """True when sensor streaming is active."""
        return self._streaming

    @property
    def position_is_estimated(self) -> bool:
        """True when ``position`` is dead-reckoned rather than measured.

        Dead-reckoned coordinates accumulate error on every leg; callers that
        log or map a position should record this alongside it.
        """
        return self._position_estimated

    @property
    def capabilities(self):
        """What this chassis can do. See :mod:`src.rover.backends.base`."""
        from .backends.base import RoverCapabilities

        return RoverCapabilities(
            name="sphero-rvr-plus",
            holonomic=False,
            has_odometry=True,  # the locator, when streaming
            has_leds=True,
            has_battery=True,
            # The RVR+ streams sensors at tens of Hz. That is enough for
            # imbalance and misalignment, and nowhere near enough for bearing
            # defect frequencies -- see README "Sensor bandwidth".
            max_sensor_rate_hz=50.0,
            max_speed_mps=self.max_speed_mps,
        )

    @property
    def connected(self) -> bool:
        if self.simulate:
            return True
        if self._ble_client is not None:
            return bool(self._ble_client.is_connected)
        return self._rvr is not None

    # -- BLE helpers --------------------------------------------------------

    async def _ble_send(
        self,
        did: int,
        cid: int,
        target_id: int,
        data: bytes = b"",
        timeout: float = 3.0,
    ) -> bytes:
        """Build a packet, write it to the API V2 characteristic, and await
        the response.  Returns the response data bytes (empty on timeout)."""
        seq = self._proto._next_seq()
        loop = asyncio.get_running_loop()
        fut = self._proto.expect_response(seq, loop)
        pkt = self._proto.build_packet(did, cid, target_id, data, seq=seq)
        logger.debug("BLE TX: %s", pkt.hex())
        char = self._api_char or _API_V2_CHARACTERISTIC
        await self._ble_client.write_gatt_char(char, pkt, response=False)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            logger.warning("BLE: response timeout for DID=0x%02X CID=0x%02X seq=%d", did, cid, seq)
            return b""

    async def _ble_send_no_response(
        self,
        did: int,
        cid: int,
        target_id: int,
        data: bytes = b"",
    ):
        """Fire-and-forget packet (no response expected)."""
        pkt = self._proto.build_packet(did, cid, target_id, data)
        logger.debug("BLE TX (no-resp): %s", pkt.hex())
        char = self._api_char or _API_V2_CHARACTERISTIC
        await self._ble_client.write_gatt_char(char, pkt, response=False)

    def _ble_notification_handler(self, _sender, data: bytearray):
        """Callback fed to bleak start_notify.

        BLE notifications may be fragmented (MTU ~20 bytes).  This accumulates
        bytes in ``_ble_rx_buffer`` and processes each complete SOP...EOP
        packet.  Streaming data (DID=0x18, CID=0x3D) is routed to
        ``_handle_streaming_data``; everything else resolves a pending command.
        """
        logger.debug("BLE RX (%d bytes): %s", len(data), data.hex())
        self._ble_rx_buffer.extend(data)

        while True:
            sop_idx = self._ble_rx_buffer.find(bytes([_SOP]))
            if sop_idx == -1:
                self._ble_rx_buffer.clear()
                break
            if sop_idx > 0:
                del self._ble_rx_buffer[:sop_idx]

            eop_idx = self._ble_rx_buffer.find(bytes([_EOP]), 1)
            if eop_idx == -1:
                break  # Incomplete packet — wait for more data

            pkt = bytes(self._ble_rx_buffer[: eop_idx + 1])
            del self._ble_rx_buffer[: eop_idx + 1]

            inner = _SpheroV2Protocol._unescape(pkt[1:-1])
            if len(inner) < 7:
                continue
            did, cid = inner[3], inner[4]
            if did == _DID_SENSOR and cid == _CID_STREAMING_DATA:
                self._handle_streaming_data(inner)
            else:
                self._proto.resolve(inner)

    def _handle_streaming_data(self, payload: bytes):
        """Parse a streaming data notification and update sensor state.

        ``payload`` is the unescaped inner bytes
        (FLAGS TID SID DID CID SEQ DATA... CHK).
        """
        data = payload[6:-1]  # strip header (6 bytes) and checksum (1 byte)
        if len(data) < 1:
            return

        token = data[0]
        sensor_bytes = data[1:]

        slot_sensors = self._streaming_slots.get(token, [])
        if not slot_sensors:
            logger.debug("Streaming data for unknown token %d (%d bytes)", token, len(sensor_bytes))
            return

        offset = 0
        bytes_per_component = 4  # we always configure 32-bit data
        max_int = (1 << 32) - 1

        for sensor_id in slot_sensors:
            ranges = _SENSOR_COMPONENT_RANGES.get(sensor_id)
            if ranges is None:
                continue

            values = []
            for min_val, max_val in ranges:
                if offset + bytes_per_component > len(sensor_bytes):
                    logger.debug("Streaming data truncated for sensor 0x%04X", sensor_id)
                    return
                raw_uint = int.from_bytes(
                    sensor_bytes[offset : offset + bytes_per_component],
                    "big",
                    signed=False,
                )
                offset += bytes_per_component
                normalized = raw_uint / max_int
                values.append(normalized * (max_val - min_val) + min_val)

            name = _SENSOR_NAMES.get(sensor_id)
            if name and name in self._sensor_data:
                self._sensor_data[name] = tuple(values)
                self._sensor_updates[name] = self._sensor_updates.get(name, 0) + 1

        # The locator is a measured position, so trust it over dead reckoning.
        if _SENSOR_LOCATOR in slot_sensors:
            self._position = self._sensor_data["locator"]
            self._position_estimated = False

        self._dispatch_callbacks()

    def _dispatch_callbacks(self) -> None:
        if not self._sensor_callbacks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        snapshot = dict(self._sensor_data)
        for cb in self._sensor_callbacks:
            task = loop.create_task(cb(snapshot))
            self._callback_tasks.add(task)
            task.add_done_callback(self._callback_tasks.discard)

    # -- connect / disconnect -----------------------------------------------

    async def connect(self):
        """Connect to the RVR+."""
        if self.simulate:
            logger.info("RVR+ simulator connected")
            return

        if self.connection == "ble":
            await self._connect_ble()
        elif self.connection == "uart":
            await self._connect_uart()
        else:
            raise ValueError(f"Unknown connection mode: {self.connection!r}")

    async def _connect_ble(self, scan_timeout: float = 60.0):
        """Native BLE connection using bleak (works on Mac M1+).

        Retries scanning until the RVR+ is found or ``scan_timeout`` seconds
        have elapsed.
        """
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            logger.error("bleak is not installed -- run `pip install bleak`")
            raise

        deadline = time.monotonic() + scan_timeout
        device = None
        attempt = 0

        while device is None:
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.state = RoverState.ERROR
                raise RuntimeError(
                    f"BLE: no Sphero RVR+ found after {scan_timeout:.0f}s of scanning"
                )

            logger.info(
                "BLE: scanning for RVR+ devices (attempt %d, %.0fs remaining) ...",
                attempt,
                remaining,
            )
            devices = await BleakScanner.discover(timeout=min(10.0, remaining))
            for d in devices:
                if (d.name or "").startswith("RV-"):
                    device = d
                    logger.info("BLE: found %s (%s)", d.name, d.address)
                    break

            if device is None:
                wait = min(5.0, max(0.0, deadline - time.monotonic()))
                if wait > 0:
                    logger.info(
                        "BLE: RVR+ not found — retrying in %.0fs (power it on if not already)",
                        wait,
                    )
                    await asyncio.sleep(wait)

        logger.info("BLE: connecting to %s ...", device.name)
        self._ble_client = BleakClient(device, timeout=20.0)
        await self._ble_client.connect()
        if not self._ble_client.is_connected:
            self.state = RoverState.ERROR
            raise RuntimeError("BLE: failed to establish connection")
        logger.info("BLE: connected")

        # Cache the API V2 characteristic (avoids repeated service lookups)
        for service in self._ble_client.services:
            for char in service.characteristics:
                if char.uuid == _API_V2_CHARACTERISTIC:
                    self._api_char = char
                    logger.info("BLE: cached API V2 characteristic (handle %s)", char.handle)
                    break

        # Anti-DOS handshake — try the standard UUID, then the RVR+ alternate
        for antidos_uuid in (_ANTIDOS_CHARACTERISTIC, _ANTIDOS_CHARACTERISTIC_ALT):
            try:
                await self._ble_client.write_gatt_char(
                    antidos_uuid,
                    _ANTIDOS_PAYLOAD,
                    response=True,
                )
                logger.info("BLE: Anti-DOS handshake completed on %s", antidos_uuid[-8:])
                break
            except Exception:
                continue

        self._proto = _SpheroV2Protocol()
        await self._ble_client.start_notify(
            self._api_char or _API_V2_CHARACTERISTIC,
            self._ble_notification_handler,
        )
        logger.info("BLE: notifications enabled on API V2 characteristic")

        logger.info("BLE: sending wake command")
        await self._ble_send(_DID_POWER, _CID_WAKE, _TID_NORDIC)
        await asyncio.sleep(2)
        logger.info("RVR+ connected via BLE")

    async def _connect_uart(self):
        """Legacy UART connection via sphero_sdk (Raspberry Pi)."""
        try:
            from sphero_sdk import SerialAsyncDal, SpheroRvrAsync

            self._rvr = SpheroRvrAsync(dal=SerialAsyncDal(port="/dev/ttyTHS1"))
            await self._rvr.wake()
            await asyncio.sleep(2)
            logger.info("RVR+ connected via UART")
        except ImportError as exc:
            self.state = RoverState.ERROR
            raise RuntimeError(
                "sphero_sdk is not installed; cannot use connection='uart'. "
                "Install it, or run with simulate=True."
            ) from exc
        except Exception as e:
            logger.error("Failed to connect to RVR+ via UART: %s", e)
            self.state = RoverState.ERROR
            raise

    async def disconnect(self):
        """Stop the motors, then disconnect from the RVR+."""
        if self._ble_client and self._ble_client.is_connected:
            try:
                await self.stop()
                await self._ble_client.stop_notify(self._api_char or _API_V2_CHARACTERISTIC)
            except Exception as e:
                logger.debug("BLE: cleanup warning: %s", e)
            if self._proto is not None:
                self._proto.cancel_pending()
            try:
                await self._ble_client.disconnect()
            except Exception as e:
                logger.debug("BLE: disconnect warning: %s", e)
            self._ble_client = None
            self._proto = None
            logger.info("RVR+ disconnected (BLE)")
        elif self._rvr:
            try:
                await self.stop()
            except Exception as e:
                logger.debug("UART: stop warning: %s", e)
            await self._rvr.close()
            self._rvr = None
            logger.info("RVR+ disconnected (UART)")
        else:
            logger.info("RVR+ disconnected (simulated)")

        for task in list(self._callback_tasks):
            task.cancel()
        self._callback_tasks.clear()
        self.state = RoverState.IDLE

    # -- navigation ---------------------------------------------------------

    def travel_time_for(self, distance_m: float, speed_fraction: float | None = None) -> float:
        """Seconds to cover ``distance_m`` at the given throttle fraction.

        Derived from the calibrated ``max_speed_mps`` rather than a fixed
        50 cm/s, and clamped to ``max_drive_seconds`` so a bad waypoint cannot
        send the robot away indefinitely.
        """
        fraction = self.speed if speed_fraction is None else speed_fraction
        ground_speed = max(0.05, fraction * self.max_speed_mps)
        return min(distance_m / ground_speed, self.max_drive_seconds)

    async def drive_to(self, waypoint: Waypoint):
        """Navigate to a waypoint.

        Navigation is open-loop unless the locator is streaming; in that case
        the measured position is used on arrival.  Either way the resulting
        pose is never asserted to be exactly the waypoint — see
        :attr:`position_is_estimated`.
        """
        if self._estop:
            raise RuntimeError("rover is in emergency stop; call clear_estop() first")

        self.state = RoverState.NAVIGATING
        logger.info(
            "Navigating to station %s (%s) at (%.1f, %.1f)",
            waypoint.station_id,
            waypoint.name,
            waypoint.x,
            waypoint.y,
        )

        dx = waypoint.x - self._position[0]
        dy = waypoint.y - self._position[1]
        distance = math.hypot(dx, dy)
        heading_int = int(round(vector_to_heading(dx, dy))) % 360
        travel_time = self.travel_time_for(distance)

        try:
            if self.simulate:
                await self._sim_sleep(min(travel_time, 2.0))
                self._position = (waypoint.x, waypoint.y)
                self._heading = float(heading_int)
                self._position_estimated = True
                logger.info("Arrived at %s (simulated)", waypoint.station_id)
            elif self._ble_client or self._rvr:
                await self.drive_with_heading(int(self.speed * 255), heading_int)
                await self._sleep_interruptible(travel_time)
                await self.stop()
                self._settle_position(waypoint, distance, heading_int, travel_time)
            else:
                logger.warning("drive_to called with no connection; no movement")
        except asyncio.CancelledError:
            await self.stop()
            raise
        except Exception:
            self.state = RoverState.ERROR
            await self.stop()
            raise

        self.state = RoverState.DWELLING

    def _settle_position(
        self,
        waypoint: Waypoint,
        distance: float,
        heading_int: int,
        travel_time: float,
    ) -> None:
        """Record where we most likely ended up after a drive leg."""
        self._heading = float(heading_int)
        if self._streaming and not self._position_estimated:
            logger.info(
                "Arrived near %s — locator reads (%.2f, %.2f), target (%.2f, %.2f)",
                waypoint.station_id,
                self._position[0],
                self._position[1],
                waypoint.x,
                waypoint.y,
            )
            return

        ground_speed = max(0.05, self.speed * self.max_speed_mps)
        travelled = min(distance, travel_time * ground_speed)
        ux, uy = heading_to_vector(heading_int)
        self._position = (
            self._position[0] + ux * travelled,
            self._position[1] + uy * travelled,
        )
        self._position_estimated = True
        logger.info(
            "Arrived near %s — dead-reckoned (%.2f, %.2f), target (%.2f, %.2f). "
            "Enable sensor streaming for a measured position.",
            waypoint.station_id,
            self._position[0],
            self._position[1],
            waypoint.x,
            waypoint.y,
        )

    async def _sim_sleep(self, seconds: float) -> None:
        """Sleep for a simulated delay, scaled by ``time_scale``."""
        await asyncio.sleep(seconds * self.time_scale)

    async def _sleep_interruptible(self, seconds: float, step: float = 0.1) -> None:
        """Sleep in slices so an e-stop takes effect promptly."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._estop:
                return
            await asyncio.sleep(min(step, max(0.0, deadline - time.monotonic())))

    async def return_home(self):
        """Return to origin position."""
        self.state = RoverState.RETURNING
        home = Waypoint(station_id="HOME", x=0.0, y=0.0, heading=0.0, name="Home Base")
        await self.drive_to(home)
        self.state = RoverState.IDLE

    # -- sensor streaming ---------------------------------------------------

    async def start_sensor_streaming(self, period_ms: int = 100):
        """Configure and start sensor streaming on the ST processor.

        Slot 1 (token 0): accelerometer + gyroscope, 32-bit
        Slot 2 (token 1): locator + velocity, 32-bit
        """
        if self.simulate:
            self._streaming = True
            logger.info("Sensor streaming started (simulated, period=%dms)", period_ms)
            return

        if not self._ble_client:
            logger.warning("start_sensor_streaming: no BLE connection")
            return

        await self.stop_sensor_streaming()

        token_0 = 0
        slot1_data = bytes(
            [
                token_0,
                (_SENSOR_ACCELEROMETER >> 8) & 0xFF,
                _SENSOR_ACCELEROMETER & 0xFF,
                _DATA_SIZE_32BIT,
                (_SENSOR_GYROSCOPE >> 8) & 0xFF,
                _SENSOR_GYROSCOPE & 0xFF,
                _DATA_SIZE_32BIT,
            ]
        )
        await self._ble_send(_DID_SENSOR, _CID_CONFIGURE_STREAMING, _TID_ST, data=slot1_data)
        self._streaming_slots[token_0] = [_SENSOR_ACCELEROMETER, _SENSOR_GYROSCOPE]

        token_1 = 1
        slot2_data = bytes(
            [
                token_1,
                (_SENSOR_LOCATOR >> 8) & 0xFF,
                _SENSOR_LOCATOR & 0xFF,
                _DATA_SIZE_32BIT,
                (_SENSOR_VELOCITY >> 8) & 0xFF,
                _SENSOR_VELOCITY & 0xFF,
                _DATA_SIZE_32BIT,
            ]
        )
        await self._ble_send(_DID_SENSOR, _CID_CONFIGURE_STREAMING, _TID_ST, data=slot2_data)
        self._streaming_slots[token_1] = [_SENSOR_LOCATOR, _SENSOR_VELOCITY]

        period_ms = max(1, min(65535, int(period_ms)))
        await self._ble_send(
            _DID_SENSOR,
            _CID_START_STREAMING,
            _TID_ST,
            data=bytes([(period_ms >> 8) & 0xFF, period_ms & 0xFF]),
        )
        self._streaming = True
        logger.info("Sensor streaming started (period=%dms)", period_ms)

    async def stop_sensor_streaming(self):
        """Stop and clear sensor streaming on the ST processor."""
        if self.simulate:
            self._streaming = False
            logger.info("Sensor streaming stopped (simulated)")
            return

        if not self._ble_client:
            self._streaming = False
            return

        for cid, label in ((_CID_STOP_STREAMING, "stop"), (_CID_CLEAR_STREAMING, "clear")):
            try:
                await self._ble_send(_DID_SENSOR, cid, _TID_ST, timeout=2.0)
            except Exception as e:
                logger.debug("%s_streaming warning: %s", label, e)

        self._streaming = False
        self._streaming_slots.clear()
        logger.info("Sensor streaming stopped")

    async def reset_locator(self):
        """Reset the locator X and Y coordinates to zero."""
        logger.info("Resetting locator origin")
        self._sensor_data["locator"] = (0.0, 0.0)
        self._position = (0.0, 0.0)
        if self.simulate:
            self._position_estimated = True
            return
        if self._ble_client:
            await self._ble_send(_DID_SENSOR, _CID_RESET_LOCATOR, _TID_ST)
            self._position_estimated = False

    def add_sensor_callback(self, callback):
        """Register an async callback that fires on each sensor data update.

        The callback receives a dict with the latest sensor values:
        ``{'locator': (x,y), 'velocity': (vx,vy),
           'accelerometer': (ax,ay,az), 'gyroscope': (gx,gy,gz)}``
        """
        self._sensor_callbacks.append(callback)

    def remove_sensor_callback(self, callback) -> None:
        if callback in self._sensor_callbacks:
            self._sensor_callbacks.remove(callback)

    @property
    def sensor_data(self) -> dict:
        """Return a copy of the latest sensor data dict."""
        return dict(self._sensor_data)

    def sensor_update_count(self, name: str) -> int:
        """How many packets have updated `name` since connect.

        Lets a consumer sample once per delivered packet instead of once per
        clock tick, which is the difference between reporting the rate the
        sensor achieved and reporting the rate you happened to poll at.
        """
        return self._sensor_updates.get(name, 0)

    def inject_sensor_data(self, **values) -> None:
        """Set sensor values directly.

        Provided so simulators can drive the controller through a supported
        entry point instead of writing to private attributes.
        """
        for key, value in values.items():
            if key in self._sensor_data:
                self._sensor_data[key] = value
                self._sensor_updates[key] = self._sensor_updates.get(key, 0) + 1
        if "locator" in values:
            self._position = tuple(values["locator"])
        self._dispatch_callbacks()

    # -- LED control --------------------------------------------------------

    async def set_leds(self, r: int, g: int, b: int):
        """Set the headlight and status LEDs to the given RGB colour."""
        r, g, b = (max(0, min(255, int(v))) for v in (r, g, b))
        logger.debug("Setting LEDs to RGB(%d, %d, %d)", r, g, b)

        if self.simulate:
            return

        if self._ble_client:
            # 8-bit mask: each bit = one LED channel on the RVR+.
            #   bits 0-2: right headlight R,G,B
            #   bits 3-5: left headlight R,G,B
            #   bits 6-7: left status indicator R,G
            # One value byte per set bit; two writes cover all channels.
            await self._ble_send_no_response(
                _DID_LEDS,
                _CID_SET_LEDS_8,
                _TID_NORDIC,
                data=bytes([0x3F, r, g, b, r, g, b]),
            )
            await asyncio.sleep(0.075)
            await self._ble_send_no_response(
                _DID_LEDS,
                _CID_SET_LEDS_8,
                _TID_NORDIC,
                data=bytes([0xC0, r, g]),
            )
            await asyncio.sleep(0.075)
        elif self._rvr:
            await self._rvr.led_control.set_all_leds_rgb(r, g, b)

    # -- battery ------------------------------------------------------------

    async def get_battery(self) -> int | None:
        """Return battery percentage (0-100), or None if unavailable."""
        if self.simulate:
            return 100

        if self._ble_client:
            resp = await self._ble_send(_DID_POWER, _CID_BATTERY_PCT, _TID_NORDIC)
            if resp:
                logger.info("Battery: %s%%", resp[0])
                return resp[0]
            logger.warning("Battery query returned no data")
            return None

        if self._rvr:
            logger.warning("get_battery() not implemented for UART mode")
        return None

    # -- low-level motor helpers --------------------------------------------

    async def reset_yaw(self):
        """Reset the yaw angle to zero (current heading becomes 0)."""
        logger.info("Resetting yaw")
        self._heading = 0.0
        if self.simulate:
            return
        if self._ble_client:
            await self._ble_send(_DID_DRIVE, _CID_RESET_YAW, _TID_ST)
        elif self._rvr:
            await self._rvr.reset_yaw()

    async def set_raw_motors(
        self,
        left_mode: int,
        left_speed: int,
        right_mode: int,
        right_speed: int,
    ):
        """Set raw motor speeds.  Modes: 0=off, 1=forward, 2=reverse."""
        if self._estop:
            return
        logger.debug(
            "Raw motors: L(%d,%d) R(%d,%d)", left_mode, left_speed, right_mode, right_speed
        )
        if self.simulate:
            return
        if self._ble_client:
            await self._ble_send(
                _DID_DRIVE,
                _CID_RAW_MOTORS,
                _TID_ST,
                data=bytes(
                    [
                        left_mode & 0xFF,
                        left_speed & 0xFF,
                        right_mode & 0xFF,
                        right_speed & 0xFF,
                    ]
                ),
            )
        elif self._rvr:
            await self._rvr.raw_motors(
                left_mode=left_mode,
                left_speed=left_speed,
                right_mode=right_mode,
                right_speed=right_speed,
            )

    async def drive_with_heading(self, speed: int, heading: int):
        """Drive at *speed* (0-255) on *heading* (0-359 degrees, RVR+ convention)."""
        if self._estop and speed:
            return
        heading = int(heading) % 360
        speed = max(0, min(255, int(speed)))
        self._heading = float(heading)
        if self.simulate:
            return
        if self._ble_client:
            await self._ble_send(
                _DID_DRIVE,
                _CID_DRIVE_WITH_HEADING,
                _TID_ST,
                data=bytes([speed, (heading >> 8) & 0xFF, heading & 0xFF, 0]),
            )
        elif self._rvr:
            await self._rvr.drive_with_heading(speed=speed, heading=heading, flags=0)

    async def stop(self):
        """Immediately stop all motors.

        Shielded from cancellation: a cancelled patrol must still leave the
        motors off, and an ``await`` inside a cancelled task is otherwise
        interrupted before the command reaches the robot.
        """
        logger.info("Stopping rover")
        if self.simulate:
            return
        heading_int = int(self._heading) % 360

        async def _issue_stop():
            if self._ble_client:
                await self._ble_send_no_response(
                    _DID_DRIVE,
                    _CID_DRIVE_WITH_HEADING,
                    _TID_ST,
                    data=bytes([0, (heading_int >> 8) & 0xFF, heading_int & 0xFF, 0]),
                )
            elif self._rvr:
                await self._rvr.drive_with_heading(speed=0, heading=heading_int, flags=0)

        # ensure_future + shield keeps the command in flight even when the
        # caller is cancelled or we stop waiting for it. The done-callback
        # consumes any late exception so an abandoned task does not surface as
        # "Task exception was never retrieved" noise during shutdown.
        task = asyncio.ensure_future(_issue_stop())
        task.add_done_callback(_consume_exception)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
        except TimeoutError:
            logger.warning("stop(): motor-stop command not confirmed within 3s")
        except asyncio.CancelledError:
            logger.warning("stop(): caller cancelled; stop command still in flight")
            raise
        except Exception as e:
            logger.error("stop(): failed to stop motors: %s", e)

    async def emergency_stop(self):
        """Latch an emergency stop: motors off, further drive commands ignored."""
        self._estop = True
        self.state = RoverState.ESTOP
        logger.warning("EMERGENCY STOP engaged")
        try:
            await self.stop()
        except Exception as e:
            logger.error("emergency_stop: %s", e)

    def clear_estop(self) -> None:
        """Release the emergency stop latch."""
        self._estop = False
        if self.state is RoverState.ESTOP:
            self.state = RoverState.IDLE
        logger.info("Emergency stop cleared")

    @property
    def estopped(self) -> bool:
        return self._estop
