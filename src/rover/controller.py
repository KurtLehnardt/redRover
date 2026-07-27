"""RVR+ motor control and navigation.

Supports three connection modes:
- ble:      native BLE via bleak (Mac M1 / any platform with BLE)
- uart:     Sphero SDK serial DAL (Raspberry Pi with UART hat)
- simulate: logs movements for development without hardware
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

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
_TID_ST = 0x02      # drive, motors, IMU, sensors

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
_CID_SET_LEDS_8 = 0x1C   # 8-bit mask — fits in a single BLE write

# Sensor streaming (DID=0x18)
_DID_SENSOR = 0x18
_CID_CONFIGURE_STREAMING = 0x39
_CID_START_STREAMING = 0x3A
_CID_STOP_STREAMING = 0x3B
_CID_CLEAR_STREAMING = 0x3C
_CID_STREAMING_DATA = 0x3D
_CID_RESET_LOCATOR = 0x13

# Sensor service IDs (16-bit)
_SENSOR_ACCELEROMETER = 0x0002
_SENSOR_GYROSCOPE = 0x0004
_SENSOR_LOCATOR = 0x0006
_SENSOR_VELOCITY = 0x0007
_SENSOR_SPEED = 0x0008
_SENSOR_ENCODERS = 0x000B
_SENSOR_IMU = 0x0001
_SENSOR_COLOR = 0x0003
_SENSOR_AMBIENT_LIGHT = 0x000A

# Data size codes
_DATA_SIZE_8BIT = 0x00
_DATA_SIZE_16BIT = 0x01
_DATA_SIZE_32BIT = 0x02

# Sensor normalization ranges (min, max, num_components)
_SENSOR_RANGES = {
    0x0002: (-16.0, 16.0, 3),      # Accelerometer: x,y,z in g
    0x0004: (-2000.0, 2000.0, 3),   # Gyroscope: x,y,z in deg/s
    0x0006: (-16000.0, 16000.0, 2), # Locator: x,y in meters
    0x0007: (-5.0, 5.0, 2),         # Velocity: vx,vy in m/s
    0x0008: (0.0, 5.0, 1),          # Speed: m/s
    0x0001: (-180.0, 180.0, 3),     # IMU: pitch,roll,yaw (yaw range is -180..180 but pitch is -180..180, roll -90..90)
    0x000B: (0, 4294967295, 2),     # Encoders: left,right ticks (raw uint32)
}

# Mapping from sensor ID to friendly name
_SENSOR_NAMES = {
    0x0002: 'accelerometer',
    0x0004: 'gyroscope',
    0x0006: 'locator',
    0x0007: 'velocity',
    0x0008: 'speed',
    0x000B: 'encoders',
    0x0001: 'imu',
    0x0003: 'color',
    0x000A: 'ambient_light',
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
        seq: Optional[int] = None,
    ) -> bytes:
        """Build a fully-framed Sphero V2 packet ready to write."""
        if seq is None:
            seq = self._next_seq()
        payload = bytes([_FLAGS_DEFAULT, target_id, _SID_BLE, did, cid, seq]) + data
        chk = self._checksum(payload)
        escaped = self._escape(payload + bytes([chk]))
        return bytes([_SOP]) + escaped + bytes([_EOP])

    # -- response parsing (fed from BLE notifications) ----------------------

    def feed(self, data: bytes):
        """Feed raw notification bytes; resolves pending futures on match."""
        self._buf.extend(data)
        self._try_parse()

    def _try_parse(self):
        while True:
            start = self._buf.find(bytes([_SOP]))
            if start == -1:
                self._buf.clear()
                return
            end = self._buf.find(bytes([_EOP]), start + 1)
            if end == -1:
                # trim anything before the SOP
                self._buf = self._buf[start:]
                return
            raw = bytes(self._buf[start + 1: end])
            self._buf = self._buf[end + 1:]
            payload = self._unescape(raw)
            if len(payload) < 7:
                logger.debug("BLE: short packet dropped (%d bytes)", len(payload))
                continue
            # payload layout: FLAGS TID SID DID CID SEQ [DATA...] CHK
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


class RoverState(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    DWELLING = "dwelling"  # Parked at station, measuring
    RETURNING = "returning"
    ERROR = "error"


@dataclass
class Waypoint:
    """A machine station location."""
    station_id: str
    x: float  # meters from origin
    y: float
    heading: float = 0.0  # degrees
    name: str = ""


@dataclass
class PatrolRoute:
    """Ordered list of waypoints to visit."""
    name: str
    waypoints: list[Waypoint] = field(default_factory=list)


class RoverController:
    """Controls RVR+ movement and navigation."""

    def __init__(self, connection: str = "ble", speed: float = 0.3, simulate: bool = False):
        self.connection = connection
        self.speed = speed
        self.simulate = simulate
        self.state = RoverState.IDLE
        self._position = (0.0, 0.0)
        self._heading = 0.0
        self._rvr = None          # sphero_sdk handle (uart mode)
        self._ble_client = None   # bleak BleakClient (ble mode)
        self._proto = None        # _SpheroV2Protocol (ble mode)
        self._api_char = None     # cached BLE characteristic object

        # Sensor streaming state
        self._sensor_data = {
            'locator': (0.0, 0.0),             # x, y in meters
            'velocity': (0.0, 0.0),            # vx, vy in m/s
            'accelerometer': (0.0, 0.0, 0.0),  # ax, ay, az in g
            'gyroscope': (0.0, 0.0, 0.0),      # gx, gy, gz in deg/s
        }
        self._streaming = False
        self._sensor_callbacks = []  # list of async callbacks
        # Tracks which sensor IDs are configured in each streaming slot (token)
        # token -> list of sensor IDs, in order configured
        self._streaming_slots = {}

    # -- BLE helpers --------------------------------------------------------

    async def _ble_send(
        self, did: int, cid: int, target_id: int, data: bytes = b"", timeout: float = 3.0,
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
        except asyncio.TimeoutError:
            logger.warning("BLE: response timeout for DID=0x%02X CID=0x%02X seq=%d", did, cid, seq)
            return b""

    async def _ble_send_no_response(
        self, did: int, cid: int, target_id: int, data: bytes = b"",
    ):
        """Fire-and-forget packet (no response expected)."""
        pkt = self._proto.build_packet(did, cid, target_id, data)
        logger.debug("BLE TX (no-resp): %s", pkt.hex())
        char = self._api_char or _API_V2_CHARACTERISTIC
        await self._ble_client.write_gatt_char(char, pkt, response=False)

    def _ble_notification_handler(self, _sender, data: bytearray):
        """Callback fed to bleak start_notify.

        Distinguishes streaming data notifications (DID=0x18, CID=0x3D)
        from regular command responses and routes them accordingly.
        """
        logger.debug("BLE RX: %s", data.hex())
        raw = bytes(data)

        # Try to detect streaming data before feeding to the protocol parser.
        # Quick peek: find SOP/EOP, unescape, check DID/CID.
        sop_idx = raw.find(bytes([_SOP]))
        eop_idx = raw.find(bytes([_EOP]), sop_idx + 1) if sop_idx != -1 else -1
        if sop_idx != -1 and eop_idx != -1:
            inner = _SpheroV2Protocol._unescape(raw[sop_idx + 1:eop_idx])
            # Payload layout: FLAGS TID SID DID CID SEQ [DATA...] CHK
            if len(inner) >= 7:
                did = inner[3]
                cid = inner[4]
                if did == _DID_SENSOR and cid == _CID_STREAMING_DATA:
                    self._handle_streaming_data(inner)
                    return

        # Not streaming data — feed to protocol for command response matching
        self._proto.feed(raw)

    def _handle_streaming_data(self, payload: bytes):
        """Parse a streaming data notification payload and update sensor state.

        ``payload`` is the unescaped inner bytes (FLAGS TID SID DID CID SEQ DATA... CHK).
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
        for sensor_id in slot_sensors:
            range_info = _SENSOR_RANGES.get(sensor_id)
            if range_info is None:
                continue
            min_val, max_val, num_components = range_info
            # We always configure 32-bit data
            bytes_per_component = 4
            bits = 32

            values = []
            for _ in range(num_components):
                if offset + bytes_per_component > len(sensor_bytes):
                    logger.debug("Streaming data truncated for sensor 0x%04X", sensor_id)
                    return
                raw_uint = int.from_bytes(
                    sensor_bytes[offset:offset + bytes_per_component], 'big', signed=False,
                )
                offset += bytes_per_component
                max_int = (1 << bits) - 1
                normalized = raw_uint / max_int if max_int else 0.0
                value = normalized * (max_val - min_val) + min_val
                values.append(value)

            name = _SENSOR_NAMES.get(sensor_id)
            if name and name in self._sensor_data:
                self._sensor_data[name] = tuple(values)

        # Also update internal position/heading from locator data
        if 'locator' in self._sensor_data:
            self._position = self._sensor_data['locator']

        # Fire registered callbacks
        if self._sensor_callbacks:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            if loop:
                for cb in self._sensor_callbacks:
                    loop.create_task(cb(dict(self._sensor_data)))

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

    async def _connect_ble(self, scan_timeout: float = 600.0):
        """Native BLE connection using bleak (works on Mac M1+).

        Retries scanning every 10 seconds until the RVR+ is found or
        ``scan_timeout`` seconds have elapsed (default 10 minutes).
        """
        try:
            from bleak import BleakScanner, BleakClient
        except ImportError:
            logger.error("bleak is not installed -- run `pip install bleak`")
            raise

        # 1. Scan for the RVR+ with retry loop
        import time as _time
        deadline = _time.monotonic() + scan_timeout
        device = None
        attempt = 0

        while device is None:
            attempt += 1
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                self.state = RoverState.ERROR
                raise RuntimeError(
                    f"BLE: no Sphero RVR+ found after {scan_timeout:.0f}s of scanning"
                )

            logger.info("BLE: scanning for RVR+ devices (attempt %d, %.0fs remaining) ...", attempt, remaining)
            devices = await BleakScanner.discover(timeout=min(10.0, remaining))
            for d in devices:
                name = d.name or ""
                if name.startswith("RV-"):
                    device = d
                    logger.info("BLE: found %s (%s)", d.name, d.address)
                    break

            if device is None:
                wait = min(5.0, remaining)
                if wait > 0:
                    logger.info("BLE: RVR+ not found — retrying in %.0fs (power it on if not already)", wait)
                    await asyncio.sleep(wait)

        # 2. Connect
        logger.info("BLE: connecting to %s ...", device.name)
        self._ble_client = BleakClient(device, timeout=20.0)
        await self._ble_client.connect()
        if not self._ble_client.is_connected:
            self.state = RoverState.ERROR
            raise RuntimeError("BLE: failed to establish connection")
        logger.info("BLE: connected")

        # 3. Cache the API V2 characteristic object (avoids repeated service lookups)
        for service in self._ble_client.services:
            for char in service.characteristics:
                if char.uuid == _API_V2_CHARACTERISTIC:
                    self._api_char = char
                    logger.info("BLE: cached API V2 characteristic (handle %s)", char.handle)
                    break

        # 4. Anti-DOS handshake — try the standard UUID, then the RVR+ alternate
        for antidos_uuid in (_ANTIDOS_CHARACTERISTIC, _ANTIDOS_CHARACTERISTIC_ALT):
            try:
                await self._ble_client.write_gatt_char(
                    antidos_uuid, _ANTIDOS_PAYLOAD, response=True,
                )
                logger.info("BLE: Anti-DOS handshake completed on %s", antidos_uuid[-8:])
                break
            except Exception:
                continue

        # 5. Subscribe to API V2 notifications
        self._proto = _SpheroV2Protocol()
        api_char = self._api_char or _API_V2_CHARACTERISTIC
        await self._ble_client.start_notify(
            api_char, self._ble_notification_handler,
        )
        logger.info("BLE: notifications enabled on API V2 characteristic")

        # 5. Wake
        logger.info("BLE: sending wake command")
        await self._ble_send(_DID_POWER, _CID_WAKE, _TID_NORDIC)
        await asyncio.sleep(2)
        logger.info("RVR+ connected via BLE")

    async def _connect_uart(self):
        """Legacy UART connection via sphero_sdk (Raspberry Pi)."""
        try:
            from sphero_sdk import SpheroRvrAsync, SerialAsyncDal
            self._rvr = SpheroRvrAsync(dal=SerialAsyncDal(port="/dev/ttyTHS1"))
            await self._rvr.wake()
            await asyncio.sleep(2)
            logger.info("RVR+ connected via UART")
        except ImportError:
            logger.warning("sphero_sdk not available, falling back to simulation")
            self.simulate = True
        except Exception as e:
            logger.error("Failed to connect to RVR+ via UART: %s", e)
            self.state = RoverState.ERROR
            raise

    async def disconnect(self):
        """Disconnect from the RVR+."""
        if self._ble_client and self._ble_client.is_connected:
            try:
                # Stop drive before disconnecting
                await self._ble_send_no_response(
                    _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                    data=bytes([0, 0, 0, 0]),
                )
                await self._ble_client.stop_notify(_API_V2_CHARACTERISTIC)
            except Exception as e:
                logger.debug("BLE: cleanup warning: %s", e)
            try:
                await self._ble_client.disconnect()
            except Exception as e:
                logger.debug("BLE: disconnect warning: %s", e)
            self._ble_client = None
            self._proto = None
            logger.info("RVR+ disconnected (BLE)")
        elif self._rvr:
            await self._rvr.close()
            self._rvr = None
            logger.info("RVR+ disconnected (UART)")
        else:
            logger.info("RVR+ disconnected (simulated)")
        self.state = RoverState.IDLE

    async def drive_to(self, waypoint: Waypoint):
        """Navigate to a waypoint."""
        self.state = RoverState.NAVIGATING
        logger.info("Navigating to station %s (%s) at (%.1f, %.1f)",
                    waypoint.station_id, waypoint.name, waypoint.x, waypoint.y)

        dx = waypoint.x - self._position[0]
        dy = waypoint.y - self._position[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        target_heading = math.degrees(math.atan2(dy, dx))
        heading_int = int(target_heading) % 360

        if self.simulate:
            travel_time = distance / (self.speed * 2.0)  # rough estimate
            await asyncio.sleep(min(travel_time, 2.0))  # cap sim time
            self._position = (waypoint.x, waypoint.y)
            self._heading = waypoint.heading
            logger.info("Arrived at %s (simulated)", waypoint.station_id)
        elif self._ble_client:
            speed_byte = int(self.speed * 255) & 0xFF
            heading_msb = (heading_int >> 8) & 0xFF
            heading_lsb = heading_int & 0xFF
            drive_flags = 0  # forward

            # Start driving
            await self._ble_send(
                _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                data=bytes([speed_byte, heading_msb, heading_lsb, drive_flags]),
            )
            # Wait proportional to distance
            distance_cm = distance * 100
            await asyncio.sleep(distance_cm / 50.0)
            # Stop
            await self._ble_send(
                _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                data=bytes([0, heading_msb, heading_lsb, 0]),
            )
            self._position = (waypoint.x, waypoint.y)
            self._heading = float(heading_int)
            logger.info("Arrived at %s (BLE)", waypoint.station_id)
        elif self._rvr:
            distance_cm = distance * 100
            await self._rvr.drive_with_heading(
                speed=int(self.speed * 255),
                heading=heading_int,
                flags=0,
            )
            await asyncio.sleep(distance_cm / 50.0)
            await self._rvr.drive_with_heading(speed=0, heading=heading_int, flags=0)
            self._position = (waypoint.x, waypoint.y)
            self._heading = float(heading_int)
            logger.info("Arrived at %s (UART)", waypoint.station_id)

        self.state = RoverState.DWELLING

    # -- sensor streaming ------------------------------------------------------

    async def start_sensor_streaming(self, period_ms: int = 100):
        """Configure and start sensor streaming on the ST processor.

        Slot 1 (token 0): accelerometer + gyroscope, 32-bit
        Slot 2 (token 1): locator + velocity, 32-bit

        ``period_ms`` sets the streaming interval in milliseconds.
        """
        if self.simulate:
            self._streaming = True
            logger.info("Sensor streaming started (simulated, period=%dms)", period_ms)
            return

        if not self._ble_client:
            logger.warning("start_sensor_streaming: no BLE connection")
            return

        # Stop any existing streaming first
        await self.stop_sensor_streaming()

        # -- Slot 1 (token=0): accelerometer (0x0002) + gyroscope (0x0004) --
        token_0 = 0
        slot1_data = bytes([
            token_0,
            (_SENSOR_ACCELEROMETER >> 8) & 0xFF, _SENSOR_ACCELEROMETER & 0xFF, _DATA_SIZE_32BIT,
            (_SENSOR_GYROSCOPE >> 8) & 0xFF, _SENSOR_GYROSCOPE & 0xFF, _DATA_SIZE_32BIT,
        ])
        await self._ble_send(
            _DID_SENSOR, _CID_CONFIGURE_STREAMING, _TID_ST, data=slot1_data,
        )
        self._streaming_slots[token_0] = [_SENSOR_ACCELEROMETER, _SENSOR_GYROSCOPE]

        # -- Slot 2 (token=1): locator (0x0006) + velocity (0x0007) --
        token_1 = 1
        slot2_data = bytes([
            token_1,
            (_SENSOR_LOCATOR >> 8) & 0xFF, _SENSOR_LOCATOR & 0xFF, _DATA_SIZE_32BIT,
            (_SENSOR_VELOCITY >> 8) & 0xFF, _SENSOR_VELOCITY & 0xFF, _DATA_SIZE_32BIT,
        ])
        await self._ble_send(
            _DID_SENSOR, _CID_CONFIGURE_STREAMING, _TID_ST, data=slot2_data,
        )
        self._streaming_slots[token_1] = [_SENSOR_LOCATOR, _SENSOR_VELOCITY]

        # -- Start streaming --
        period_hi = (period_ms >> 8) & 0xFF
        period_lo = period_ms & 0xFF
        await self._ble_send(
            _DID_SENSOR, _CID_START_STREAMING, _TID_ST,
            data=bytes([period_hi, period_lo]),
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

        try:
            await self._ble_send(
                _DID_SENSOR, _CID_STOP_STREAMING, _TID_ST, timeout=2.0,
            )
        except Exception as e:
            logger.debug("stop_streaming warning: %s", e)
        try:
            await self._ble_send(
                _DID_SENSOR, _CID_CLEAR_STREAMING, _TID_ST, timeout=2.0,
            )
        except Exception as e:
            logger.debug("clear_streaming warning: %s", e)

        self._streaming = False
        self._streaming_slots.clear()
        logger.info("Sensor streaming stopped")

    async def reset_locator(self):
        """Reset the locator X and Y coordinates to zero."""
        logger.info("Resetting locator origin")
        if self.simulate:
            self._sensor_data['locator'] = (0.0, 0.0)
            self._position = (0.0, 0.0)
            return
        if self._ble_client:
            await self._ble_send(
                _DID_SENSOR, _CID_RESET_LOCATOR, _TID_ST,
            )
            self._sensor_data['locator'] = (0.0, 0.0)
            self._position = (0.0, 0.0)

    def add_sensor_callback(self, callback):
        """Register an async callback that fires on each sensor data update.

        The callback receives a dict with the latest sensor values:
        ``{'locator': (x,y), 'velocity': (vx,vy), 'accelerometer': (ax,ay,az), 'gyroscope': (gx,gy,gz)}``
        """
        self._sensor_callbacks.append(callback)

    @property
    def sensor_data(self) -> dict:
        """Return the latest sensor data dict."""
        return self._sensor_data

    # -- LED control ----------------------------------------------------------

    async def set_leds(self, r: int, g: int, b: int):
        """Set all LEDs to the given RGB colour (0-255 per channel).

        Uses a 32-bit bitmask of 0x3FFFFFFF to address every LED on the rover,
        followed by the RGB value for each of the 10 LED groups (headlights,
        brakelights, etc.) -- we send the same colour for all of them.
        """
        r, g, b = (max(0, min(255, v)) for v in (r, g, b))
        logger.info("Setting LEDs to RGB(%d, %d, %d)", r, g, b)

        if self.simulate:
            logger.info("LEDs set (simulated)")
            return

        if self._ble_client:
            # The LED-set command expects: 4-byte LED bitmask + 3 bytes (R,G,B)
            # per active LED group.  The bitmask 0x3FFFFFFF covers all 10 groups
            # (30 bits), so we need 10 * 3 = 30 colour bytes.
            # 8-bit mask: each bit = one LED channel. RVR+ channel layout:
            #   bit 0-2: right headlight R,G,B
            #   bit 3-5: left headlight R,G,B
            #   bit 6-7: left status indicator R,G
            # One byte of value per set bit. Two writes for full coverage.
            # Write 1: headlights (mask 0x3F = 6 channels)
            led_data = bytes([0x3F, r, g, b, r, g, b])
            await self._ble_send_no_response(
                _DID_LEDS, _CID_SET_LEDS_8, _TID_NORDIC,
                data=led_data,
            )
            await asyncio.sleep(0.075)
            # Write 2: status indicators (mask 0xC0 = 2 channels)
            led_data2 = bytes([0xC0, r, g])
            await self._ble_send_no_response(
                _DID_LEDS, _CID_SET_LEDS_8, _TID_NORDIC,
                data=led_data2,
            )
            await asyncio.sleep(0.075)
            logger.info("LEDs set (BLE)")
        elif self._rvr:
            await self._rvr.led_control.set_all_leds_rgb(r, g, b)
            logger.info("LEDs set (UART)")

    # -- battery ------------------------------------------------------------

    async def get_battery(self) -> Optional[int]:
        """Return battery percentage (0-100), or None if unavailable."""
        if self.simulate:
            logger.info("Battery: 100%% (simulated)")
            return 100

        if self._ble_client:
            resp = await self._ble_send(
                _DID_POWER, _CID_BATTERY_PCT, _TID_NORDIC,
            )
            if resp:
                pct = resp[0] if len(resp) >= 1 else None
                logger.info("Battery: %s%%", pct)
                return pct
            logger.warning("Battery query returned no data")
            return None
        elif self._rvr:
            # sphero_sdk uses a callback pattern -- not easily awaitable
            logger.warning("get_battery() not implemented for UART mode")
            return None

    # -- low-level motor helpers --------------------------------------------

    async def reset_yaw(self):
        """Reset the yaw angle to zero (current heading becomes 0)."""
        logger.info("Resetting yaw")
        if self.simulate:
            self._heading = 0.0
            return
        if self._ble_client:
            await self._ble_send(_DID_DRIVE, _CID_RESET_YAW, _TID_ST)
            self._heading = 0.0
        elif self._rvr:
            await self._rvr.reset_yaw()
            self._heading = 0.0

    async def set_raw_motors(
        self, left_mode: int, left_speed: int, right_mode: int, right_speed: int,
    ):
        """Set raw motor speeds.  Modes: 0=off, 1=forward, 2=reverse."""
        logger.info("Raw motors: L(%d,%d) R(%d,%d)", left_mode, left_speed, right_mode, right_speed)
        if self.simulate:
            return
        if self._ble_client:
            await self._ble_send(
                _DID_DRIVE, _CID_RAW_MOTORS, _TID_ST,
                data=bytes([
                    left_mode & 0xFF, left_speed & 0xFF,
                    right_mode & 0xFF, right_speed & 0xFF,
                ]),
            )
        elif self._rvr:
            await self._rvr.raw_motors(
                left_mode=left_mode, left_speed=left_speed,
                right_mode=right_mode, right_speed=right_speed,
            )

    async def stop(self):
        """Immediately stop all motors."""
        logger.info("Stopping rover")
        if self.simulate:
            return
        if self._ble_client:
            heading_int = int(self._heading) % 360
            await self._ble_send(
                _DID_DRIVE, _CID_DRIVE_WITH_HEADING, _TID_ST,
                data=bytes([0, (heading_int >> 8) & 0xFF, heading_int & 0xFF, 0]),
            )
        elif self._rvr:
            await self._rvr.drive_with_heading(speed=0, heading=int(self._heading) % 360, flags=0)

    # -- navigation ---------------------------------------------------------

    async def return_home(self):
        """Return to origin position."""
        self.state = RoverState.RETURNING
        home = Waypoint(station_id="HOME", x=0.0, y=0.0, heading=0.0, name="Home Base")
        await self.drive_to(home)
        self.state = RoverState.IDLE

    @property
    def position(self) -> tuple[float, float]:
        return self._position

    @property
    def heading(self) -> float:
        return self._heading
