"""Cross-implementation tests for the firmware wire protocol.

The golden vectors in ``tests/data/wire_vectors.json`` are produced by the C++
implementation (``firmware/tools/gen_vectors.cpp``). Decoding them here proves
the two implementations agree byte for byte; re-encoding proves the agreement
runs both ways. A change to either side that breaks compatibility fails these
tests instead of failing on a robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rover import wire
from src.rover.wire import (
    Ack,
    AckCode,
    Frame,
    FrameReader,
    HelloAck,
    LogMessage,
    MsgType,
    NodeState,
    ProtocolError,
    Sample,
    SensorDescriptor,
    SensorKind,
    Status,
    Unit,
    cobs_decode,
    cobs_encode,
    crc16,
)

VECTORS_PATH = Path(__file__).parent / "data" / "wire_vectors.json"


@pytest.fixture(scope="module")
def vectors() -> dict:
    return json.loads(VECTORS_PATH.read_text())


def test_protocol_versions_match(vectors):
    assert vectors["protocol_version"] == wire.PROTOCOL_VERSION


def test_every_golden_frame_decodes(vectors):
    for case in vectors["frames"]:
        encoded = bytes.fromhex(case["wire"])
        assert encoded[-1] == wire.DELIMITER, case["name"]
        frame = Frame.decode(encoded[:-1])
        assert frame.type == case["type"], case["name"]
        assert frame.seq == case["seq"], case["name"]
        assert frame.payload.hex() == case["payload"], case["name"]


def test_every_golden_frame_re_encodes_identically(vectors):
    """Python must produce the exact bytes the firmware produces."""
    for case in vectors["frames"]:
        frame = Frame(
            type=MsgType(case["type"]),
            seq=case["seq"],
            payload=bytes.fromhex(case["payload"]),
        )
        assert frame.encode().hex() == case["wire"], case["name"]


def test_golden_frames_cover_the_message_types_that_matter(vectors):
    seen = {case["type"] for case in vectors["frames"]}
    for required in (MsgType.DRIVE, MsgType.SAMPLE, MsgType.DESCRIPTOR,
                     MsgType.STATUS, MsgType.ACK, MsgType.HELLO_ACK):
        assert int(required) in seen


# === COBS and CRC ===


def test_crc16_known_vector():
    assert crc16(b"123456789") == 0x29B1


@pytest.mark.parametrize(
    "data",
    [b"", b"\x01\x02\x03", b"\x00", b"\x00" * 40, bytes(range(1, 256)), bytes(300)],
)
def test_cobs_round_trip(data):
    encoded = cobs_encode(data)
    assert 0 not in encoded  # the delimiter can never appear inside a frame
    assert cobs_decode(encoded) == data


def test_cobs_rejects_zero_code_byte():
    with pytest.raises(ProtocolError):
        cobs_decode(b"\x00\x01")


def test_cobs_rejects_overrunning_run():
    with pytest.raises(ProtocolError):
        cobs_decode(b"\x10\x01\x02")


# === Frames ===


def test_frame_rejects_corrupted_payload():
    """A corrupted frame must be dropped, not acted on.

    A zero-free payload encodes as a single COBS run, so the flipped byte
    survives decoding and the CRC is what rejects it.
    """
    encoded = bytearray(Frame(MsgType.LOG, 1, b"abcdef").encode())
    encoded[6] ^= 0x20
    with pytest.raises(ProtocolError, match="CRC"):
        Frame.decode(bytes(encoded[:-1]))


@pytest.mark.parametrize("index", range(1, 12))
def test_any_single_bit_flip_is_rejected(index):
    """Every byte of the frame is covered by either the CRC or a length check."""
    encoded = bytearray(Frame(MsgType.DRIVE, 1, wire.drive_payload(200, 1000)).encode())
    encoded[index] ^= 0x01
    with pytest.raises(ProtocolError):
        Frame.decode(bytes(encoded[:-1]))


def test_frame_rejects_wrong_version():
    raw = bytes([wire.PROTOCOL_VERSION + 1, int(MsgType.PING), 0, 0])
    import struct
    payload = cobs_encode(raw + struct.pack("<H", crc16(raw)))
    with pytest.raises(ProtocolError, match="version"):
        Frame.decode(payload)


def test_frame_rejects_oversized_payload():
    with pytest.raises(ProtocolError, match="exceeds"):
        Frame(MsgType.LOG, 0, b"x" * 300).encode()


# === Streaming reassembly ===


def test_reader_recovers_after_garbage():
    """A receiver that cannot resynchronise is one glitch away from deaf."""
    reader = FrameReader()
    good = Frame(MsgType.DRIVE, 5, wire.drive_payload(150, 0)).encode()

    frames = reader.feed(b"\x05\x01\x02\x00" + good)
    assert len(frames) == 1
    assert frames[0].seq == 5
    assert reader.dropped == 1


def test_reader_handles_split_frames():
    reader = FrameReader()
    encoded = Frame(MsgType.PING, 1).encode()
    assert reader.feed(encoded[:2]) == []
    assert reader.feed(encoded[2:4]) == []
    frames = reader.feed(encoded[4:])
    assert len(frames) == 1
    assert frames[0].type is MsgType.PING


def test_reader_discards_an_oversized_run():
    reader = FrameReader(max_frame=32)
    reader.feed(b"\x01" * 100)
    frames = reader.feed(b"\x00" + Frame(MsgType.PING, 2).encode())
    assert reader.dropped == 1
    assert len(frames) == 1


# === Payload parsers ===


def test_parse_sample(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "sample_accel")
    sample = Sample.parse(bytes.fromhex(case["payload"]))
    assert sample.sensor_id == 2
    assert sample.timestamp_ms == 123456
    assert sample.values == [-1024, 0, 1000]


def test_parse_descriptor_and_scale(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "descriptor_accel")
    descriptor = SensorDescriptor.parse(bytes.fromhex(case["payload"]))
    assert descriptor.id == 2
    assert descriptor.kind is SensorKind.ACCELERATION
    assert descriptor.unit is Unit.METRE_PER_SECOND2
    assert descriptor.channels == 3
    assert descriptor.rate_hz == 1000
    assert descriptor.name == "vibration"
    assert not descriptor.failed
    # scale_exp of -3 means the integers are milli-units.
    assert descriptor.scale == pytest.approx(0.001)
    assert descriptor.to_physical(-1024) == pytest.approx(-1.024)


def test_parse_status(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "status_driving")
    status = Status.parse(bytes.fromhex(case["payload"]))
    assert status.state is NodeState.DRIVING
    assert not status.estop
    assert status.battery_mv == 11700
    assert status.dropped_frames == 4


def test_parse_ack(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "ack_estop_active")
    ack = Ack.parse(bytes.fromhex(case["payload"]))
    assert ack.of_type == int(MsgType.DRIVE)
    assert ack.code is AckCode.ESTOP_ACTIVE
    assert not ack.ok


def test_parse_hello_ack(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "hello_ack")
    hello = HelloAck.parse(bytes.fromhex(case["payload"]))
    assert hello.protocol_version == wire.PROTOCOL_VERSION
    assert hello.sensor_count == 3
    assert not hello.holonomic
    assert hello.max_linear_mm_s == 450
    assert hello.command_timeout_ms == 500
    assert hello.board == "basic-rover"


def test_parse_log(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "log_warn")
    message = LogMessage.parse(bytes.fromhex(case["payload"]))
    assert message.level is wire.LogLevel.WARN
    assert message.text == "command timeout"


# === Angular units ===


def test_drive_payload_uses_milliradians(vectors):
    """Regression: millidegrees in an int16 caps the robot at 32.7 deg/s.

    90 deg/s is 1571 mrad/s and fits comfortably; as millidegrees it would be
    90000 and overflow.
    """
    case = next(c for c in vectors["frames"] if c["name"] == "drive_forward_turn")
    payload = wire.drive_payload(250, 1571, 0)
    assert payload.hex() == case["payload"]

    import struct
    assert struct.unpack("<hhh", payload) == (250, 1571, 0)
    with pytest.raises(struct.error):
        wire.drive_payload(0, 90000)  # millidegrees would not fit


def test_drive_payload_round_trips_negatives(vectors):
    case = next(c for c in vectors["frames"] if c["name"] == "drive_reverse_strafe")
    assert wire.drive_payload(-500, -3000, -120).hex() == case["payload"]
