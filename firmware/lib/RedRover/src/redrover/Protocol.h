// redRover wire protocol, version 1.
//
// Frame layout, before COBS encoding:
//
//     +-----+------+-----+-----+----------+---------+
//     | ver | type | seq | len | payload  | crc16   |
//     |  1  |  1   |  1  |  1  |   len    |    2    |
//     +-----+------+-----+-----+----------+---------+
//
// The CRC covers ver..payload. Multi-byte fields are little-endian, which
// matches every target in platformio.ini and avoids a byte-swap on the hot
// path.
#pragma once

#include "Types.h"

namespace redrover {

enum class MsgType : uint8_t {
    // Host -> device
    Hello = 0x01,
    Describe = 0x02,
    SetStream = 0x03,
    Drive = 0x04,
    DriveRaw = 0x05,
    Stop = 0x06,
    EStop = 0x07,
    SetAux = 0x08,
    Ping = 0x09,
    ReadOnce = 0x0A,

    // Device -> host
    HelloAck = 0x81,
    Descriptor = 0x82,
    Sample = 0x83,
    Status = 0x85,
    Log = 0x86,
    Ack = 0x87,
};

enum class AckCode : uint8_t {
    Ok = 0,
    UnknownType = 1,
    BadLength = 2,
    Rejected = 3,
    NotSupported = 4,
    EStopActive = 5,
};

enum class LogLevel : uint8_t { Debug = 0, Info = 1, Warn = 2, Error = 3 };

constexpr uint8_t kHeaderLen = 4;
constexpr uint8_t kCrcLen = 2;
constexpr uint8_t kMaxPayload = kMaxFrame - kHeaderLen - kCrcLen;

struct Frame {
    MsgType type = MsgType::Ping;
    uint8_t seq = 0;
    uint8_t len = 0;
    uint8_t payload[kMaxPayload] = {0};
};

// Serialise `frame` into `out` (COBS-encoded, delimiter appended).
// Returns bytes written, or 0 if `out` is too small.
size_t encodeFrame(const Frame& frame, uint8_t* out, size_t outCapacity);

// Parse one delimiter-free COBS frame. Returns true on success; false on a
// CRC mismatch, a version mismatch, or a malformed frame.
bool decodeFrame(const uint8_t* src, size_t len, Frame& out);

// --- payload helpers (little-endian, no unaligned access) -----------------

inline void putU16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
}
inline void putU32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}
inline void putI16(uint8_t* p, int16_t v) { putU16(p, static_cast<uint16_t>(v)); }
inline void putI32(uint8_t* p, int32_t v) { putU32(p, static_cast<uint32_t>(v)); }

inline uint16_t getU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t getU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline int16_t getI16(const uint8_t* p) { return static_cast<int16_t>(getU16(p)); }
inline int32_t getI32(const uint8_t* p) { return static_cast<int32_t>(getU32(p)); }

}  // namespace redrover
