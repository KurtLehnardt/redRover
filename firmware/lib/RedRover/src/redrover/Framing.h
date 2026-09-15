// Wire framing: COBS + CRC-16/CCITT-FALSE.
//
// COBS is used instead of escape bytes because it gives a guaranteed
// worst-case overhead (1 byte per 254) and a delimiter (0x00) that cannot
// appear inside a frame, so a receiver that joins mid-stream resynchronises on
// the very next delimiter. The previous BLE code hand-rolled escaping and had
// to reassemble fragments twice.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace redrover {

uint16_t crc16(const uint8_t* data, size_t len);

// Encode `len` bytes into `out`, which must hold at least cobsMaxEncoded(len).
// Returns the encoded length, excluding the trailing delimiter.
size_t cobsEncode(const uint8_t* src, size_t len, uint8_t* out);

// Decode a delimiter-free COBS frame. Returns the decoded length, or 0 if the
// frame is malformed.
size_t cobsDecode(const uint8_t* src, size_t len, uint8_t* out, size_t outCapacity);

constexpr size_t cobsMaxEncoded(size_t len) { return len + (len / 254) + 2; }

}  // namespace redrover
