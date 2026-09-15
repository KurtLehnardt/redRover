#include "redrover/Protocol.h"

#include "redrover/Framing.h"

namespace redrover {

size_t encodeFrame(const Frame& frame, uint8_t* out, size_t outCapacity) {
    if (frame.len > kMaxPayload) {
        return 0;
    }

    uint8_t raw[kMaxFrame];
    const size_t bodyLen = kHeaderLen + frame.len;
    raw[0] = kProtocolVersion;
    raw[1] = static_cast<uint8_t>(frame.type);
    raw[2] = frame.seq;
    raw[3] = frame.len;
    for (uint8_t i = 0; i < frame.len; ++i) {
        raw[kHeaderLen + i] = frame.payload[i];
    }

    const uint16_t crc = crc16(raw, bodyLen);
    putU16(&raw[bodyLen], crc);
    const size_t totalLen = bodyLen + kCrcLen;

    if (outCapacity < cobsMaxEncoded(totalLen)) {
        return 0;
    }
    const size_t encoded = cobsEncode(raw, totalLen, out);
    out[encoded] = 0x00;  // frame delimiter
    return encoded + 1;
}

bool decodeFrame(const uint8_t* src, size_t len, Frame& out) {
    uint8_t raw[kMaxFrame];
    const size_t decoded = cobsDecode(src, len, raw, sizeof(raw));
    if (decoded < kHeaderLen + kCrcLen) {
        return false;
    }
    if (raw[0] != kProtocolVersion) {
        return false;
    }

    const uint8_t payloadLen = raw[3];
    const size_t expected = kHeaderLen + payloadLen + kCrcLen;
    if (expected != decoded || payloadLen > kMaxPayload) {
        return false;
    }

    const uint16_t wantCrc = getU16(&raw[kHeaderLen + payloadLen]);
    if (crc16(raw, kHeaderLen + payloadLen) != wantCrc) {
        return false;
    }

    out.type = static_cast<MsgType>(raw[1]);
    out.seq = raw[2];
    out.len = payloadLen;
    for (uint8_t i = 0; i < payloadLen; ++i) {
        out.payload[i] = raw[kHeaderLen + i];
    }
    return true;
}

}  // namespace redrover
