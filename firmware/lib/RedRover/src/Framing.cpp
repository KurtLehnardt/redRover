#include "redrover/Framing.h"

namespace redrover {

uint16_t crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (uint8_t bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

size_t cobsEncode(const uint8_t* src, size_t len, uint8_t* out) {
    size_t readIndex = 0;
    size_t writeIndex = 1;
    size_t codeIndex = 0;
    uint8_t code = 1;

    while (readIndex < len) {
        if (src[readIndex] == 0) {
            out[codeIndex] = code;
            code = 1;
            codeIndex = writeIndex++;
            ++readIndex;
            continue;
        }
        out[writeIndex++] = src[readIndex++];
        ++code;
        if (code == 0xFF) {
            out[codeIndex] = code;
            code = 1;
            codeIndex = writeIndex++;
        }
    }
    out[codeIndex] = code;
    return writeIndex;
}

size_t cobsDecode(const uint8_t* src, size_t len, uint8_t* out, size_t outCapacity) {
    size_t readIndex = 0;
    size_t writeIndex = 0;

    while (readIndex < len) {
        const uint8_t code = src[readIndex];
        if (code == 0) {
            return 0;  // a zero code byte cannot occur inside a valid frame
        }
        ++readIndex;
        if (readIndex + code - 1 > len) {
            return 0;  // run overruns the frame
        }
        for (uint8_t i = 1; i < code; ++i) {
            if (writeIndex >= outCapacity) {
                return 0;
            }
            out[writeIndex++] = src[readIndex++];
        }
        if (code != 0xFF && readIndex < len) {
            if (writeIndex >= outCapacity) {
                return 0;
            }
            out[writeIndex++] = 0;
        }
    }
    return writeIndex;
}

}  // namespace redrover
