// USB / UART transport over any Arduino Stream.
#pragma once

#if defined(ARDUINO)

#include <Arduino.h>

#include "redrover/ITransport.h"

namespace redrover {

class SerialTransport : public ITransport {
public:
    explicit SerialTransport(Stream& stream) : stream_(stream) {}

    bool begin() override { return true; }

    size_t available() override {
        const int n = stream_.available();
        return n > 0 ? static_cast<size_t>(n) : 0;
    }

    size_t read(uint8_t* buffer, size_t len) override {
        size_t count = 0;
        while (count < len && stream_.available() > 0) {
            const int byte = stream_.read();
            if (byte < 0) {
                break;
            }
            buffer[count++] = static_cast<uint8_t>(byte);
        }
        return count;
    }

    size_t write(const uint8_t* buffer, size_t len) override {
        return stream_.write(buffer, len);
    }

private:
    Stream& stream_;
};

}  // namespace redrover

#endif  // ARDUINO
