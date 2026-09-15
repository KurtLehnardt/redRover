// Byte transport: USB serial, BLE, WiFi/UDP, ESP-NOW, or a test double.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace redrover {

class ITransport {
public:
    virtual ~ITransport() = default;

    virtual bool begin() = 0;

    // Number of bytes available to read without blocking.
    virtual size_t available() = 0;
    // Read up to `len` bytes; returns the count actually read.
    virtual size_t read(uint8_t* buffer, size_t len) = 0;
    // Write `len` bytes; returns the count actually written. A short write is
    // reported rather than silently dropped so the node can count it.
    virtual size_t write(const uint8_t* buffer, size_t len) = 0;

    // True when a peer is present. Transports that cannot tell (plain UART)
    // return true.
    virtual bool connected() { return true; }
};

}  // namespace redrover
