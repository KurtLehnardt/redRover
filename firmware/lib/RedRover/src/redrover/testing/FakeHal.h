// Host-side doubles, so the kinematics and the protocol can be tested without
// a board attached. Included by the native test environment.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ITransport.h"

namespace redrover {
namespace testing {

class FakeHal : public Hal {
public:
    static constexpr uint8_t kPins = 64;

    void configureOutput(uint8_t pin) override { mode[pin] = 1; }
    void configureInput(uint8_t pin, bool pullup) override { mode[pin] = pullup ? 3 : 2; }

    void writeDigital(uint8_t pin, bool high) override { digital[pin] = high; }
    bool readDigital(uint8_t pin) override { return digital[pin]; }

    void writePwm(uint8_t pin, uint8_t duty) override { pwm[pin] = duty; }
    uint16_t readAnalog(uint8_t pin) override { return analog[pin]; }
    uint16_t adcMax() const override { return 1023; }

    uint32_t millis() override { return nowMs; }
    uint32_t micros() override { return nowMs * 1000; }
    void delayMicros(uint16_t us) override { nowMs += (us + 999) / 1000; }

    uint32_t pulseInMicros(uint8_t, bool, uint32_t) override { return pulseUs; }

    void advance(uint32_t ms) { nowMs += ms; }

    uint8_t mode[kPins] = {0};
    bool digital[kPins] = {false};
    uint8_t pwm[kPins] = {0};
    uint16_t analog[kPins] = {0};
    uint32_t nowMs = 0;
    uint32_t pulseUs = 0;
};

// In-memory transport: what the node writes lands in `out`, and whatever is
// pushed with push() is what it reads.
class LoopbackTransport : public ITransport {
public:
    static constexpr size_t kCapacity = 2048;

    bool begin() override { return true; }

    size_t available() override { return inLen - inPos; }

    size_t read(uint8_t* buffer, size_t len) override {
        size_t count = 0;
        while (count < len && inPos < inLen) {
            buffer[count++] = in[inPos++];
        }
        return count;
    }

    size_t write(const uint8_t* buffer, size_t len) override {
        size_t count = 0;
        while (count < len && outLen < kCapacity) {
            out[outLen++] = buffer[count++];
        }
        return count;
    }

    void push(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len && inLen < kCapacity; ++i) {
            in[inLen++] = data[i];
        }
    }

    void clearOut() { outLen = 0; }

    uint8_t in[kCapacity] = {0};
    size_t inLen = 0;
    size_t inPos = 0;
    uint8_t out[kCapacity] = {0};
    size_t outLen = 0;
};

}  // namespace testing
}  // namespace redrover
