// Hardware abstraction: the one place the core talks to a pin or a clock.
//
// Every driver takes a Hal reference rather than calling digitalWrite
// directly. That is what lets the same DifferentialDrive run on an Uno, an
// ESP32, and a host-side unit test with a FakeHal that records the calls.
#pragma once

#include <stdint.h>

namespace redrover {

class Hal {
public:
    virtual ~Hal() = default;

    virtual void configureOutput(uint8_t pin) = 0;
    virtual void configureInput(uint8_t pin, bool pullup) = 0;

    virtual void writeDigital(uint8_t pin, bool high) = 0;
    virtual bool readDigital(uint8_t pin) = 0;

    // Duty cycle 0-255, mapped onto whatever resolution the board provides.
    virtual void writePwm(uint8_t pin, uint8_t duty) = 0;
    // Raw ADC counts; resolution reported by adcMax().
    virtual uint16_t readAnalog(uint8_t pin) = 0;
    virtual uint16_t adcMax() const = 0;

    virtual uint32_t millis() = 0;
    virtual uint32_t micros() = 0;
    virtual void delayMicros(uint16_t us) = 0;

    // Blocking pulse measurement, used by ultrasonic rangefinders.
    // Returns 0 on timeout.
    virtual uint32_t pulseInMicros(uint8_t pin, bool level, uint32_t timeoutUs) = 0;
};

}  // namespace redrover
