// Arduino implementation of the Hal. This is the only file in the library
// that includes Arduino.h, which is what keeps the core buildable — and
// unit-testable — on the host.
#pragma once

#if defined(ARDUINO)

#include <Arduino.h>

#include "redrover/Hal.h"

namespace redrover {

class ArduinoHal : public Hal {
public:
    void configureOutput(uint8_t pin) override { pinMode(pin, OUTPUT); }
    void configureInput(uint8_t pin, bool pullup) override {
        pinMode(pin, pullup ? INPUT_PULLUP : INPUT);
    }

    void writeDigital(uint8_t pin, bool high) override {
        digitalWrite(pin, high ? HIGH : LOW);
    }
    bool readDigital(uint8_t pin) override { return digitalRead(pin) == HIGH; }

    void writePwm(uint8_t pin, uint8_t duty) override {
#if defined(ESP32)
        // The ESP32 core routes analogWrite through LEDC from 3.x; older
        // cores need an explicit channel, so fall back to a bit-banged
        // full-on/full-off when LEDC is unavailable.
#if defined(analogWrite) || ESP_ARDUINO_VERSION_MAJOR >= 3
        analogWrite(pin, duty);
#else
        digitalWrite(pin, duty > 127 ? HIGH : LOW);
#endif
#else
        analogWrite(pin, duty);
#endif
    }

    uint16_t readAnalog(uint8_t pin) override { return analogRead(pin); }

    uint16_t adcMax() const override {
#if defined(ESP32) || defined(ARDUINO_ARCH_RP2040) || defined(TEENSYDUINO)
        return 4095;  // 12-bit by default on these cores
#else
        return 1023;  // 10-bit AVR / SAMD default
#endif
    }

    uint32_t millis() override { return ::millis(); }
    uint32_t micros() override { return ::micros(); }
    void delayMicros(uint16_t us) override { delayMicroseconds(us); }

    uint32_t pulseInMicros(uint8_t pin, bool level, uint32_t timeoutUs) override {
        return pulseIn(pin, level ? HIGH : LOW, timeoutUs);
    }

    void enterCritical() override {
        savedInterrupts_ = interruptsEnabled();
        noInterrupts();
    }

    void exitCritical() override {
        // Restore rather than unconditionally re-enabling: nesting, or a call
        // from inside an ISR, must not turn interrupts on early.
        if (savedInterrupts_) {
            interrupts();
        }
    }

private:
    static bool interruptsEnabled() {
#if defined(__AVR__)
        return (SREG & 0x80) != 0;
#elif defined(__arm__)
        return (__get_PRIMASK() & 1) == 0;
#else
        return true;
#endif
    }

    bool savedInterrupts_ = true;
};

}  // namespace redrover

#endif  // ARDUINO
