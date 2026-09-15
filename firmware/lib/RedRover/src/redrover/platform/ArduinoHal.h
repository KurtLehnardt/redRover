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
        // Depth-counted rather than probing a hardware flag: CMSIS intrinsics
        // like __get_PRIMASK are not declared on every ARM Arduino core
        // (Teensy's, for one), and a portability layer cannot depend on them.
        if (depth_++ == 0) {
#if defined(__AVR__)
            // Saving SREG restores the interrupt bit exactly, so this stays
            // correct even if it is ever reached with interrupts already off.
            savedStatus_ = SREG;
#endif
            noInterrupts();
        }
    }

    void exitCritical() override {
        if (depth_ == 0) {
            return;  // unbalanced exit; never turn interrupts on speculatively
        }
        if (--depth_ == 0) {
#if defined(__AVR__)
            SREG = savedStatus_;
#else
            interrupts();
#endif
        }
    }

private:
    uint8_t depth_ = 0;
#if defined(__AVR__)
    uint8_t savedStatus_ = 0;
#endif
};

}  // namespace redrover

#endif  // ARDUINO
