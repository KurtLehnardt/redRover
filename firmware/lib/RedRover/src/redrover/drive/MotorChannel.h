// One H-bridge channel, covering the three wiring styles you actually meet:
//
//   * two direction pins + PWM enable (L298N, L293D, TB6612)
//   * one direction pin + PWM (DRV8871, many hobby drivers)
//   * PWM-only (ESC / servo-style, no reverse)
#pragma once

#include "redrover/Hal.h"

namespace redrover {

constexpr uint8_t kNoPin = 0xFF;

class MotorChannel {
public:
    MotorChannel() = default;
    MotorChannel(uint8_t pwmPin, uint8_t dirA, uint8_t dirB = kNoPin, bool inverted = false)
        : pwmPin_(pwmPin), dirA_(dirA), dirB_(dirB), inverted_(inverted) {}

    void begin(Hal& hal) {
        if (pwmPin_ != kNoPin) hal.configureOutput(pwmPin_);
        if (dirA_ != kNoPin) hal.configureOutput(dirA_);
        if (dirB_ != kNoPin) hal.configureOutput(dirB_);
        setDuty(hal, 0);
    }

    // `duty` is signed: -255..255. Sign selects direction.
    void setDuty(Hal& hal, int16_t duty) {
        if (duty > 255) duty = 255;
        if (duty < -255) duty = -255;
        if (inverted_) duty = static_cast<int16_t>(-duty);

        const bool forward = duty >= 0;
        const uint8_t magnitude = static_cast<uint8_t>(forward ? duty : -duty);

        if (dirA_ != kNoPin && dirB_ != kNoPin) {
            // Two direction pins: drive both low at zero so the bridge coasts
            // rather than braking, which is gentler on the gearbox.
            hal.writeDigital(dirA_, magnitude != 0 && forward);
            hal.writeDigital(dirB_, magnitude != 0 && !forward);
        } else if (dirA_ != kNoPin) {
            hal.writeDigital(dirA_, forward);
        }
        if (pwmPin_ != kNoPin) {
            hal.writePwm(pwmPin_, magnitude);
        }
        lastDuty_ = duty;
    }

    int16_t lastDuty() const { return lastDuty_; }

private:
    uint8_t pwmPin_ = kNoPin;
    uint8_t dirA_ = kNoPin;
    uint8_t dirB_ = kNoPin;
    bool inverted_ = false;
    int16_t lastDuty_ = 0;
};

}  // namespace redrover
