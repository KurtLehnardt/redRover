// Two-wheel differential (tank) chassis — the default for Arduino robots.
#pragma once

#include "redrover/IDriveBase.h"
#include "redrover/Hal.h"
#include "MotorChannel.h"

namespace redrover {

struct DifferentialGeometry {
    // Distance between the wheel contact patches, in millimetres.
    uint16_t trackWidthMm = 150;
    // Ground speed of one wheel at full PWM, in mm/s. Measure it: run one
    // wheel at full duty for five seconds and divide the distance by five.
    // Every velocity command is scaled by this, so a guess here is a
    // systematic error in everything downstream.
    uint16_t maxWheelMmPerS = 500;
    // Duty below which the motors buzz but the robot does not move. Commands
    // between 1 and this value are lifted to it rather than being sent as a
    // stall.
    uint8_t minEffectiveDuty = 0;
};

class DifferentialDrive : public IDriveBase {
public:
    DifferentialDrive(Hal& hal, MotorChannel left, MotorChannel right,
                      const DifferentialGeometry& geometry = DifferentialGeometry())
        : hal_(hal), left_(left), right_(right), geometry_(geometry) {
        caps_.holonomic = false;
        caps_.maxLinearMmPerS = geometry.maxWheelMmPerS;
        caps_.maxAngularMradPerS = angularLimitMradPerS();
    }

    bool begin() override {
        left_.begin(hal_);
        right_.begin(hal_);
        return true;
    }

    void drive(const DriveCommand& cmd) override;
    void stop() override;
    void driveRaw(int16_t leftPerMil, int16_t rightPerMil) override;

    const DriveCapabilities& capabilities() const override { return caps_; }

    // Wheel speeds the last command resolved to, in mm/s. Exposed for tests
    // and for hosts that want to check the kinematics.
    int16_t lastLeftMmPerS() const { return lastLeft_; }
    int16_t lastRightMmPerS() const { return lastRight_; }

private:
    int16_t angularLimitMradPerS() const;
    int16_t toDuty(int16_t wheelMmPerS) const;

    Hal& hal_;
    MotorChannel left_;
    MotorChannel right_;
    DifferentialGeometry geometry_;
    DriveCapabilities caps_;
    int16_t lastLeft_ = 0;
    int16_t lastRight_ = 0;
};

}  // namespace redrover
