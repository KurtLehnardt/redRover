// Four-wheel mecanum chassis — holonomic, so it accepts the lateral component
// of a DriveCommand that a differential base must refuse.
#pragma once

#include "redrover/IDriveBase.h"
#include "redrover/Hal.h"
#include "MotorChannel.h"

namespace redrover {

struct MecanumGeometry {
    uint16_t trackWidthMm = 200;   // left-right wheel separation
    uint16_t wheelBaseMm = 200;    // front-rear wheel separation
    uint16_t maxWheelMmPerS = 500;
    uint8_t minEffectiveDuty = 0;
};

class MecanumDrive : public IDriveBase {
public:
    MecanumDrive(Hal& hal, MotorChannel frontLeft, MotorChannel frontRight,
                 MotorChannel rearLeft, MotorChannel rearRight,
                 const MecanumGeometry& geometry = MecanumGeometry())
        : hal_(hal), fl_(frontLeft), fr_(frontRight), rl_(rearLeft), rr_(rearRight),
          geometry_(geometry) {
        caps_.holonomic = true;
        caps_.maxLinearMmPerS = geometry.maxWheelMmPerS;
        caps_.maxAngularMradPerS = 5000;
    }

    bool begin() override {
        fl_.begin(hal_);
        fr_.begin(hal_);
        rl_.begin(hal_);
        rr_.begin(hal_);
        return true;
    }

    void drive(const DriveCommand& cmd) override;
    void stop() override;

    const DriveCapabilities& capabilities() const override { return caps_; }

    // Resolved wheel speeds in mm/s, front-left, front-right, rear-left,
    // rear-right.
    const int16_t* lastWheelsMmPerS() const { return last_; }

private:
    int16_t toDuty(int16_t wheelMmPerS) const;

    Hal& hal_;
    MotorChannel fl_, fr_, rl_, rr_;
    MecanumGeometry geometry_;
    DriveCapabilities caps_;
    int16_t last_[4] = {0, 0, 0, 0};
};

}  // namespace redrover
