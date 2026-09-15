#include "redrover/drive/DifferentialDrive.h"

namespace redrover {

namespace {
// v[mm/s] = omega[mrad/s] * r[mm] / 1000. Working in radians means the
// conversion is a single divide with no pi approximation to get wrong.
constexpr int32_t kMradScale = 1000;

int16_t clampToInt16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}
}  // namespace

int16_t DifferentialDrive::angularLimitMradPerS() const {
    // The fastest spin-in-place: both wheels at full speed in opposition, so
    // omega = 2 * v_wheel / track.
    const int32_t track = geometry_.trackWidthMm > 0 ? geometry_.trackWidthMm : 1;
    const int32_t limit =
        (2 * static_cast<int32_t>(geometry_.maxWheelMmPerS) * kMradScale) / track;
    return clampToInt16(limit);
}

int16_t DifferentialDrive::toDuty(int16_t wheelMmPerS) const {
    if (wheelMmPerS == 0 || geometry_.maxWheelMmPerS == 0) {
        return 0;
    }
    int32_t duty = (static_cast<int32_t>(wheelMmPerS) * 255) / geometry_.maxWheelMmPerS;
    if (duty > 255) duty = 255;
    if (duty < -255) duty = -255;

    // Lift a command out of the dead band instead of stalling the motor.
    if (geometry_.minEffectiveDuty > 0) {
        if (duty > 0 && duty < geometry_.minEffectiveDuty) {
            duty = geometry_.minEffectiveDuty;
        } else if (duty < 0 && duty > -static_cast<int32_t>(geometry_.minEffectiveDuty)) {
            duty = -static_cast<int32_t>(geometry_.minEffectiveDuty);
        }
    }
    return static_cast<int16_t>(duty);
}

void DifferentialDrive::drive(const DriveCommand& cmd) {
    // v_wheel = v ± omega * (track / 2)
    const int32_t half = static_cast<int32_t>(geometry_.trackWidthMm) / 2;
    const int32_t differential =
        (static_cast<int32_t>(cmd.angularMradPerS) * half) / kMradScale;

    int32_t leftMmPerS = static_cast<int32_t>(cmd.linearMmPerS) - differential;
    int32_t rightMmPerS = static_cast<int32_t>(cmd.linearMmPerS) + differential;

    // Scale both wheels together when either saturates, so the robot follows
    // the requested arc more slowly rather than veering off it.
    const int32_t maxWheel = geometry_.maxWheelMmPerS;
    int32_t peak = leftMmPerS >= 0 ? leftMmPerS : -leftMmPerS;
    const int32_t rightAbs = rightMmPerS >= 0 ? rightMmPerS : -rightMmPerS;
    if (rightAbs > peak) {
        peak = rightAbs;
    }
    if (maxWheel > 0 && peak > maxWheel) {
        leftMmPerS = (leftMmPerS * maxWheel) / peak;
        rightMmPerS = (rightMmPerS * maxWheel) / peak;
    }

    lastLeft_ = clampToInt16(leftMmPerS);
    lastRight_ = clampToInt16(rightMmPerS);

    left_.setDuty(hal_, toDuty(lastLeft_));
    right_.setDuty(hal_, toDuty(lastRight_));
}

void DifferentialDrive::driveRaw(int16_t leftPerMil, int16_t rightPerMil) {
    if (leftPerMil > 1000) leftPerMil = 1000;
    if (leftPerMil < -1000) leftPerMil = -1000;
    if (rightPerMil > 1000) rightPerMil = 1000;
    if (rightPerMil < -1000) rightPerMil = -1000;

    lastLeft_ = clampToInt16((static_cast<int32_t>(leftPerMil) *
                              geometry_.maxWheelMmPerS) / 1000);
    lastRight_ = clampToInt16((static_cast<int32_t>(rightPerMil) *
                               geometry_.maxWheelMmPerS) / 1000);

    left_.setDuty(hal_, static_cast<int16_t>((static_cast<int32_t>(leftPerMil) * 255) / 1000));
    right_.setDuty(hal_, static_cast<int16_t>((static_cast<int32_t>(rightPerMil) * 255) / 1000));
}

void DifferentialDrive::stop() {
    left_.setDuty(hal_, 0);
    right_.setDuty(hal_, 0);
    lastLeft_ = 0;
    lastRight_ = 0;
}

}  // namespace redrover
