#include "redrover/drive/MecanumDrive.h"

namespace redrover {

namespace {
constexpr int32_t kMradScale = 1000;  // v[mm/s] = omega[mrad/s] * r[mm] / 1000

int16_t clampToInt16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}
}  // namespace

int16_t MecanumDrive::toDuty(int16_t wheelMmPerS) const {
    if (wheelMmPerS == 0 || geometry_.maxWheelMmPerS == 0) {
        return 0;
    }
    int32_t duty = (static_cast<int32_t>(wheelMmPerS) * 255) / geometry_.maxWheelMmPerS;
    if (duty > 255) duty = 255;
    if (duty < -255) duty = -255;
    if (geometry_.minEffectiveDuty > 0) {
        if (duty > 0 && duty < geometry_.minEffectiveDuty) {
            duty = geometry_.minEffectiveDuty;
        } else if (duty < 0 && duty > -static_cast<int32_t>(geometry_.minEffectiveDuty)) {
            duty = -static_cast<int32_t>(geometry_.minEffectiveDuty);
        }
    }
    return static_cast<int16_t>(duty);
}

void MecanumDrive::drive(const DriveCommand& cmd) {
    // Standard mecanum inverse kinematics:
    //   fl = vx - vy - omega*(L+W)/2
    //   fr = vx + vy + omega*(L+W)/2
    //   rl = vx + vy - omega*(L+W)/2
    //   rr = vx - vy + omega*(L+W)/2
    const int32_t halfSum =
        (static_cast<int32_t>(geometry_.trackWidthMm) +
         static_cast<int32_t>(geometry_.wheelBaseMm)) / 2;
    const int32_t rot = (static_cast<int32_t>(cmd.angularMradPerS) * halfSum) /
                        kMradScale;
    const int32_t vx = cmd.linearMmPerS;
    const int32_t vy = cmd.lateralMmPerS;

    int32_t wheels[4] = {
        vx - vy - rot,  // front left
        vx + vy + rot,  // front right
        vx + vy - rot,  // rear left
        vx - vy + rot,  // rear right
    };

    // Uniform scaling preserves the commanded direction when a wheel saturates.
    int32_t peak = 0;
    for (int32_t w : wheels) {
        const int32_t magnitude = w >= 0 ? w : -w;
        if (magnitude > peak) {
            peak = magnitude;
        }
    }
    const int32_t maxWheel = geometry_.maxWheelMmPerS;
    if (maxWheel > 0 && peak > maxWheel) {
        for (int32_t& w : wheels) {
            w = (w * maxWheel) / peak;
        }
    }

    for (uint8_t i = 0; i < 4; ++i) {
        last_[i] = clampToInt16(wheels[i]);
    }
    fl_.setDuty(hal_, toDuty(last_[0]));
    fr_.setDuty(hal_, toDuty(last_[1]));
    rl_.setDuty(hal_, toDuty(last_[2]));
    rr_.setDuty(hal_, toDuty(last_[3]));
}

void MecanumDrive::stop() {
    fl_.setDuty(hal_, 0);
    fr_.setDuty(hal_, 0);
    rl_.setDuty(hal_, 0);
    rr_.setDuty(hal_, 0);
    for (int16_t& w : last_) {
        w = 0;
    }
}

}  // namespace redrover
