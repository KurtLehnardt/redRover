// Chassis abstraction. Differential, mecanum, and steered bases all accept the
// same DriveCommand, so swapping a chassis never changes the host.
#pragma once

#include "Types.h"

namespace redrover {

struct DriveCapabilities {
    bool holonomic = false;          // can it translate sideways?
    uint16_t maxLinearMmPerS = 500;
    uint16_t maxAngularMradPerS = 5000;  // ~286 deg/s
    bool hasEncoders = false;
};

class IDriveBase {
public:
    virtual ~IDriveBase() = default;

    virtual bool begin() = 0;

    // Apply a velocity command. Implementations clamp to their capabilities
    // rather than saturating silently at the motor driver.
    virtual void drive(const DriveCommand& cmd) = 0;

    // Cut motor output immediately. Must be safe to call from any state,
    // including before begin() and repeatedly.
    virtual void stop() = 0;

    virtual const DriveCapabilities& capabilities() const = 0;

    // Direct per-wheel control in per-mil (-1000..1000), for calibration and
    // for hosts that want to close their own loop.
    virtual void driveRaw(int16_t /*leftPerMil*/, int16_t /*rightPerMil*/) {}
};

}  // namespace redrover
