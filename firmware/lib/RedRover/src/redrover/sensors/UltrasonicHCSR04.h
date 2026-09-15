// HC-SR04 / HY-SRF05 ultrasonic rangefinder.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ISensor.h"

namespace redrover {

class UltrasonicHCSR04 : public SensorBase {
public:
    UltrasonicHCSR04(Hal& hal, uint8_t trigPin, uint8_t echoPin,
                     const char* name = "ultrasonic", uint16_t rateHz = 10,
                     uint16_t maxRangeMm = 4000)
        : hal_(hal), trig_(trigPin), echo_(echoPin), maxRangeMm_(maxRangeMm) {
        descriptor_.kind = SensorKind::Distance;
        descriptor_.unit = Unit::Metre;
        descriptor_.channels = 1;
        descriptor_.scaleExp = -3;  // values are millimetres
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool begin() override {
        hal_.configureOutput(trig_);
        hal_.configureInput(echo_, false);
        hal_.writeDigital(trig_, false);
        return true;
    }

    bool read(SensorSample& out) override {
        hal_.writeDigital(trig_, false);
        hal_.delayMicros(3);
        hal_.writeDigital(trig_, true);
        hal_.delayMicros(10);
        hal_.writeDigital(trig_, false);

        // Round trip at ~343 m/s: timeout is the far-range flight time plus
        // margin, so a missing echo costs a bounded wait instead of blocking.
        const uint32_t timeoutUs = (static_cast<uint32_t>(maxRangeMm_) * 2 * 1000) / 343 + 2000;
        const uint32_t echoUs = hal_.pulseInMicros(echo_, true, timeoutUs);
        if (echoUs == 0) {
            // No echo is "out of range", which is genuinely different from
            // "zero distance"; report no sample rather than a false obstacle
            // directly in front of the robot.
            return false;
        }

        const uint32_t mm = (echoUs * 343) / 2000;  // us -> mm
        if (mm > maxRangeMm_) {
            return false;
        }
        out.channels = 1;
        out.values[0] = static_cast<int32_t>(mm);
        return true;
    }

private:
    Hal& hal_;
    uint8_t trig_;
    uint8_t echo_;
    uint16_t maxRangeMm_;
};

}  // namespace redrover
