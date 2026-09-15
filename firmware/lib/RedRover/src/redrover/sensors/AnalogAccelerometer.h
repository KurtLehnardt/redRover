// Three-axis analog accelerometer (ADXL335, ADXL1002, MMA7361, ...).
//
// This is the sensor that makes real vibration analysis possible: an ADC read
// per axis costs microseconds, so the node can stream at kHz rates, which is
// what bearing envelope analysis needs and what a BLE rover IMU cannot give.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ISensor.h"

namespace redrover {

struct AnalogAccelCalibration {
    // ADC counts at 0 g on each axis (typically mid-scale).
    uint16_t zeroG[3] = {512, 512, 512};
    // Sensitivity in ADC counts per g.
    uint16_t countsPerG[3] = {102, 102, 102};
};

class AnalogAccelerometer : public SensorBase {
public:
    AnalogAccelerometer(Hal& hal, uint8_t pinX, uint8_t pinY, uint8_t pinZ,
                        const AnalogAccelCalibration& calibration,
                        const char* name = "accel", uint16_t rateHz = 1000)
        : hal_(hal), pins_{pinX, pinY, pinZ}, cal_(calibration) {
        descriptor_.kind = SensorKind::Acceleration;
        descriptor_.unit = Unit::MetrePerSecond2;
        descriptor_.channels = 3;
        descriptor_.scaleExp = -3;  // values are milli-g
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool read(SensorSample& out) override {
        out.channels = 3;
        for (uint8_t axis = 0; axis < 3; ++axis) {
            const int32_t raw = static_cast<int32_t>(hal_.readAnalog(pins_[axis]));
            const int32_t counts = raw - static_cast<int32_t>(cal_.zeroG[axis]);
            const int32_t perG = cal_.countsPerG[axis] != 0 ? cal_.countsPerG[axis] : 1;
            out.values[axis] = (counts * 1000) / perG;  // milli-g
        }
        return true;
    }

    // Re-zero against the current reading. Call with the robot stationary and
    // level; the axis pointing down keeps its 1 g offset.
    void calibrateZero(uint8_t downAxis = 2) {
        for (uint8_t axis = 0; axis < 3; ++axis) {
            const uint16_t raw = hal_.readAnalog(pins_[axis]);
            cal_.zeroG[axis] = (axis == downAxis)
                                   ? static_cast<uint16_t>(raw + cal_.countsPerG[axis])
                                   : raw;
        }
    }

private:
    Hal& hal_;
    uint8_t pins_[3];
    AnalogAccelCalibration cal_;
};

}  // namespace redrover
