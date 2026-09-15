// Generic single-channel analog sensor.
//
// This is the "twenty lines" path: most add-on sensors are an ADC pin plus a
// linear conversion, and this covers them without writing a driver at all.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ISensor.h"

namespace redrover {

class AnalogSensor : public SensorBase {
public:
    // `scaleNum/scaleDen` converts raw ADC counts into the reported unit,
    // then the descriptor's scaleExp gives the decimal point. Integer maths
    // throughout keeps this usable on an AVR.
    AnalogSensor(Hal& hal, uint8_t pin, const char* name, SensorKind kind, Unit unit,
                 int32_t scaleNum = 1, int32_t scaleDen = 1, int8_t scaleExp = 0,
                 uint16_t rateHz = 10, int32_t offset = 0)
        : hal_(hal), pin_(pin), scaleNum_(scaleNum),
          scaleDen_(scaleDen == 0 ? 1 : scaleDen), offset_(offset) {
        descriptor_.kind = kind;
        descriptor_.unit = unit;
        descriptor_.channels = 1;
        descriptor_.scaleExp = scaleExp;
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool read(SensorSample& out) override {
        const int32_t raw = static_cast<int32_t>(hal_.readAnalog(pin_));
        out.channels = 1;
        out.values[0] = (raw * scaleNum_) / scaleDen_ + offset_;
        return true;
    }

private:
    Hal& hal_;
    uint8_t pin_;
    int32_t scaleNum_;
    int32_t scaleDen_;
    int32_t offset_;
};

}  // namespace redrover
