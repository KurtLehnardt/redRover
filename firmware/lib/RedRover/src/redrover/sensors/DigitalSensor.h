// Binary input: bumper, limit switch, line sensor, e-stop button.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ISensor.h"

namespace redrover {

class DigitalSensor : public SensorBase {
public:
    DigitalSensor(Hal& hal, uint8_t pin, const char* name,
                  SensorKind kind = SensorKind::Bumper, bool activeLow = true,
                  bool pullup = true, uint16_t rateHz = 50)
        : hal_(hal), pin_(pin), activeLow_(activeLow), pullup_(pullup) {
        descriptor_.kind = kind;
        descriptor_.unit = Unit::Boolean;
        descriptor_.channels = 1;
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool begin() override {
        hal_.configureInput(pin_, pullup_);
        return true;
    }

    bool read(SensorSample& out) override {
        const bool level = hal_.readDigital(pin_);
        out.channels = 1;
        out.values[0] = (activeLow_ ? !level : level) ? 1 : 0;
        return true;
    }

private:
    Hal& hal_;
    uint8_t pin_;
    bool activeLow_;
    bool pullup_;
};

}  // namespace redrover
