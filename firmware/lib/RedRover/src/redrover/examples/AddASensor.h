// Example: adding a sensor the library does not ship.
//
// A driver is one class with two methods. Everything else — the id, the wire
// format, the host-side plumbing, the streaming cadence — follows from the
// descriptor you fill in, so the host needs no changes at all.
#pragma once

#if defined(ARDUINO)

#include <Wire.h>
#include <redRover.h>

namespace {

// --- the entire driver ----------------------------------------------------
// An MLX90614 non-contact thermometer on I2C. Twenty lines.
class Mlx90614 : public redrover::SensorBase {
public:
    explicit Mlx90614(uint8_t address = 0x5A) : address_(address) {
        descriptor_.kind = redrover::SensorKind::Temperature;
        descriptor_.unit = redrover::Unit::Celsius;
        descriptor_.channels = 1;
        descriptor_.scaleExp = -2;  // values are centidegrees
        descriptor_.rateHz = 4;
        redrover::setName(descriptor_.name, "object_temp");
    }

    bool begin() override {
        Wire.begin();
        return true;
    }

    bool read(redrover::SensorSample& out) override {
        Wire.beginTransmission(address_);
        Wire.write(0x07);  // object temperature register
        if (Wire.endTransmission(false) != 0) {
            return false;  // no ACK: report nothing rather than a stale value
        }
        if (Wire.requestFrom(address_, static_cast<uint8_t>(3)) != 3) {
            return false;
        }
        const uint16_t raw = Wire.read() | (static_cast<uint16_t>(Wire.read()) << 8);
        Wire.read();  // PEC, ignored
        // 0.02 K per count, minus 273.15 K, reported in centidegrees Celsius.
        out.channels = 1;
        out.values[0] = static_cast<int32_t>(raw) * 2 - 27315;
        return true;
    }

private:
    uint8_t address_;
};

// --- wiring it up ---------------------------------------------------------
redrover::ArduinoHal hal;
redrover::SerialTransport transport(Serial);
redrover::DifferentialDrive drive(hal, redrover::MotorChannel(5, 4, 7),
                                  redrover::MotorChannel(6, 8, 9));

Mlx90614 objectTemp;                                        // the new sensor
redrover::AnalogSensor ambient(hal, A1, "ambient_temp",     // and a stock one
                               redrover::SensorKind::Temperature,
                               redrover::Unit::Celsius, 500, 1024, -2, 1);

redrover::SensorRegistry sensors;
redrover::Node node(transport, drive, sensors, hal);

}  // namespace

void setup() {
    Serial.begin(115200);

    sensors.add(&objectTemp);  // <- that is the whole integration step
    sensors.add(&ambient);

    redrover::NodeConfig config;
    redrover::setName(config.board, "thermal-rover");
    node.configure(config);
    node.begin();
}

void loop() {
    node.spin();
}

#endif  // ARDUINO
