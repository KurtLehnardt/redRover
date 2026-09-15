// Example: a two-motor rover with a bumper, an ultrasonic rangefinder, and a
// battery monitor, talking to the host over USB serial.
//
// This is the minimum viable redRover node: about forty lines, of which the
// sensor list is four.
#pragma once

#if defined(ARDUINO)

#include <redRover.h>

namespace {

// --- pins (change these to match your wiring) ----------------------------
constexpr uint8_t kLeftPwm = 5, kLeftDirA = 4, kLeftDirB = 7;
constexpr uint8_t kRightPwm = 6, kRightDirA = 8, kRightDirB = 9;
constexpr uint8_t kBumperPin = 2;
constexpr uint8_t kTrigPin = 10, kEchoPin = 11;
constexpr uint8_t kBatteryPin = A0;

redrover::ArduinoHal hal;
redrover::SerialTransport transport(Serial);

redrover::DifferentialGeometry geometry = {
    /*trackWidthMm=*/150,
    // Measure this: full duty for five seconds, distance / 5.
    /*maxWheelMmPerS=*/450,
    /*minEffectiveDuty=*/40,
};

redrover::DifferentialDrive drive(
    hal,
    redrover::MotorChannel(kLeftPwm, kLeftDirA, kLeftDirB),
    redrover::MotorChannel(kRightPwm, kRightDirA, kRightDirB),
    geometry);

// --- sensors: one line each ----------------------------------------------
redrover::DigitalSensor bumper(hal, kBumperPin, "front_bumper");
redrover::UltrasonicHCSR04 range(hal, kTrigPin, kEchoPin, "front_range");
// A 1:11 divider off a 12 V pack: 1024 counts = 5 V at the pin = 55 V at the
// pack, so millivolts = raw * 55000 / 1024.
redrover::AnalogSensor battery(hal, kBatteryPin, "battery", redrover::SensorKind::Battery,
                               redrover::Unit::Volt, 55000, 1024, -3, 1);

redrover::SensorRegistry sensors;
redrover::Node node(transport, drive, sensors, hal);

}  // namespace

void setup() {
    Serial.begin(115200);

    sensors.add(&bumper);
    sensors.add(&range);
    sensors.add(&battery);

    redrover::NodeConfig config;
    config.commandTimeoutMs = 500;  // stop if the host goes quiet
    redrover::setName(config.board, "basic-rover");
    node.configure(config);

    node.begin();
}

void loop() {
    node.spin();
}

#endif  // ARDUINO
