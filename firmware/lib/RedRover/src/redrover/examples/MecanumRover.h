// Example: a holonomic mecanum rover with wheel encoders and a high-rate
// analog accelerometer — the configuration that can actually do bearing
// analysis, because the ADC samples at kHz rather than the tens of Hz a BLE
// toy rover streams at.
#pragma once

#if defined(ARDUINO)

#include <redRover.h>

namespace {

redrover::ArduinoHal hal;
redrover::SerialTransport transport(Serial);

// trackWidthMm, wheelBaseMm, maxWheelMmPerS, minEffectiveDuty.
redrover::MecanumGeometry geometry(220, 180, 600, 35);

redrover::MecanumDrive drive(
    hal,
    redrover::MotorChannel(3, 22, 23),   // front left
    redrover::MotorChannel(5, 24, 25),   // front right
    redrover::MotorChannel(6, 26, 27),   // rear left
    redrover::MotorChannel(9, 28, 29),   // rear right
    geometry);

// ADXL335 on a 10-bit ADC: 330 mV/g over a 5 V reference is ~68 counts/g.
redrover::AnalogAccelCalibration accelCal(512, 68);
redrover::AnalogAccelerometer accel(hal, A0, A1, A2, accelCal, "vibration", 1000);

redrover::QuadratureEncoder leftEncoder(hal, 18, 19, "left_encoder");
redrover::QuadratureEncoder rightEncoder(hal, 20, 21, "right_encoder");

redrover::SensorRegistry sensors;
redrover::Node node(transport, drive, sensors, hal);

// Aux channel 0 drives a status LED; the host sets it from the diagnosis.
void handleAux(uint8_t channel, int16_t value, void*) {
    if (channel == 0) {
        hal.writePwm(13, static_cast<uint8_t>(value > 255 ? 255 : (value < 0 ? 0 : value)));
    }
}

}  // namespace

void setup() {
    Serial.begin(500000);  // kHz-rate streaming needs the bandwidth

    sensors.add(&accel);
    sensors.add(&leftEncoder);
    sensors.add(&rightEncoder);

    redrover::NodeConfig config;
    config.commandTimeoutMs = 300;
    config.statusIntervalMs = 500;
    redrover::setName(config.board, "mecanum-rover");
    node.configure(config);
    node.setAuxHandler(handleAux);

    hal.configureOutput(13);
    node.begin();

    // Re-zero the accelerometer while the robot is still stationary.
    accel.calibrateZero();
}

void loop() {
    node.spin();
}

#endif  // ARDUINO
