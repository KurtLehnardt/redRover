// Sensor driver behaviour, including the concurrency guard that a host build
// can exercise but a board cannot easily be made to fail on demand.
#include <unity.h>

#include "redrover/sensors/AnalogSensor.h"
#include "redrover/sensors/AnalogAccelerometer.h"
#include "redrover/sensors/DigitalSensor.h"
#include "redrover/sensors/QuadratureEncoder.h"
#include "redrover/sensors/UltrasonicHCSR04.h"
#include "redrover/testing/FakeHal.h"

using namespace redrover;
using redrover::testing::FakeHal;

void setUp() {}
void tearDown() {}

// --- encoder --------------------------------------------------------------

void test_encoder_read_is_guarded_against_interrupts() {
    // Regression: read() copied a volatile int32_t directly. On AVR that is
    // four byte-loads, and an ISR landing between them returns a count that
    // was never real.
    FakeHal hal;
    QuadratureEncoder encoder(hal, 2, 3);
    encoder.begin();

    hal.criticalEntries = 0;
    SensorSample sample;
    TEST_ASSERT_TRUE(encoder.read(sample));
    TEST_ASSERT_TRUE(hal.criticalEntries > 0);
    // And the guard must be balanced, or interrupts stay off forever.
    TEST_ASSERT_EQUAL_INT(0, hal.criticalDepth);
}

void test_encoder_reset_is_guarded_too() {
    FakeHal hal;
    QuadratureEncoder encoder(hal, 2, 3);
    encoder.begin();
    hal.criticalEntries = 0;
    encoder.reset();
    TEST_ASSERT_TRUE(hal.criticalEntries > 0);
    TEST_ASSERT_EQUAL_INT(0, hal.criticalDepth);
}

void test_encoder_counts_forward_and_backward() {
    FakeHal hal;
    QuadratureEncoder encoder(hal, 2, 3);
    encoder.begin();

    // Gray sequence 00 -> 10 -> 11 -> 01 -> 00 is one full forward cycle.
    const bool forward[4][2] = {{true, false}, {true, true}, {false, true}, {false, false}};
    for (const auto& step : forward) {
        hal.digital[2] = step[0];
        hal.digital[3] = step[1];
        encoder.poll(0);
    }
    TEST_ASSERT_EQUAL_INT32(4, encoder.count());

    // Reversing the sequence must unwind it exactly.
    const bool backward[4][2] = {{false, true}, {true, true}, {true, false}, {false, false}};
    for (const auto& step : backward) {
        hal.digital[2] = step[0];
        hal.digital[3] = step[1];
        encoder.poll(0);
    }
    TEST_ASSERT_EQUAL_INT32(0, encoder.count());
}

// --- ultrasonic -----------------------------------------------------------

void test_ultrasonic_converts_echo_time_to_millimetres() {
    FakeHal hal;
    UltrasonicHCSR04 range(hal, 9, 10);
    range.begin();

    // 1 m out and back at ~343 m/s is about 5831 us.
    hal.pulseUs = 5831;
    SensorSample sample;
    TEST_ASSERT_TRUE(range.read(sample));
    TEST_ASSERT_INT_WITHIN(20, 1000, sample.values[0]);
}

void test_ultrasonic_reports_no_sample_on_timeout() {
    // "No echo" means out of range, which is not the same as zero distance.
    // Returning 0 would put a phantom obstacle against the bumper.
    FakeHal hal;
    UltrasonicHCSR04 range(hal, 9, 10);
    range.begin();

    hal.pulseUs = 0;
    SensorSample sample;
    TEST_ASSERT_FALSE(range.read(sample));
}

void test_ultrasonic_rejects_readings_beyond_max_range() {
    FakeHal hal;
    UltrasonicHCSR04 range(hal, 9, 10, "front", 10, /*maxRangeMm=*/1000);
    range.begin();

    hal.pulseUs = 40000;  // ~6.8 m, well past the configured limit
    SensorSample sample;
    TEST_ASSERT_FALSE(range.read(sample));
}

// --- analog ---------------------------------------------------------------

void test_analog_sensor_applies_its_scale() {
    FakeHal hal;
    // A 1:11 divider: 1024 counts == 55000 mV.
    AnalogSensor battery(hal, 0, "battery", SensorKind::Battery, Unit::Volt,
                         55000, 1024, -3, 1);
    hal.analog[0] = 512;

    SensorSample sample;
    TEST_ASSERT_TRUE(battery.read(sample));
    TEST_ASSERT_EQUAL_UINT8(1, sample.channels);
    TEST_ASSERT_INT_WITHIN(50, 27500, sample.values[0]);  // ~27.5 V in mV
}

void test_digital_sensor_honours_active_low() {
    FakeHal hal;
    DigitalSensor bumper(hal, 4, "bumper", SensorKind::Bumper, /*activeLow=*/true);
    bumper.begin();

    SensorSample sample;
    hal.digital[4] = true;   // pulled up: not pressed
    TEST_ASSERT_TRUE(bumper.read(sample));
    TEST_ASSERT_EQUAL_INT32(0, sample.values[0]);

    hal.digital[4] = false;  // shorted to ground: pressed
    TEST_ASSERT_TRUE(bumper.read(sample));
    TEST_ASSERT_EQUAL_INT32(1, sample.values[0]);
}

void test_accelerometer_converts_counts_to_milli_g() {
    FakeHal hal;
    AnalogAccelCalibration cal(512, 68);  // ADXL335 on a 10-bit ADC
    AnalogAccelerometer accel(hal, 0, 1, 2, cal);

    hal.analog[0] = 512;        // 0 g
    hal.analog[1] = 512 + 68;   // +1 g
    hal.analog[2] = 512 - 34;   // -0.5 g

    SensorSample sample;
    TEST_ASSERT_TRUE(accel.read(sample));
    TEST_ASSERT_EQUAL_UINT8(3, sample.channels);
    TEST_ASSERT_INT_WITHIN(5, 0, sample.values[0]);
    TEST_ASSERT_INT_WITHIN(5, 1000, sample.values[1]);
    TEST_ASSERT_INT_WITHIN(15, -500, sample.values[2]);
}

void test_accelerometer_descriptor_advertises_its_rate() {
    // The host decides whether bearing analysis is possible from this number.
    FakeHal hal;
    AnalogAccelCalibration cal(512, 68);
    AnalogAccelerometer accel(hal, 0, 1, 2, cal, "vibration", 2000);

    TEST_ASSERT_EQUAL_UINT16(2000, accel.descriptor().rateHz);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)SensorKind::Acceleration,
                            (uint8_t)accel.descriptor().kind);
    TEST_ASSERT_EQUAL_INT(-3, accel.descriptor().scaleExp);
    TEST_ASSERT_EQUAL_STRING("vibration", accel.descriptor().name);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_encoder_read_is_guarded_against_interrupts);
    RUN_TEST(test_encoder_reset_is_guarded_too);
    RUN_TEST(test_encoder_counts_forward_and_backward);
    RUN_TEST(test_ultrasonic_converts_echo_time_to_millimetres);
    RUN_TEST(test_ultrasonic_reports_no_sample_on_timeout);
    RUN_TEST(test_ultrasonic_rejects_readings_beyond_max_range);
    RUN_TEST(test_analog_sensor_applies_its_scale);
    RUN_TEST(test_digital_sensor_honours_active_low);
    RUN_TEST(test_accelerometer_converts_counts_to_milli_g);
    RUN_TEST(test_accelerometer_descriptor_advertises_its_rate);
    return UNITY_END();
}
