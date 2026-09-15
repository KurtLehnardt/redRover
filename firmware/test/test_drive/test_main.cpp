// Kinematics and motor-mapping tests. These run on the host, so a wiring
// mistake or a sign error is caught before anything is on wheels.
#include <unity.h>

#include "redrover/drive/DifferentialDrive.h"
#include "redrover/drive/MecanumDrive.h"
#include "redrover/testing/FakeHal.h"

using namespace redrover;
using redrover::testing::FakeHal;

void setUp() {}
void tearDown() {}

static DifferentialGeometry geometry() {
    DifferentialGeometry g;
    g.trackWidthMm = 200;
    g.maxWheelMmPerS = 500;
    g.minEffectiveDuty = 0;
    return g;
}

void test_straight_line_drives_both_wheels_equally() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 250;
    drive.drive(cmd);

    TEST_ASSERT_EQUAL_INT16(250, drive.lastLeftMmPerS());
    TEST_ASSERT_EQUAL_INT16(250, drive.lastRightMmPerS());
    // 250 of 500 mm/s is half duty.
    TEST_ASSERT_INT_WITHIN(2, 127, hal.pwm[5]);
    TEST_ASSERT_INT_WITHIN(2, 127, hal.pwm[6]);
    TEST_ASSERT_TRUE(hal.digital[4]);   // left forward
    TEST_ASSERT_FALSE(hal.digital[7]);
}

void test_reverse_flips_direction_pins() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = -250;
    drive.drive(cmd);

    TEST_ASSERT_FALSE(hal.digital[4]);
    TEST_ASSERT_TRUE(hal.digital[7]);
    TEST_ASSERT_INT_WITHIN(2, 127, hal.pwm[5]);
}

void test_spin_in_place_opposes_the_wheels() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    DriveCommand cmd;
    cmd.angularMradPerS = 1571;  // 90 deg/s counter-clockwise
    drive.drive(cmd);

    // omega * track/2 = 1.571 rad/s * 100 mm = 157 mm/s per wheel.
    TEST_ASSERT_INT_WITHIN(5, -157, drive.lastLeftMmPerS());
    TEST_ASSERT_INT_WITHIN(5, 157, drive.lastRightMmPerS());
    TEST_ASSERT_TRUE(drive.lastLeftMmPerS() < 0);
    TEST_ASSERT_TRUE(drive.lastRightMmPerS() > 0);
}

void test_saturation_scales_both_wheels_together() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 500;       // already at the limit
    cmd.angularMradPerS = 1571;  // ...and asked to turn as well
    drive.drive(cmd);

    // Clamping each wheel independently would straighten the arc out; uniform
    // scaling keeps the requested curvature and just goes slower.
    TEST_ASSERT_TRUE(drive.lastRightMmPerS() <= 500);
    TEST_ASSERT_TRUE(drive.lastLeftMmPerS() <= 500);
    TEST_ASSERT_EQUAL_INT16(500, drive.lastRightMmPerS());

    const int32_t before = 500 + 157;
    const int32_t left = drive.lastLeftMmPerS();
    // ratio preserved: left/right == (v - d)/(v + d)
    TEST_ASSERT_INT_WITHIN(10, (int)((500 - 157) * 500 / before), (int)left);
}

void test_stop_zeroes_pwm() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 400;
    drive.drive(cmd);
    TEST_ASSERT_TRUE(hal.pwm[5] > 0);

    drive.stop();
    TEST_ASSERT_EQUAL_UINT8(0, hal.pwm[5]);
    TEST_ASSERT_EQUAL_UINT8(0, hal.pwm[6]);
    TEST_ASSERT_EQUAL_INT16(0, drive.lastLeftMmPerS());
    // Both direction pins low at rest: the bridge coasts rather than braking.
    TEST_ASSERT_FALSE(hal.digital[4]);
    TEST_ASSERT_FALSE(hal.digital[7]);
}

void test_dead_band_lifts_tiny_commands() {
    FakeHal hal;
    DifferentialGeometry g = geometry();
    g.minEffectiveDuty = 60;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), g);
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 10;  // ~5 duty: enough to buzz, not to move
    drive.drive(cmd);

    TEST_ASSERT_EQUAL_UINT8(60, hal.pwm[5]);
}

void test_drive_raw_maps_per_mil_to_duty() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    drive.begin();

    drive.driveRaw(1000, -500);
    TEST_ASSERT_EQUAL_UINT8(255, hal.pwm[5]);
    TEST_ASSERT_INT_WITHIN(2, 127, hal.pwm[6]);
    TEST_ASSERT_FALSE(hal.digital[8]);  // right reversed
}

void test_capabilities_report_the_real_angular_limit() {
    FakeHal hal;
    DifferentialDrive drive(hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9), geometry());
    // 500 mm/s wheels on a 200 mm track spin at 2*0.5/0.2 = 5 rad/s.
    TEST_ASSERT_INT_WITHIN(20, 5000, drive.capabilities().maxAngularMradPerS);
    TEST_ASSERT_FALSE(drive.capabilities().holonomic);
}

// --- mecanum --------------------------------------------------------------

void test_mecanum_strafes_sideways() {
    FakeHal hal;
    MecanumGeometry g;
    g.trackWidthMm = 200;
    g.wheelBaseMm = 200;
    g.maxWheelMmPerS = 500;
    MecanumDrive drive(hal, MotorChannel(3, 22, 23), MotorChannel(5, 24, 25),
                       MotorChannel(6, 26, 27), MotorChannel(9, 28, 29), g);
    drive.begin();

    DriveCommand cmd;
    cmd.lateralMmPerS = 200;
    drive.drive(cmd);

    const int16_t* w = drive.lastWheelsMmPerS();
    // Strafing right: front-left and rear-right roll backwards, the diagonal
    // pair forwards.
    TEST_ASSERT_EQUAL_INT16(-200, w[0]);
    TEST_ASSERT_EQUAL_INT16(200, w[1]);
    TEST_ASSERT_EQUAL_INT16(200, w[2]);
    TEST_ASSERT_EQUAL_INT16(-200, w[3]);
    TEST_ASSERT_TRUE(drive.capabilities().holonomic);
}

void test_mecanum_forward_drives_all_wheels_alike() {
    FakeHal hal;
    MecanumDrive drive(hal, MotorChannel(3, 22, 23), MotorChannel(5, 24, 25),
                       MotorChannel(6, 26, 27), MotorChannel(9, 28, 29));
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 300;
    drive.drive(cmd);

    const int16_t* w = drive.lastWheelsMmPerS();
    for (uint8_t i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_INT16(300, w[i]);
    }
}

void test_mecanum_scales_uniformly_when_saturated() {
    FakeHal hal;
    MecanumGeometry g;
    g.maxWheelMmPerS = 300;
    MecanumDrive drive(hal, MotorChannel(3, 22, 23), MotorChannel(5, 24, 25),
                       MotorChannel(6, 26, 27), MotorChannel(9, 28, 29), g);
    drive.begin();

    DriveCommand cmd;
    cmd.linearMmPerS = 300;
    cmd.lateralMmPerS = 300;
    drive.drive(cmd);

    const int16_t* w = drive.lastWheelsMmPerS();
    for (uint8_t i = 0; i < 4; ++i) {
        TEST_ASSERT_TRUE(w[i] <= 300 && w[i] >= -300);
    }
    // The diagonal that carries vx+vy saturates; the other collapses to zero,
    // preserving the 45-degree direction.
    TEST_ASSERT_EQUAL_INT16(300, w[1]);
    TEST_ASSERT_EQUAL_INT16(0, w[0]);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_straight_line_drives_both_wheels_equally);
    RUN_TEST(test_reverse_flips_direction_pins);
    RUN_TEST(test_spin_in_place_opposes_the_wheels);
    RUN_TEST(test_saturation_scales_both_wheels_together);
    RUN_TEST(test_stop_zeroes_pwm);
    RUN_TEST(test_dead_band_lifts_tiny_commands);
    RUN_TEST(test_drive_raw_maps_per_mil_to_duty);
    RUN_TEST(test_capabilities_report_the_real_angular_limit);
    RUN_TEST(test_mecanum_strafes_sideways);
    RUN_TEST(test_mecanum_forward_drives_all_wheels_alike);
    RUN_TEST(test_mecanum_scales_uniformly_when_saturated);
    return UNITY_END();
}
