// Sensor registry and node-level behaviour, including the safety properties
// the firmware must hold regardless of what the host does.
#include <unity.h>

#include "redrover/Node.h"
#include "redrover/Protocol.h"
#include "redrover/SensorRegistry.h"
#include "redrover/drive/DifferentialDrive.h"
#include "redrover/testing/FakeHal.h"

using namespace redrover;
using redrover::testing::FakeHal;
using redrover::testing::LoopbackTransport;

void setUp() {}
void tearDown() {}

namespace {

class CountingSensor : public SensorBase {
public:
    CountingSensor(const char* name, uint16_t rateHz, bool startsOk = true,
                   bool producesData = true)
        : startsOk_(startsOk), producesData_(producesData) {
        descriptor_.kind = SensorKind::Generic;
        descriptor_.unit = Unit::Count;
        descriptor_.channels = 1;
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool begin() override { return startsOk_; }

    bool read(SensorSample& out) override {
        ++reads;
        if (!producesData_) {
            return false;
        }
        out.channels = 1;
        out.values[0] = ++value;
        return true;
    }

    int32_t value = 0;
    int reads = 0;

private:
    bool startsOk_;
    bool producesData_;
};

// Helper: send a frame into the node's transport.
void push(LoopbackTransport& transport, MsgType type, const uint8_t* payload,
          uint8_t len, uint8_t seq = 1) {
    Frame frame;
    frame.type = type;
    frame.seq = seq;
    frame.len = len;
    for (uint8_t i = 0; i < len; ++i) frame.payload[i] = payload[i];

    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(frame, wire, sizeof(wire));
    transport.push(wire, n);
}

// Helper: find the first frame of a given type in what the node sent.
bool findSent(LoopbackTransport& transport, MsgType type, Frame& out) {
    size_t start = 0;
    for (size_t i = 0; i < transport.outLen; ++i) {
        if (transport.out[i] != 0x00) {
            continue;
        }
        Frame frame;
        if (i > start && decodeFrame(&transport.out[start], i - start, frame)) {
            if (frame.type == type) {
                out = frame;
                return true;
            }
        }
        start = i + 1;
    }
    return false;
}

}  // namespace

// --- registry -------------------------------------------------------------

void test_registry_assigns_sequential_ids() {
    SensorRegistry registry;
    CountingSensor a("a", 10), b("b", 10);
    TEST_ASSERT_EQUAL_UINT8(0, registry.add(&a));
    TEST_ASSERT_EQUAL_UINT8(1, registry.add(&b));
    TEST_ASSERT_EQUAL_UINT8(2, registry.count());
    TEST_ASSERT_EQUAL_UINT8(0, a.descriptor().id);
    TEST_ASSERT_EQUAL_UINT8(1, b.descriptor().id);
    TEST_ASSERT_TRUE(registry.byId(1) == &b);
}

void test_registry_refuses_null_and_overflow() {
    SensorRegistry registry;
    TEST_ASSERT_EQUAL_UINT8(0xFF, registry.add(nullptr));

    CountingSensor filler("f", 1);
    for (uint8_t i = 0; i < kMaxSensors; ++i) {
        registry.add(&filler);
    }
    TEST_ASSERT_EQUAL_UINT8(0xFF, registry.add(&filler));
}

void test_failed_sensor_is_marked_and_never_sampled() {
    SensorRegistry registry;
    CountingSensor ok("ok", 100);
    CountingSensor broken("broken", 100, /*startsOk=*/false);
    registry.add(&ok);
    registry.add(&broken);

    TEST_ASSERT_EQUAL_UINT8(1, registry.beginAll());
    TEST_ASSERT_FALSE(registry.failed(0));
    TEST_ASSERT_TRUE(registry.failed(1));
    // A sensor that did not start must not be scheduled: publishing zeros for
    // it would be indistinguishable from a real reading.
    TEST_ASSERT_FALSE(registry.due(1, 10000, 0));
}

void test_on_demand_sensor_is_never_streamed() {
    SensorRegistry registry;
    CountingSensor onDemand("manual", 0);
    registry.add(&onDemand);
    registry.beginAll();
    TEST_ASSERT_FALSE(registry.due(0, 100000, 100));
}

void test_rate_limiting_respects_both_native_and_requested_rates() {
    SensorRegistry registry;
    CountingSensor fast("fast", 100);  // 10 ms period
    registry.add(&fast);
    registry.beginAll();

    registry.markSampled(0, 1000);
    TEST_ASSERT_FALSE(registry.due(0, 1005, 0));
    TEST_ASSERT_TRUE(registry.due(0, 1010, 0));

    // A host asking for 10 Hz must not get 100 Hz.
    registry.markSampled(0, 2000);
    TEST_ASSERT_FALSE(registry.due(0, 2050, 10));
    TEST_ASSERT_TRUE(registry.due(0, 2100, 10));
}

void test_rate_limiting_survives_millis_rollover() {
    SensorRegistry registry;
    CountingSensor sensor("s", 10);  // 100 ms
    registry.add(&sensor);
    registry.beginAll();

    // millis() wraps every ~49 days; unsigned subtraction must still give the
    // true elapsed time, or the rover freezes its sensors for a month.
    const uint32_t justBeforeWrap = 0xFFFFFF00u;
    registry.markSampled(0, justBeforeWrap);
    TEST_ASSERT_FALSE(registry.due(0, justBeforeWrap + 50, 0));
    TEST_ASSERT_TRUE(registry.due(0, justBeforeWrap + 150, 0));  // wrapped
}

void test_enabled_mask_selects_sensors() {
    SensorRegistry registry;
    CountingSensor a("a", 100), b("b", 100);
    registry.add(&a);
    registry.add(&b);
    registry.beginAll();

    registry.setEnabledMask(0x02);  // only index 1
    TEST_ASSERT_FALSE(registry.due(0, 10000, 0));
    TEST_ASSERT_TRUE(registry.due(1, 10000, 0));
}

// --- node -----------------------------------------------------------------

struct Rig {
    FakeHal hal;
    LoopbackTransport transport;
    DifferentialDrive drive{hal, MotorChannel(5, 4, 7), MotorChannel(6, 8, 9)};
    SensorRegistry registry;
    Node node{transport, drive, registry, hal};
};

void test_hello_reports_capabilities() {
    Rig rig;
    CountingSensor sensor("s", 10);
    rig.registry.add(&sensor);
    rig.node.begin();
    rig.transport.clearOut();

    push(rig.transport, MsgType::Hello, nullptr, 0);
    rig.node.spin();

    Frame ack;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::HelloAck, ack));
    TEST_ASSERT_EQUAL_UINT8(kProtocolVersion, ack.payload[0]);
    TEST_ASSERT_EQUAL_UINT8(1, ack.payload[1]);  // sensor count
    TEST_ASSERT_EQUAL_UINT8(0, ack.payload[2]);  // not holonomic
}

void test_describe_emits_one_descriptor_per_sensor() {
    Rig rig;
    CountingSensor a("alpha", 10), b("beta", 10);
    rig.registry.add(&a);
    rig.registry.add(&b);
    rig.node.begin();
    rig.transport.clearOut();

    push(rig.transport, MsgType::Describe, nullptr, 0);
    rig.node.spin();

    Frame descriptor;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Descriptor, descriptor));
    TEST_ASSERT_EQUAL_UINT8(0, descriptor.payload[0]);
    TEST_ASSERT_EQUAL_STRING("alpha", (const char*)&descriptor.payload[8]);
}

void test_drive_command_moves_the_wheels() {
    Rig rig;
    rig.node.begin();
    rig.transport.clearOut();

    uint8_t payload[4];
    putI16(&payload[0], 250);
    putI16(&payload[2], 0);
    push(rig.transport, MsgType::Drive, payload, sizeof(payload));
    rig.node.spin();

    TEST_ASSERT_EQUAL_INT16(250, rig.drive.lastLeftMmPerS());
    TEST_ASSERT_EQUAL_UINT8((uint8_t)NodeState::Driving, (uint8_t)rig.node.state());
}

void test_lateral_command_is_refused_by_a_differential_base() {
    Rig rig;
    rig.node.begin();
    rig.transport.clearOut();

    uint8_t payload[6];
    putI16(&payload[0], 0);
    putI16(&payload[2], 0);
    putI16(&payload[4], 200);  // strafe, which this chassis cannot do
    push(rig.transport, MsgType::Drive, payload, sizeof(payload));
    rig.node.spin();

    Frame ack;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Ack, ack));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)AckCode::NotSupported, ack.payload[1]);
    // Refused, not silently reinterpreted as something else.
    TEST_ASSERT_EQUAL_INT16(0, rig.drive.lastLeftMmPerS());
}

void test_estop_latches_and_refuses_drive() {
    Rig rig;
    rig.node.begin();

    const uint8_t engage[1] = {1};
    push(rig.transport, MsgType::EStop, engage, 1);
    rig.node.spin();
    TEST_ASSERT_TRUE(rig.node.emergencyStopped());

    rig.transport.clearOut();
    uint8_t payload[4];
    putI16(&payload[0], 400);
    putI16(&payload[2], 0);
    push(rig.transport, MsgType::Drive, payload, sizeof(payload), 2);
    rig.node.spin();

    Frame ack;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Ack, ack));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)AckCode::EStopActive, ack.payload[1]);
    TEST_ASSERT_EQUAL_INT16(0, rig.drive.lastLeftMmPerS());

    const uint8_t clear[1] = {0};
    push(rig.transport, MsgType::EStop, clear, 1, 3);
    rig.node.spin();
    TEST_ASSERT_FALSE(rig.node.emergencyStopped());
}

void test_watchdog_stops_the_motors_when_the_host_goes_quiet() {
    Rig rig;
    NodeConfig config;
    config.commandTimeoutMs = 500;
    config.statusIntervalMs = 0;
    rig.node.configure(config);
    rig.node.begin();

    uint8_t payload[4];
    putI16(&payload[0], 400);
    putI16(&payload[2], 0);
    push(rig.transport, MsgType::Drive, payload, sizeof(payload));
    rig.node.spin();
    TEST_ASSERT_EQUAL_INT16(400, rig.drive.lastLeftMmPerS());

    // Host falls silent. A robot that keeps its last command is a robot that
    // drives into a wall when the cable is kicked out.
    rig.hal.advance(200);
    rig.node.spin();
    TEST_ASSERT_EQUAL_INT16(400, rig.drive.lastLeftMmPerS());

    rig.hal.advance(400);
    rig.node.spin();
    TEST_ASSERT_EQUAL_INT16(0, rig.drive.lastLeftMmPerS());
    TEST_ASSERT_EQUAL_UINT8(0, rig.hal.pwm[5]);
}

void test_watchdog_is_held_off_by_any_traffic() {
    Rig rig;
    NodeConfig config;
    config.commandTimeoutMs = 500;
    config.statusIntervalMs = 0;
    rig.node.configure(config);
    rig.node.begin();

    uint8_t payload[4];
    putI16(&payload[0], 400);
    putI16(&payload[2], 0);
    push(rig.transport, MsgType::Drive, payload, sizeof(payload));
    rig.node.spin();

    for (uint8_t i = 0; i < 5; ++i) {
        rig.hal.advance(300);
        push(rig.transport, MsgType::Ping, nullptr, 0, (uint8_t)(10 + i));
        rig.node.spin();
        TEST_ASSERT_EQUAL_INT16(400, rig.drive.lastLeftMmPerS());
    }
}

void test_streaming_publishes_samples_at_the_configured_rate() {
    Rig rig;
    CountingSensor sensor("s", 100);  // 10 ms
    rig.registry.add(&sensor);
    NodeConfig config;
    config.statusIntervalMs = 0;
    config.commandTimeoutMs = 0;
    rig.node.configure(config);
    rig.node.begin();
    rig.transport.clearOut();

    for (uint8_t i = 0; i < 3; ++i) {
        rig.hal.advance(10);
        rig.node.spin();
    }

    Frame sample;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Sample, sample));
    TEST_ASSERT_EQUAL_UINT8(0, sample.payload[0]);        // sensor id
    TEST_ASSERT_EQUAL_UINT8(1, sample.payload[5]);        // channel count
    TEST_ASSERT_TRUE(sensor.reads >= 3);
}

void test_silent_sensor_publishes_nothing() {
    Rig rig;
    CountingSensor quiet("quiet", 100, true, /*producesData=*/false);
    rig.registry.add(&quiet);
    NodeConfig config;
    config.statusIntervalMs = 0;
    config.commandTimeoutMs = 0;
    rig.node.configure(config);
    rig.node.begin();
    rig.transport.clearOut();

    for (uint8_t i = 0; i < 5; ++i) {
        rig.hal.advance(20);
        rig.node.spin();
    }

    Frame sample;
    // read() returning false means "no fresh data"; the node must stay silent
    // rather than publish the zeroed sample struct as a measurement.
    TEST_ASSERT_FALSE(findSent(rig.transport, MsgType::Sample, sample));
    TEST_ASSERT_TRUE(quiet.reads > 0);
}

void test_garbage_input_is_dropped_and_counted() {
    Rig rig;
    rig.node.begin();

    const uint8_t garbage[] = {0xFF, 0xAB, 0x12, 0x00};
    rig.transport.push(garbage, sizeof(garbage));
    rig.node.spin();

    TEST_ASSERT_TRUE(rig.node.droppedFrames() > 0);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)NodeState::Idle, (uint8_t)rig.node.state());
}

void test_resyncs_after_a_truncated_frame() {
    Rig rig;
    rig.node.begin();
    rig.transport.clearOut();

    // Half a frame, then a delimiter, then a good one. A receiver that cannot
    // resynchronise here is one power glitch away from being deaf forever.
    const uint8_t partial[] = {0x05, 0x01, 0x02, 0x00};
    rig.transport.push(partial, sizeof(partial));

    uint8_t payload[4];
    putI16(&payload[0], 150);
    putI16(&payload[2], 0);
    push(rig.transport, MsgType::Drive, payload, sizeof(payload), 9);

    rig.node.spin();
    TEST_ASSERT_EQUAL_INT16(150, rig.drive.lastLeftMmPerS());
}

void test_read_once_rejects_an_unknown_sensor() {
    Rig rig;
    rig.node.begin();
    rig.transport.clearOut();

    const uint8_t payload[1] = {77};
    push(rig.transport, MsgType::ReadOnce, payload, 1);
    rig.node.spin();

    Frame ack;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Ack, ack));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)AckCode::Rejected, ack.payload[1]);
}

void test_short_drive_payload_is_rejected() {
    Rig rig;
    rig.node.begin();
    rig.transport.clearOut();

    const uint8_t payload[2] = {0x10, 0x00};
    push(rig.transport, MsgType::Drive, payload, sizeof(payload));
    rig.node.spin();

    Frame ack;
    TEST_ASSERT_TRUE(findSent(rig.transport, MsgType::Ack, ack));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)AckCode::BadLength, ack.payload[1]);
    TEST_ASSERT_EQUAL_INT16(0, rig.drive.lastLeftMmPerS());
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_registry_assigns_sequential_ids);
    RUN_TEST(test_registry_refuses_null_and_overflow);
    RUN_TEST(test_failed_sensor_is_marked_and_never_sampled);
    RUN_TEST(test_on_demand_sensor_is_never_streamed);
    RUN_TEST(test_rate_limiting_respects_both_native_and_requested_rates);
    RUN_TEST(test_rate_limiting_survives_millis_rollover);
    RUN_TEST(test_enabled_mask_selects_sensors);
    RUN_TEST(test_hello_reports_capabilities);
    RUN_TEST(test_describe_emits_one_descriptor_per_sensor);
    RUN_TEST(test_drive_command_moves_the_wheels);
    RUN_TEST(test_lateral_command_is_refused_by_a_differential_base);
    RUN_TEST(test_estop_latches_and_refuses_drive);
    RUN_TEST(test_watchdog_stops_the_motors_when_the_host_goes_quiet);
    RUN_TEST(test_watchdog_is_held_off_by_any_traffic);
    RUN_TEST(test_streaming_publishes_samples_at_the_configured_rate);
    RUN_TEST(test_silent_sensor_publishes_nothing);
    RUN_TEST(test_garbage_input_is_dropped_and_counted);
    RUN_TEST(test_resyncs_after_a_truncated_frame);
    RUN_TEST(test_read_once_rejects_an_unknown_sensor);
    RUN_TEST(test_short_drive_payload_is_rejected);
    return UNITY_END();
}
