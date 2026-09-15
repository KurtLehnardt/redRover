// Host-side tests for framing and the wire protocol.
#include <unity.h>

#include "redrover/Framing.h"
#include "redrover/Protocol.h"

using namespace redrover;

void setUp() {}
void tearDown() {}

// --- COBS -----------------------------------------------------------------

static void roundTrip(const uint8_t* data, size_t len) {
    uint8_t encoded[512];
    uint8_t decoded[512];
    const size_t n = cobsEncode(data, len, encoded);
    TEST_ASSERT_TRUE(n <= cobsMaxEncoded(len));
    // The whole point of COBS: no zero byte can appear inside a frame, so the
    // delimiter is unambiguous.
    for (size_t i = 0; i < n; ++i) {
        TEST_ASSERT_TRUE(encoded[i] != 0);
    }
    const size_t back = cobsDecode(encoded, n, decoded, sizeof(decoded));
    TEST_ASSERT_EQUAL_UINT32((uint32_t)len, (uint32_t)back);
    for (size_t i = 0; i < len; ++i) {
        TEST_ASSERT_EQUAL_UINT8(data[i], decoded[i]);
    }
}

void test_cobs_round_trip_simple() {
    const uint8_t data[] = {1, 2, 3, 4, 5};
    roundTrip(data, sizeof(data));
}

void test_cobs_round_trip_with_zeros() {
    const uint8_t data[] = {0, 1, 0, 0, 2, 0};
    roundTrip(data, sizeof(data));
}

void test_cobs_round_trip_all_zeros() {
    uint8_t data[32];
    for (uint8_t& b : data) b = 0;
    roundTrip(data, sizeof(data));
}

void test_cobs_round_trip_long_run_without_zeros() {
    // A run longer than 254 forces the code-byte split path.
    uint8_t data[300];
    for (size_t i = 0; i < sizeof(data); ++i) data[i] = (uint8_t)(1 + (i % 250));
    roundTrip(data, sizeof(data));
}

void test_cobs_rejects_zero_code_byte() {
    const uint8_t bad[] = {0x00, 0x01};
    uint8_t out[8];
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)cobsDecode(bad, sizeof(bad), out, sizeof(out)));
}

void test_cobs_rejects_overrunning_run() {
    const uint8_t bad[] = {0x10, 0x01, 0x02};  // claims 15 bytes, has 2
    uint8_t out[8];
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)cobsDecode(bad, sizeof(bad), out, sizeof(out)));
}

// --- CRC ------------------------------------------------------------------

void test_crc16_known_vector() {
    // CRC-16/CCITT-FALSE of "123456789" is 0x29B1.
    const uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    TEST_ASSERT_EQUAL_HEX16(0x29B1, crc16(data, sizeof(data)));
}

void test_crc16_detects_single_bit_flip() {
    uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x10};
    const uint16_t before = crc16(data, sizeof(data));
    data[2] ^= 0x01;
    TEST_ASSERT_TRUE(crc16(data, sizeof(data)) != before);
}

// --- frames ---------------------------------------------------------------

void test_frame_round_trip() {
    Frame in;
    in.type = MsgType::Drive;
    in.seq = 42;
    in.len = 4;
    putI16(&in.payload[0], -250);
    putI16(&in.payload[2], 9000);

    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(in, wire, sizeof(wire));
    TEST_ASSERT_TRUE(n > 0);
    TEST_ASSERT_EQUAL_UINT8(0, wire[n - 1]);  // delimiter

    Frame out;
    TEST_ASSERT_TRUE(decodeFrame(wire, n - 1, out));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)MsgType::Drive, (uint8_t)out.type);
    TEST_ASSERT_EQUAL_UINT8(42, out.seq);
    TEST_ASSERT_EQUAL_UINT8(4, out.len);
    TEST_ASSERT_EQUAL_INT16(-250, getI16(&out.payload[0]));
    TEST_ASSERT_EQUAL_INT16(9000, getI16(&out.payload[2]));
}

void test_frame_round_trip_empty_payload() {
    Frame in;
    in.type = MsgType::Stop;
    in.seq = 7;
    in.len = 0;

    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(in, wire, sizeof(wire));
    Frame out;
    TEST_ASSERT_TRUE(decodeFrame(wire, n - 1, out));
    TEST_ASSERT_EQUAL_UINT8((uint8_t)MsgType::Stop, (uint8_t)out.type);
    TEST_ASSERT_EQUAL_UINT8(0, out.len);
}

void test_frame_rejects_corrupted_payload() {
    Frame in;
    in.type = MsgType::Drive;
    in.seq = 1;
    in.len = 4;
    putI16(&in.payload[0], 100);
    putI16(&in.payload[2], 0);

    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(in, wire, sizeof(wire));
    wire[4] ^= 0x20;  // flip a bit inside the frame

    Frame out;
    // A corrupted drive command must be dropped, not acted on: this is a
    // motor command travelling over a noisy USB or radio link.
    TEST_ASSERT_FALSE(decodeFrame(wire, n - 1, out));
}

void test_frame_rejects_wrong_protocol_version() {
    Frame in;
    in.type = MsgType::Ping;
    in.seq = 0;
    in.len = 0;
    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(in, wire, sizeof(wire));

    uint8_t raw[kMaxFrame];
    const size_t decoded = cobsDecode(wire, n - 1, raw, sizeof(raw));
    raw[0] = kProtocolVersion + 1;
    putU16(&raw[decoded - 2], crc16(raw, decoded - 2));  // keep the CRC valid

    uint8_t rewire[kMaxFrame * 2];
    const size_t m = cobsEncode(raw, decoded, rewire);

    Frame out;
    TEST_ASSERT_FALSE(decodeFrame(rewire, m, out));
}

void test_frame_rejects_oversized_payload() {
    Frame in;
    in.type = MsgType::Log;
    in.len = kMaxPayload + 1;
    uint8_t wire[kMaxFrame * 2];
    TEST_ASSERT_EQUAL_UINT32(0, (uint32_t)encodeFrame(in, wire, sizeof(wire)));
}

void test_payload_codecs_are_little_endian() {
    uint8_t buf[4];
    putU32(buf, 0x12345678u);
    TEST_ASSERT_EQUAL_UINT8(0x78, buf[0]);
    TEST_ASSERT_EQUAL_UINT8(0x12, buf[3]);
    TEST_ASSERT_EQUAL_UINT32(0x12345678u, getU32(buf));

    putI32(buf, -123456);
    TEST_ASSERT_EQUAL_INT32(-123456, getI32(buf));

    uint8_t small[2];
    putI16(small, -1);
    TEST_ASSERT_EQUAL_INT16(-1, getI16(small));
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_cobs_round_trip_simple);
    RUN_TEST(test_cobs_round_trip_with_zeros);
    RUN_TEST(test_cobs_round_trip_all_zeros);
    RUN_TEST(test_cobs_round_trip_long_run_without_zeros);
    RUN_TEST(test_cobs_rejects_zero_code_byte);
    RUN_TEST(test_cobs_rejects_overrunning_run);
    RUN_TEST(test_crc16_known_vector);
    RUN_TEST(test_crc16_detects_single_bit_flip);
    RUN_TEST(test_frame_round_trip);
    RUN_TEST(test_frame_round_trip_empty_payload);
    RUN_TEST(test_frame_rejects_corrupted_payload);
    RUN_TEST(test_frame_rejects_wrong_protocol_version);
    RUN_TEST(test_frame_rejects_oversized_payload);
    RUN_TEST(test_payload_codecs_are_little_endian);
    return UNITY_END();
}
