// Emits golden wire vectors from the C++ implementation.
//
// The Python host mirrors this protocol by hand; `tests/test_wire.py` replays
// these vectors so a change on either side that breaks byte compatibility
// fails the test suite rather than the robot.
//
//   g++ -std=c++17 -Ifirmware/lib/RedRover/src \
//       firmware/lib/RedRover/src/{Framing,Protocol}.cpp \
//       firmware/tools/gen_vectors.cpp -o gen_vectors
//   ./gen_vectors > tests/data/wire_vectors.json
#include <cstdio>
#include <cstring>

#include "redrover/Protocol.h"

using namespace redrover;

namespace {

bool first = true;

void emit(const char* name, const Frame& frame) {
    uint8_t wire[kMaxFrame * 2];
    const size_t n = encodeFrame(frame, wire, sizeof(wire));
    printf("%s\n    {\"name\": \"%s\", \"type\": %u, \"seq\": %u, \"payload\": \"",
           first ? "" : ",", name, (unsigned)frame.type, frame.seq);
    for (uint8_t i = 0; i < frame.len; ++i) printf("%02x", frame.payload[i]);
    printf("\", \"wire\": \"");
    for (size_t i = 0; i < n; ++i) printf("%02x", wire[i]);
    printf("\"}");
    first = false;
}

Frame make(MsgType type, uint8_t seq, const uint8_t* payload, uint8_t len) {
    Frame frame;
    frame.type = type;
    frame.seq = seq;
    frame.len = len;
    for (uint8_t i = 0; i < len; ++i) frame.payload[i] = payload[i];
    return frame;
}

}  // namespace

int main() {
    printf("{\n  \"protocol_version\": %u,\n  \"frames\": [", kProtocolVersion);

    emit("ping", make(MsgType::Ping, 0, nullptr, 0));
    emit("hello", make(MsgType::Hello, 1, nullptr, 0));
    emit("stop", make(MsgType::Stop, 255, nullptr, 0));

    {   // drive: forward and turning
        uint8_t p[6];
        putI16(&p[0], 250);
        putI16(&p[2], 1571);
        putI16(&p[4], 0);
        emit("drive_forward_turn", make(MsgType::Drive, 42, p, sizeof(p)));
    }
    {   // drive: reverse, negative values exercise sign handling
        uint8_t p[6];
        putI16(&p[0], -500);
        putI16(&p[2], -3000);
        putI16(&p[4], -120);
        emit("drive_reverse_strafe", make(MsgType::Drive, 7, p, sizeof(p)));
    }
    {   // payload containing 0x00 bytes, which is what COBS exists for
        uint8_t p[6];
        putI16(&p[0], 0);
        putI16(&p[2], 0);
        putI16(&p[4], 0);
        emit("drive_all_zero", make(MsgType::Drive, 3, p, sizeof(p)));
    }
    {   // e-stop engage
        const uint8_t p[1] = {1};
        emit("estop_engage", make(MsgType::EStop, 9, p, 1));
    }
    {   // set_stream at 200 Hz, first three sensors
        uint8_t p[6];
        putU16(&p[0], 200);
        putU32(&p[2], 0x00000007u);
        emit("set_stream", make(MsgType::SetStream, 11, p, sizeof(p)));
    }
    {   // sample: three channels of milli-g
        uint8_t p[18];
        p[0] = 2;
        putU32(&p[1], 123456u);
        p[5] = 3;
        putI32(&p[6], -1024);
        putI32(&p[10], 0);
        putI32(&p[14], 1000);
        emit("sample_accel", make(MsgType::Sample, 77, p, sizeof(p)));
    }
    {   // descriptor for that accelerometer
        uint8_t p[8 + kMaxNameLen];
        memset(p, 0, sizeof(p));
        p[0] = 2;
        p[1] = static_cast<uint8_t>(SensorKind::Acceleration);
        p[2] = static_cast<uint8_t>(Unit::MetrePerSecond2);
        p[3] = 3;
        p[4] = static_cast<uint8_t>(static_cast<int8_t>(-3));
        putU16(&p[5], 1000);
        p[7] = 0;
        const char* name = "vibration";
        for (uint8_t i = 0; name[i] != '\0'; ++i) p[8 + i] = static_cast<uint8_t>(name[i]);
        emit("descriptor_accel", make(MsgType::Descriptor, 12, p, sizeof(p)));
    }
    {   // status
        uint8_t p[10];
        p[0] = static_cast<uint8_t>(NodeState::Driving);
        p[1] = 0;
        putU16(&p[2], 11700);
        putU32(&p[4], 98765u);
        putU16(&p[8], 4);
        emit("status_driving", make(MsgType::Status, 13, p, sizeof(p)));
    }
    {   // ack
        const uint8_t p[2] = {static_cast<uint8_t>(MsgType::Drive),
                              static_cast<uint8_t>(AckCode::EStopActive)};
        emit("ack_estop_active", make(MsgType::Ack, 42, p, 2));
    }
    {   // hello_ack
        uint8_t p[9 + kMaxNameLen];
        memset(p, 0, sizeof(p));
        p[0] = kProtocolVersion;
        p[1] = 3;
        p[2] = 0;
        putU16(&p[3], 450);
        putU16(&p[5], 4500);
        putU16(&p[7], 500);
        const char* board = "basic-rover";
        for (uint8_t i = 0; board[i] != '\0'; ++i) p[9 + i] = static_cast<uint8_t>(board[i]);
        emit("hello_ack", make(MsgType::HelloAck, 1, p, sizeof(p)));
    }
    {   // log line
        uint8_t p[32];
        p[0] = static_cast<uint8_t>(LogLevel::Warn);
        const char* text = "command timeout";
        uint8_t i = 0;
        for (; text[i] != '\0'; ++i) p[1 + i] = static_cast<uint8_t>(text[i]);
        emit("log_warn", make(MsgType::Log, 14, p, static_cast<uint8_t>(1 + i)));
    }

    printf("\n  ]\n}\n");
    return 0;
}
