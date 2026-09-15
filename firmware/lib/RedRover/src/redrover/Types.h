// redRover portable firmware — core value types.
//
// Nothing in the core includes Arduino.h or allocates. It compiles for an
// ATmega328P (2 KB RAM) as readily as for an ESP32, and for the host so the
// protocol can be unit-tested natively.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace redrover {

constexpr uint8_t kProtocolVersion = 1;

// Upper bounds. Sized so a full descriptor table and one sample frame fit in
// an Uno's SRAM with room to spare; raise them on larger targets by defining
// these before including the library.
#ifndef REDROVER_MAX_SENSORS
#define REDROVER_MAX_SENSORS 12
#endif
#ifndef REDROVER_MAX_CHANNELS
#define REDROVER_MAX_CHANNELS 6
#endif
#ifndef REDROVER_MAX_FRAME
#define REDROVER_MAX_FRAME 96
#endif

constexpr uint8_t kMaxSensors = REDROVER_MAX_SENSORS;
constexpr uint8_t kMaxChannels = REDROVER_MAX_CHANNELS;
constexpr uint8_t kMaxFrame = REDROVER_MAX_FRAME;
constexpr uint8_t kMaxNameLen = 16;

// What a sensor measures. The host maps these onto its own modalities, so a
// new sensor type needs no host-side code as long as it reports a known kind.
enum class SensorKind : uint8_t {
    Generic = 0,
    Acceleration = 1,   // vibration / IMU
    AngularRate = 2,
    Magnetic = 3,
    Distance = 4,       // ultrasonic, ToF, lidar point
    Temperature = 5,
    Humidity = 6,
    Pressure = 7,
    Sound = 8,          // microphone / acoustic emission
    Light = 9,
    Voltage = 10,
    Current = 11,
    Gas = 12,
    Encoder = 13,
    Bumper = 14,        // binary contact
    Odometry = 15,      // x, y, heading
    Battery = 16,
    Custom = 200,
};

enum class Unit : uint8_t {
    None = 0,
    MetrePerSecond2 = 1,
    DegreePerSecond = 2,
    Metre = 3,
    Celsius = 4,
    Percent = 5,
    Pascal = 6,
    Volt = 7,
    Ampere = 8,
    Tesla = 9,
    Lux = 10,
    Count = 11,
    PartsPerMillion = 12,
    Degree = 13,
    Boolean = 14,
};

// A sensor describes itself once; the host builds its pipeline from this and
// never needs a hard-coded table of device types.
struct SensorDescriptor {
    uint8_t id = 0;
    SensorKind kind = SensorKind::Generic;
    Unit unit = Unit::None;
    uint8_t channels = 1;
    // Reported values are integers; the real value is raw * 10^scaleExp.
    // Integer transport keeps the wire format and the AVR maths float-free.
    int8_t scaleExp = 0;
    // Native sampling rate in Hz (0 = on demand only).
    uint16_t rateHz = 0;
    char name[kMaxNameLen] = {0};
};

// One reading. `values` holds `channels` entries in the descriptor's units.
struct SensorSample {
    uint8_t id = 0;
    uint32_t timestampMs = 0;
    uint8_t channels = 0;
    int32_t values[kMaxChannels] = {0};
};

// Chassis-level velocity command, in integer units so the whole control path
// is float-free.
//
// Angular velocity is in **milliradians per second**, not millidegrees: an
// int16 of millidegrees saturates at 32.7 deg/s, which is slower than any real
// robot turns. Milliradians give +/-1877 deg/s at 0.057 deg/s resolution, and
// v = omega * r needs no pi constant when omega is in rad.
struct DriveCommand {
    int16_t linearMmPerS = 0;    // forward positive
    int16_t angularMradPerS = 0; // counter-clockwise positive
    int16_t lateralMmPerS = 0;   // holonomic bases only; 0 otherwise
};

enum class NodeState : uint8_t {
    Booting = 0,
    Idle = 1,
    Driving = 2,
    EmergencyStop = 3,
    Fault = 4,
};

// Copy a C string into a fixed-size name field without <cstring>, which is
// not uniformly available on every embedded toolchain.
inline void setName(char* dest, const char* src, uint8_t capacity = kMaxNameLen) {
    uint8_t i = 0;
    if (src != nullptr) {
        for (; i + 1 < capacity && src[i] != '\0'; ++i) {
            dest[i] = src[i];
        }
    }
    for (; i < capacity; ++i) {
        dest[i] = '\0';
    }
}

}  // namespace redrover
