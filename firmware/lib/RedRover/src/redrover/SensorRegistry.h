// Fixed-capacity sensor table.
//
// Adding a sensor is one `registry.add(&mySensor)` call. The registry assigns
// the id, tracks which sensors failed begin(), and paces each sensor at its
// own rate so a 1 kHz accelerometer and a 1 Hz thermometer coexist without
// either one dictating the loop.
#pragma once

#include "ISensor.h"
#include "Types.h"

namespace redrover {

// The enabled/failed masks below are uint32_t, so index N is addressed as
// 1UL << N. Past 32 sensors that shift is undefined behaviour and the masks
// would silently stop tracking the extra sensors.
static_assert(kMaxSensors <= 32,
              "REDROVER_MAX_SENSORS cannot exceed 32: the enabled and failed "
              "masks are 32-bit. Widen them to uint64_t to raise this.");
static_assert(kMaxChannels >= 1, "REDROVER_MAX_CHANNELS must be at least 1");

class SensorRegistry {
public:
    // Register a sensor. Returns the assigned id, or 0xFF if the table is
    // full. The pointer must outlive the registry; sensors are normally
    // static or global objects, and the registry never allocates.
    uint8_t add(ISensor* sensor);

    // Call begin() on every sensor. Returns the number that started
    // successfully; the rest are marked failed and never publish.
    uint8_t beginAll();

    uint8_t count() const { return count_; }
    ISensor* at(uint8_t index) const {
        return index < count_ ? sensors_[index] : nullptr;
    }
    ISensor* byId(uint8_t id) const;

    bool failed(uint8_t index) const {
        return index < count_ && (failedMask_ & (1UL << index)) != 0;
    }

    // Give every healthy sensor a chance to advance its state machine.
    void pollAll(uint32_t nowMs);

    // True when `index` is due for a sample at `nowMs`, given the streaming
    // rate. A sensor with rateHz == 0 is on-demand only and never due.
    bool due(uint8_t index, uint32_t nowMs, uint16_t streamRateHz) const;
    void markSampled(uint8_t index, uint32_t nowMs);

    // Streaming selection: bit N enables the sensor at index N.
    void setEnabledMask(uint32_t mask) { enabledMask_ = mask; }
    uint32_t enabledMask() const { return enabledMask_; }
    bool enabled(uint8_t index) const { return (enabledMask_ & (1UL << index)) != 0; }

private:
    ISensor* sensors_[kMaxSensors] = {nullptr};
    uint32_t lastSampleMs_[kMaxSensors] = {0};
    uint8_t count_ = 0;
    uint32_t failedMask_ = 0;
    uint32_t enabledMask_ = 0xFFFFFFFFUL;
};

}  // namespace redrover
