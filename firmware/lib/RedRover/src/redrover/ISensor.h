// The sensor interface. Implementing it is the whole cost of adding a sensor:
// describe yourself, and fill in a sample when asked.
#pragma once

#include "Types.h"

namespace redrover {

class ISensor {
public:
    virtual ~ISensor() = default;

    // Called once at startup. Return false to have the node skip this sensor
    // and report it as failed rather than publishing meaningless readings.
    virtual bool begin() = 0;

    // Fill `out` with the latest reading. Return false when no new sample is
    // available; the node will not publish a stale value as if it were fresh.
    virtual bool read(SensorSample& out) = 0;

    // Self-description used by the host to build its pipeline.
    virtual const SensorDescriptor& descriptor() const = 0;

    // Optional periodic work for sensors that need to drive a state machine
    // between reads (settling delays, conversion waits, DMA completion).
    virtual void poll(uint32_t /*nowMs*/) {}

    // Assigned by SensorRegistry at registration time.
    void assignId(uint8_t id) { mutableDescriptor().id = id; }

protected:
    // Implementations return their own storage; the default read-only
    // descriptor() then simply forwards to it.
    virtual SensorDescriptor& mutableDescriptor() = 0;
};

// Convenience base that owns the descriptor, so a concrete sensor only has to
// fill it in and implement read().
class SensorBase : public ISensor {
public:
    const SensorDescriptor& descriptor() const override { return descriptor_; }
    bool begin() override { return true; }

protected:
    SensorDescriptor& mutableDescriptor() override { return descriptor_; }
    SensorDescriptor descriptor_;
};

}  // namespace redrover
