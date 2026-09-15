#include "redrover/SensorRegistry.h"

namespace redrover {

uint8_t SensorRegistry::add(ISensor* sensor) {
    if (sensor == nullptr || count_ >= kMaxSensors) {
        return 0xFF;
    }
    const uint8_t id = count_;
    sensors_[count_] = sensor;
    lastSampleMs_[count_] = 0;
    sensor->assignId(id);
    ++count_;
    return id;
}

uint8_t SensorRegistry::beginAll() {
    uint8_t ok = 0;
    for (uint8_t i = 0; i < count_; ++i) {
        if (sensors_[i]->begin()) {
            ++ok;
        } else {
            failedMask_ |= (1UL << i);
        }
    }
    return ok;
}

ISensor* SensorRegistry::byId(uint8_t id) const {
    for (uint8_t i = 0; i < count_; ++i) {
        if (sensors_[i]->descriptor().id == id) {
            return sensors_[i];
        }
    }
    return nullptr;
}

void SensorRegistry::pollAll(uint32_t nowMs) {
    for (uint8_t i = 0; i < count_; ++i) {
        if (!failed(i)) {
            sensors_[i]->poll(nowMs);
        }
    }
}

bool SensorRegistry::due(uint8_t index, uint32_t nowMs, uint16_t streamRateHz) const {
    if (index >= count_ || failed(index) || !enabled(index)) {
        return false;
    }
    const uint16_t native = sensors_[index]->descriptor().rateHz;
    if (native == 0) {
        return false;  // on-demand sensors are read only via ReadOnce
    }
    // Never sample faster than the sensor's own rate, nor faster than the host
    // asked for.
    uint16_t rate = native;
    if (streamRateHz != 0 && streamRateHz < rate) {
        rate = streamRateHz;
    }
    const uint32_t periodMs = rate >= 1000 ? 1u : (1000u / rate);
    // Unsigned subtraction handles the 49-day millis() rollover correctly.
    return (nowMs - lastSampleMs_[index]) >= periodMs;
}

void SensorRegistry::markSampled(uint8_t index, uint32_t nowMs) {
    if (index < count_) {
        lastSampleMs_[index] = nowMs;
    }
}

}  // namespace redrover
