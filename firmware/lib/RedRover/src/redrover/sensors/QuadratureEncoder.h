// Quadrature wheel encoder, polled or interrupt-fed.
//
// On boards with spare interrupts, call tickA()/tickB() from an ISR and leave
// `polled` false. On an Uno with both interrupt pins already spoken for, the
// polled mode reads the pins in poll(), which is adequate up to a few kHz.
#pragma once

#include "redrover/Hal.h"
#include "redrover/ISensor.h"

namespace redrover {

class QuadratureEncoder : public SensorBase {
public:
    QuadratureEncoder(Hal& hal, uint8_t pinA, uint8_t pinB, const char* name = "encoder",
                      uint16_t rateHz = 50, bool polled = true)
        : hal_(hal), pinA_(pinA), pinB_(pinB), polled_(polled) {
        descriptor_.kind = SensorKind::Encoder;
        descriptor_.unit = Unit::Count;
        descriptor_.channels = 1;
        descriptor_.rateHz = rateHz;
        setName(descriptor_.name, name);
    }

    bool begin() override {
        hal_.configureInput(pinA_, true);
        hal_.configureInput(pinB_, true);
        lastState_ = encodeState();
        return true;
    }

    void poll(uint32_t) override {
        if (!polled_) {
            return;
        }
        const uint8_t state = encodeState();
        if (state != lastState_) {
            count_ += delta(lastState_, state);
            lastState_ = state;
        }
    }

    // Call from an interrupt when `polled` is false.
    void onEdge() {
        const uint8_t state = encodeState();
        count_ += delta(lastState_, state);
        lastState_ = state;
    }

    bool read(SensorSample& out) override {
        out.channels = 1;
        out.values[0] = count();
        return true;
    }

    // Guarded: onEdge() may run from an ISR, and a 32-bit load on AVR is four
    // instructions. An interrupt between them returns a count that was never
    // real -- typically off by 2^8 or 2^16, which reads as a wheel that
    // teleported.
    int32_t count() const {
        CriticalSection guard(hal_);
        return count_;
    }

    void reset() {
        CriticalSection guard(hal_);
        count_ = 0;
    }

private:
    uint8_t encodeState() const {
        return static_cast<uint8_t>((hal_.readDigital(pinA_) ? 0x02 : 0) |
                                    (hal_.readDigital(pinB_) ? 0x01 : 0));
    }

    // Gray-code transition table: +1 forward, -1 reverse, 0 for no motion or
    // an illegal double transition (which means we missed an edge).
    static int8_t delta(uint8_t from, uint8_t to) {
        static const int8_t kTable[16] = {
            0, -1, +1, 0,
            +1, 0, 0, -1,
            -1, 0, 0, +1,
            0, +1, -1, 0,
        };
        return kTable[((from & 0x03) << 2) | (to & 0x03)];
    }

    Hal& hal_;
    uint8_t pinA_;
    uint8_t pinB_;
    bool polled_;
    uint8_t lastState_ = 0;
    volatile int32_t count_ = 0;
};

}  // namespace redrover
