// The firmware node: one object that owns the transport, the chassis, and the
// sensor table, and is driven from loop().
//
// Safety properties it enforces, independent of what the host does:
//
//   * **Command watchdog.** Motors stop if no Drive/Ping arrives within
//     `commandTimeoutMs`. A dropped USB cable or a crashed host leaves the
//     robot stationary, not running away at its last commanded speed.
//   * **Latching e-stop.** Once engaged, drive commands are refused until the
//     host explicitly clears it.
//   * **Bounded work per loop.** Frame parsing and sample publishing are
//     capped so a flood of input cannot starve the watchdog.
#pragma once

#include "IDriveBase.h"
#include "ITransport.h"
#include "Hal.h"
#include "Protocol.h"
#include "SensorRegistry.h"
#include "Types.h"

namespace redrover {

struct NodeConfig {
    // Stop the motors if no command is heard for this long. 0 disables the
    // watchdog, which is only appropriate on a bench.
    uint16_t commandTimeoutMs = 500;
    // Unsolicited status cadence.
    uint16_t statusIntervalMs = 1000;
    // Default streaming rate until the host sets one. 0 = use each sensor's
    // native rate.
    uint16_t streamRateHz = 0;
    // Human-readable board identifier reported in HelloAck.
    char board[kMaxNameLen] = {0};
};

// Optional hook for aux channels (LEDs, servos, relays) so a sketch can
// respond to SetAux without subclassing Node.
using AuxHandler = void (*)(uint8_t channel, int16_t value, void* user);

class Node {
public:
    Node(ITransport& transport, IDriveBase& drive, SensorRegistry& sensors, Hal& hal)
        : transport_(transport), drive_(drive), sensors_(sensors), hal_(hal) {}

    void configure(const NodeConfig& config) { config_ = config; }
    void setAuxHandler(AuxHandler handler, void* user = nullptr) {
        auxHandler_ = handler;
        auxUser_ = user;
    }
    // Reported in Status so the host can show a real battery level.
    void setBatteryMillivolts(uint16_t mv) { batteryMv_ = mv; }

    // Bring up transport, chassis, and sensors. Returns false if the chassis
    // failed to start, which is the one failure that makes the node unsafe to
    // run at all.
    bool begin();

    // Call from loop() as often as possible.
    void spin();

    NodeState state() const { return state_; }
    bool emergencyStopped() const { return estop_; }

    // Engage or clear the latching emergency stop locally, e.g. from a
    // hardware button wired into the sketch.
    void engageEmergencyStop();
    void clearEmergencyStop();

    void log(LogLevel level, const char* text);

    uint16_t droppedFrames() const { return dropped_; }

private:
    void pumpInput();
    void handleFrame(const Frame& frame);
    void handleDrive(const Frame& frame);
    void handleDriveRaw(const Frame& frame);
    void handleSetStream(const Frame& frame);
    void handleReadOnce(const Frame& frame);
    void handleSetAux(const Frame& frame);

    void publishSamples(uint32_t nowMs);
    void publishSample(const SensorSample& sample);
    void publishDescriptors();
    void publishHelloAck(uint8_t seq);
    void publishStatus();
    void sendAck(MsgType ofType, uint8_t seq, AckCode code);
    void send(const Frame& frame);

    void checkWatchdog(uint32_t nowMs);
    void applyStop();

    ITransport& transport_;
    IDriveBase& drive_;
    SensorRegistry& sensors_;
    Hal& hal_;
    NodeConfig config_;

    AuxHandler auxHandler_ = nullptr;
    void* auxUser_ = nullptr;

    NodeState state_ = NodeState::Booting;
    bool estop_ = false;
    bool motorsActive_ = false;

    uint32_t lastCommandMs_ = 0;
    uint32_t lastStatusMs_ = 0;
    uint16_t batteryMv_ = 0;
    uint16_t dropped_ = 0;
    uint8_t txSeq_ = 0;

    // Receive reassembly: bytes accumulate until a 0x00 delimiter completes a
    // frame. An oversized run is discarded rather than growing a buffer.
    uint8_t rxBuffer_[kMaxFrame * 2];
    uint16_t rxLen_ = 0;
    bool rxOverflow_ = false;
};

}  // namespace redrover
