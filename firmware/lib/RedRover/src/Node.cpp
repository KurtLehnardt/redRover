#include "redrover/Node.h"

#include "redrover/Framing.h"

namespace redrover {

namespace {
// Cap the work done per spin() so a flooded input cannot delay the watchdog.
constexpr uint8_t kMaxFramesPerSpin = 4;
constexpr uint8_t kMaxSamplesPerSpin = 8;
}  // namespace

bool Node::begin() {
    transport_.begin();
    const bool driveOk = drive_.begin();
    sensors_.beginAll();

    lastCommandMs_ = hal_.millis();
    lastStatusMs_ = lastCommandMs_;
    state_ = driveOk ? NodeState::Idle : NodeState::Fault;
    if (!driveOk) {
        log(LogLevel::Error, "drive base failed to start");
    }
    return driveOk;
}

void Node::spin() {
    const uint32_t nowMs = hal_.millis();

    pumpInput();
    sensors_.pollAll(nowMs);
    publishSamples(nowMs);
    checkWatchdog(nowMs);

    if (config_.statusIntervalMs != 0 &&
        (nowMs - lastStatusMs_) >= config_.statusIntervalMs) {
        publishStatus();
        lastStatusMs_ = nowMs;
    }
}

// -- input -----------------------------------------------------------------

void Node::pumpInput() {
    uint8_t framesHandled = 0;
    uint8_t chunk[32];

    while (transport_.available() > 0 && framesHandled < kMaxFramesPerSpin) {
        const size_t got = transport_.read(chunk, sizeof(chunk));
        if (got == 0) {
            break;
        }
        for (size_t i = 0; i < got; ++i) {
            const uint8_t byte = chunk[i];
            if (byte == 0x00) {
                if (!rxOverflow_ && rxLen_ > 0) {
                    Frame frame;
                    if (decodeFrame(rxBuffer_, rxLen_, frame)) {
                        handleFrame(frame);
                        ++framesHandled;
                    } else {
                        ++dropped_;
                    }
                } else if (rxOverflow_) {
                    ++dropped_;
                }
                rxLen_ = 0;
                rxOverflow_ = false;
                continue;
            }
            if (rxLen_ >= sizeof(rxBuffer_)) {
                // A run longer than any legal frame means we are out of sync.
                // Drop until the next delimiter instead of corrupting memory.
                rxOverflow_ = true;
                continue;
            }
            rxBuffer_[rxLen_++] = byte;
        }
    }
}

void Node::handleFrame(const Frame& frame) {
    // Any well-formed frame counts as proof of life for the watchdog.
    lastCommandMs_ = hal_.millis();

    switch (frame.type) {
        case MsgType::Hello:
            publishHelloAck(frame.seq);
            break;
        case MsgType::Describe:
            publishDescriptors();
            sendAck(frame.type, frame.seq, AckCode::Ok);
            break;
        case MsgType::SetStream:
            handleSetStream(frame);
            break;
        case MsgType::Drive:
            handleDrive(frame);
            break;
        case MsgType::DriveRaw:
            handleDriveRaw(frame);
            break;
        case MsgType::Stop:
            applyStop();
            sendAck(frame.type, frame.seq, AckCode::Ok);
            break;
        case MsgType::EStop:
            if (frame.len >= 1 && frame.payload[0] != 0) {
                engageEmergencyStop();
            } else {
                clearEmergencyStop();
            }
            sendAck(frame.type, frame.seq, AckCode::Ok);
            break;
        case MsgType::SetAux:
            handleSetAux(frame);
            break;
        case MsgType::Ping:
            sendAck(frame.type, frame.seq, AckCode::Ok);
            break;
        case MsgType::ReadOnce:
            handleReadOnce(frame);
            break;
        default:
            sendAck(frame.type, frame.seq, AckCode::UnknownType);
            break;
    }
}

void Node::handleDrive(const Frame& frame) {
    if (frame.len < 4) {
        sendAck(frame.type, frame.seq, AckCode::BadLength);
        return;
    }
    if (estop_) {
        sendAck(frame.type, frame.seq, AckCode::EStopActive);
        return;
    }

    DriveCommand cmd;
    cmd.linearMmPerS = getI16(&frame.payload[0]);
    cmd.angularMradPerS = getI16(&frame.payload[2]);
    if (frame.len >= 6) {
        cmd.lateralMmPerS = getI16(&frame.payload[4]);
    }
    if (cmd.lateralMmPerS != 0 && !drive_.capabilities().holonomic) {
        // Silently ignoring a sideways command on a differential base would
        // make the robot drive somewhere the host did not ask for.
        sendAck(frame.type, frame.seq, AckCode::NotSupported);
        return;
    }

    drive_.drive(cmd);
    motorsActive_ = (cmd.linearMmPerS != 0 || cmd.angularMradPerS != 0 ||
                     cmd.lateralMmPerS != 0);
    state_ = motorsActive_ ? NodeState::Driving : NodeState::Idle;
    sendAck(frame.type, frame.seq, AckCode::Ok);
}

void Node::handleDriveRaw(const Frame& frame) {
    if (frame.len < 4) {
        sendAck(frame.type, frame.seq, AckCode::BadLength);
        return;
    }
    if (estop_) {
        sendAck(frame.type, frame.seq, AckCode::EStopActive);
        return;
    }
    const int16_t left = getI16(&frame.payload[0]);
    const int16_t right = getI16(&frame.payload[2]);
    drive_.driveRaw(left, right);
    motorsActive_ = (left != 0 || right != 0);
    state_ = motorsActive_ ? NodeState::Driving : NodeState::Idle;
    sendAck(frame.type, frame.seq, AckCode::Ok);
}

void Node::handleSetStream(const Frame& frame) {
    if (frame.len < 6) {
        sendAck(frame.type, frame.seq, AckCode::BadLength);
        return;
    }
    config_.streamRateHz = getU16(&frame.payload[0]);
    sensors_.setEnabledMask(getU32(&frame.payload[2]));
    sendAck(frame.type, frame.seq, AckCode::Ok);
}

void Node::handleReadOnce(const Frame& frame) {
    if (frame.len < 1) {
        sendAck(frame.type, frame.seq, AckCode::BadLength);
        return;
    }
    ISensor* sensor = sensors_.byId(frame.payload[0]);
    if (sensor == nullptr) {
        sendAck(frame.type, frame.seq, AckCode::Rejected);
        return;
    }
    SensorSample sample;
    if (sensor->read(sample)) {
        sample.id = sensor->descriptor().id;
        sample.timestampMs = hal_.millis();
        publishSample(sample);
        sendAck(frame.type, frame.seq, AckCode::Ok);
    } else {
        // No fresh reading: report the failure rather than publishing a stale
        // or zeroed sample the host would treat as a measurement.
        sendAck(frame.type, frame.seq, AckCode::Rejected);
    }
}

void Node::handleSetAux(const Frame& frame) {
    if (frame.len < 3) {
        sendAck(frame.type, frame.seq, AckCode::BadLength);
        return;
    }
    if (auxHandler_ == nullptr) {
        sendAck(frame.type, frame.seq, AckCode::NotSupported);
        return;
    }
    auxHandler_(frame.payload[0], getI16(&frame.payload[1]), auxUser_);
    sendAck(frame.type, frame.seq, AckCode::Ok);
}

// -- output ----------------------------------------------------------------

void Node::publishSamples(uint32_t nowMs) {
    uint8_t published = 0;
    for (uint8_t i = 0; i < sensors_.count() && published < kMaxSamplesPerSpin; ++i) {
        if (!sensors_.due(i, nowMs, config_.streamRateHz)) {
            continue;
        }
        ISensor* sensor = sensors_.at(i);
        SensorSample sample;
        if (sensor->read(sample)) {
            sample.id = sensor->descriptor().id;
            sample.timestampMs = nowMs;
            publishSample(sample);
            ++published;
        }
        // Mark the attempt either way so a silent sensor cannot spin the loop.
        sensors_.markSampled(i, nowMs);
    }
}

void Node::publishSample(const SensorSample& sample) {
    Frame frame;
    frame.type = MsgType::Sample;
    frame.seq = txSeq_++;

    uint8_t channels = sample.channels;
    if (channels > kMaxChannels) {
        channels = kMaxChannels;
    }
    const uint8_t needed = 6 + channels * 4;
    if (needed > kMaxPayload) {
        return;
    }

    frame.payload[0] = sample.id;
    putU32(&frame.payload[1], sample.timestampMs);
    frame.payload[5] = channels;
    for (uint8_t c = 0; c < channels; ++c) {
        putI32(&frame.payload[6 + c * 4], sample.values[c]);
    }
    frame.len = needed;
    send(frame);
}

void Node::publishDescriptors() {
    for (uint8_t i = 0; i < sensors_.count(); ++i) {
        const SensorDescriptor& d = sensors_.at(i)->descriptor();
        Frame frame;
        frame.type = MsgType::Descriptor;
        frame.seq = txSeq_++;
        frame.payload[0] = d.id;
        frame.payload[1] = static_cast<uint8_t>(d.kind);
        frame.payload[2] = static_cast<uint8_t>(d.unit);
        frame.payload[3] = d.channels;
        frame.payload[4] = static_cast<uint8_t>(d.scaleExp);
        putU16(&frame.payload[5], d.rateHz);
        frame.payload[7] = sensors_.failed(i) ? 1 : 0;
        for (uint8_t c = 0; c < kMaxNameLen; ++c) {
            frame.payload[8 + c] = static_cast<uint8_t>(d.name[c]);
        }
        frame.len = 8 + kMaxNameLen;
        send(frame);
    }
}

void Node::publishHelloAck(uint8_t seq) {
    Frame frame;
    frame.type = MsgType::HelloAck;
    frame.seq = seq;
    frame.payload[0] = kProtocolVersion;
    frame.payload[1] = sensors_.count();
    frame.payload[2] = drive_.capabilities().holonomic ? 1 : 0;
    putU16(&frame.payload[3], drive_.capabilities().maxLinearMmPerS);
    putU16(&frame.payload[5], drive_.capabilities().maxAngularMradPerS);
    putU16(&frame.payload[7], config_.commandTimeoutMs);
    for (uint8_t c = 0; c < kMaxNameLen; ++c) {
        frame.payload[9 + c] = static_cast<uint8_t>(config_.board[c]);
    }
    frame.len = 9 + kMaxNameLen;
    send(frame);
}

void Node::publishStatus() {
    Frame frame;
    frame.type = MsgType::Status;
    frame.seq = txSeq_++;
    frame.payload[0] = static_cast<uint8_t>(state_);
    frame.payload[1] = estop_ ? 1 : 0;
    putU16(&frame.payload[2], batteryMv_);
    putU32(&frame.payload[4], hal_.millis());
    putU16(&frame.payload[8], dropped_);
    frame.len = 10;
    send(frame);
}

void Node::sendAck(MsgType ofType, uint8_t seq, AckCode code) {
    Frame frame;
    frame.type = MsgType::Ack;
    frame.seq = seq;
    frame.payload[0] = static_cast<uint8_t>(ofType);
    frame.payload[1] = static_cast<uint8_t>(code);
    frame.len = 2;
    send(frame);
}

void Node::log(LogLevel level, const char* text) {
    Frame frame;
    frame.type = MsgType::Log;
    frame.seq = txSeq_++;
    frame.payload[0] = static_cast<uint8_t>(level);
    uint8_t i = 0;
    while (text != nullptr && text[i] != '\0' && i + 1 < kMaxPayload) {
        frame.payload[1 + i] = static_cast<uint8_t>(text[i]);
        ++i;
    }
    frame.len = static_cast<uint8_t>(1 + i);
    send(frame);
}

void Node::send(const Frame& frame) {
    uint8_t out[cobsMaxEncoded(kMaxFrame) + 1];
    const size_t n = encodeFrame(frame, out, sizeof(out));
    if (n == 0) {
        ++dropped_;
        return;
    }
    if (transport_.write(out, n) != n) {
        ++dropped_;
    }
}

// -- safety ----------------------------------------------------------------

void Node::checkWatchdog(uint32_t nowMs) {
    if (config_.commandTimeoutMs == 0 || !motorsActive_) {
        return;
    }
    if ((nowMs - lastCommandMs_) >= config_.commandTimeoutMs) {
        applyStop();
        log(LogLevel::Warn, "command timeout: motors stopped");
    }
}

void Node::applyStop() {
    drive_.stop();
    motorsActive_ = false;
    if (state_ == NodeState::Driving) {
        state_ = NodeState::Idle;
    }
}

void Node::engageEmergencyStop() {
    estop_ = true;
    drive_.stop();
    motorsActive_ = false;
    state_ = NodeState::EmergencyStop;
}

void Node::clearEmergencyStop() {
    estop_ = false;
    if (state_ == NodeState::EmergencyStop) {
        state_ = NodeState::Idle;
    }
}

}  // namespace redrover
