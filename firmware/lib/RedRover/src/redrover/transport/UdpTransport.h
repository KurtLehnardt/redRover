// WiFi/UDP transport for ESP32-class boards — an untethered rover on the same
// network as the host, with the same framing as the serial link.
#pragma once

#if defined(ESP32) || defined(ESP8266)

#include <WiFi.h>
#include <WiFiUdp.h>

#include "redrover/ITransport.h"

namespace redrover {

class UdpTransport : public ITransport {
public:
    UdpTransport(uint16_t localPort = 4242) : localPort_(localPort) {}

    bool begin() override { return udp_.begin(localPort_) == 1; }

    size_t available() override {
        if (rxLen_ > rxPos_) {
            return rxLen_ - rxPos_;
        }
        const int packet = udp_.parsePacket();
        if (packet <= 0) {
            return 0;
        }
        // Remember the sender so replies reach whoever is actually driving.
        peer_ = udp_.remoteIP();
        peerPort_ = udp_.remotePort();
        const int got = udp_.read(rxBuffer_, sizeof(rxBuffer_));
        rxLen_ = got > 0 ? static_cast<size_t>(got) : 0;
        rxPos_ = 0;
        return rxLen_;
    }

    size_t read(uint8_t* buffer, size_t len) override {
        size_t count = 0;
        while (count < len && rxPos_ < rxLen_) {
            buffer[count++] = rxBuffer_[rxPos_++];
        }
        return count;
    }

    size_t write(const uint8_t* buffer, size_t len) override {
        if (peerPort_ == 0) {
            return 0;  // nobody has said hello yet; nowhere to send
        }
        udp_.beginPacket(peer_, peerPort_);
        const size_t written = udp_.write(buffer, len);
        udp_.endPacket();
        return written;
    }

    bool connected() override { return peerPort_ != 0; }

private:
    WiFiUDP udp_;
    uint16_t localPort_;
    IPAddress peer_;
    uint16_t peerPort_ = 0;
    uint8_t rxBuffer_[512];
    size_t rxLen_ = 0;
    size_t rxPos_ = 0;
};

}  // namespace redrover

#endif  // ESP32 || ESP8266
