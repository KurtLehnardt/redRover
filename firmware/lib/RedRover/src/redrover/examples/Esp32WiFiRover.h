// Example: an untethered ESP32 rover reachable over WiFi/UDP.
//
// Same node, same protocol, same host code — only the transport changes.
#pragma once

#if defined(ESP32)

#include <WiFi.h>
#include <redRover.h>

namespace {

// Supply these at build time so credentials stay out of the repository:
//   build_flags = -DREDROVER_WIFI_SSID='"my-net"' -DREDROVER_WIFI_PASS='"..."'
#ifndef REDROVER_WIFI_SSID
#define REDROVER_WIFI_SSID "set-REDROVER_WIFI_SSID"
#endif
#ifndef REDROVER_WIFI_PASS
#define REDROVER_WIFI_PASS ""
#endif

redrover::ArduinoHal hal;
redrover::UdpTransport transport(4242);

redrover::DifferentialGeometry geometry = {200, 700, 30};
redrover::DifferentialDrive drive(hal,
                                  redrover::MotorChannel(25, 26, 27),
                                  redrover::MotorChannel(32, 33, 14),
                                  geometry);

redrover::UltrasonicHCSR04 range(hal, 5, 18, "front_range");
redrover::DigitalSensor bumper(hal, 19, "front_bumper");

redrover::SensorRegistry sensors;
redrover::Node node(transport, drive, sensors, hal);

}  // namespace

void setup() {
    Serial.begin(115200);

    WiFi.mode(WIFI_STA);
    WiFi.begin(REDROVER_WIFI_SSID, REDROVER_WIFI_PASS);
    // Bounded wait: the node still comes up without WiFi, so the watchdog and
    // the e-stop work even when the network does not.
    for (uint8_t i = 0; i < 40 && WiFi.status() != WL_CONNECTED; ++i) {
        delay(250);
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("redRover UDP endpoint: ");
        Serial.print(WiFi.localIP());
        Serial.println(":4242");
    } else {
        Serial.println("WiFi unavailable; node is up but unreachable");
    }

    sensors.add(&range);
    sensors.add(&bumper);

    redrover::NodeConfig config;
    // A wireless link drops more often than a cable, so fail safe sooner.
    config.commandTimeoutMs = 400;
    redrover::setName(config.board, "esp32-rover");
    node.configure(config);
    node.begin();
}

void loop() {
    node.spin();
}

#endif  // ESP32
