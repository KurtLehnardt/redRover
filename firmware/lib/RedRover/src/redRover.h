// redRover portable firmware — umbrella header.
//
//   #include <redRover.h>
//
// Brings in the core (types, protocol, node, registry), the Arduino HAL and
// serial transport when building for a board, and the bundled drive bases and
// sensors. Include individual headers instead if you want a smaller build.
#pragma once

#include "redrover/Types.h"
#include "redrover/Hal.h"
#include "redrover/ISensor.h"
#include "redrover/IDriveBase.h"
#include "redrover/ITransport.h"
#include "redrover/Framing.h"
#include "redrover/Protocol.h"
#include "redrover/SensorRegistry.h"
#include "redrover/Node.h"

#include "redrover/drive/MotorChannel.h"
#include "redrover/drive/DifferentialDrive.h"
#include "redrover/drive/MecanumDrive.h"

#include "redrover/sensors/AnalogSensor.h"
#include "redrover/sensors/DigitalSensor.h"
#include "redrover/sensors/UltrasonicHCSR04.h"
#include "redrover/sensors/AnalogAccelerometer.h"
#include "redrover/sensors/QuadratureEncoder.h"

#if defined(ARDUINO)
#include "redrover/platform/ArduinoHal.h"
#include "redrover/transport/SerialTransport.h"
#endif

#if defined(ESP32) || defined(ESP8266)
#include "redrover/transport/UdpTransport.h"
#endif
