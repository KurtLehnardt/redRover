// PlatformIO entry point.
//
// Each example is a header that defines setup() and loop(); the build selects
// one with a -D flag (see platformio.ini). The same headers back the Arduino
// IDE sketches under examples/, so there is one copy of each example.
#if defined(REDROVER_EXAMPLE_MECANUM)
#include "redrover/examples/MecanumRover.h"
#elif defined(REDROVER_EXAMPLE_ESP32_WIFI)
#include "redrover/examples/Esp32WiFiRover.h"
#elif defined(REDROVER_EXAMPLE_ADD_SENSOR)
#include "redrover/examples/AddASensor.h"
#else
#include "redrover/examples/BasicRover.h"
#endif
