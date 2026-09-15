# redRover portable firmware

Runs the redRover rover role on **any Arduino-class controller** — Uno, Nano,
Mega, ESP32, RP2040/Pico, Teensy, STM32 — and speaks the same protocol the
Python host already drives. Swap the chassis, keep the patrol.

```
   your sketch                     the library                    the host
 ┌──────────────┐          ┌──────────────────────┐         ┌────────────────┐
 │ pins, motors │  ──────▶ │ SensorRegistry       │ ◀─────▶ │ FirmwareRover  │
 │ sensor list  │          │ IDriveBase           │  COBS   │ fusion engine  │
 │ 40 lines     │          │ Node (watchdog,      │ +CRC16  │ dashboard      │
 └──────────────┘          │       e-stop)        │         │ ROS 2 bridge   │
                           └──────────────────────┘         └────────────────┘
```

## Why this exists

Two reasons, one practical and one that changes what the system can measure.

**Portability.** `RoverController` *was* the Sphero BLE protocol. Anything that
was not an RVR+ could not run a patrol. The node here implements the same
contract over a serial link, so the rover becomes a component you can replace.

**Bandwidth.** The RVR+ streams its IMU at tens of Hz. Bearing defect
frequencies and their resonance carriers live between roughly 1 and 5 kHz, so
that robot *cannot* do bearing analysis — no amount of host-side cleverness
fixes a sensor that never saw the band. An Uno reading an analog accelerometer
samples at kHz. The descriptor reports that rate, the host believes it, and
bearing verdicts stop being suppressed.

## Quick start

```bash
pip install platformio

pio run -d firmware                      # build for every board
pio run -d firmware -e uno -t upload     # flash an Uno
pio test -d firmware -e native           # 55 host-side unit tests, no board

pio device monitor -b 115200             # watch the link
```

Then point the host at it:

```toml
# config/default.toml
[rover]
connection = "serial"
serial_port = ""        # empty auto-detects
serial_baud = 115200
max_speed_mps = 0.45    # calibrate this — see below

[sensors]
sensor_type = "firmware"
```

```bash
python -m src.main --real
```

## Supported boards

| Board | Env | SRAM | Notes |
|---|---|---|---|
| Arduino Uno / Nano | `uno`, `nanoatmega328` | 2 KB | The baseline. 6 sensors, 64-byte frames. |
| Arduino Mega 2560 | `mega2560` | 8 KB | 16 sensors; room for a mecanum base. |
| ESP32 | `esp32dev`, `esp32-wifi` | 320 KB | WiFi/UDP transport, FPU. |
| RP2040 / Pico | `pico` | 264 KB | Fast ADC; good for kHz vibration. |
| Teensy 4.0 / 4.1 | `teensy40`, `teensy41` | 1 MB | 600 MHz; the fastest sampling option. |
| STM32 Blue Pill | `bluepill` | 20 KB | |

The core has **no dynamic allocation, no exceptions, no `Arduino.h`** — which
is both why it fits on a 2 KB part and why the same code is unit-tested on your
laptop.

## Adding a sensor

A driver is one class with two methods. Nothing else changes: not the host, not
the protocol, not the dashboard. The board describes the sensor, and the host
builds its pipeline from that description.

```cpp
class Mlx90614 : public redrover::SensorBase {
public:
    Mlx90614() {
        descriptor_.kind = redrover::SensorKind::Temperature;
        descriptor_.unit = redrover::Unit::Celsius;
        descriptor_.channels = 1;
        descriptor_.scaleExp = -2;      // values are centidegrees
        descriptor_.rateHz = 4;
        redrover::setName(descriptor_.name, "object_temp");
    }

    bool begin() override { Wire.begin(); return true; }

    bool read(redrover::SensorSample& out) override {
        Wire.beginTransmission(0x5A);
        Wire.write(0x07);
        if (Wire.endTransmission(false) != 0) return false;   // no ACK: say nothing
        if (Wire.requestFrom(0x5A, 3) != 3) return false;
        const uint16_t raw = Wire.read() | (Wire.read() << 8);
        Wire.read();
        out.channels = 1;
        out.values[0] = raw * 2 - 27315;   // 0.02 K/count, K -> centi-C
        return true;
    }
};
```

```cpp
Mlx90614 objectTemp;
sensors.add(&objectTemp);   // the entire integration step
```

`examples/AddASensor` is this, complete and buildable.

If your driver keeps state that an ISR also writes -- an encoder count, a
capture buffer -- read it inside a `CriticalSection`:

```cpp
int32_t count() const {
    redrover::CriticalSection guard(hal_);   // 32-bit reads are not atomic on AVR
    return count_;
}
```

Three rules the interface enforces, because they are the ones that cause silent
wrong answers:

1. **`read()` returns `false` when there is no fresh reading.** The node then
   publishes nothing. Returning `true` with a zeroed struct would put a
   fabricated measurement in the database.
2. **`begin()` returns `false` when the hardware is absent.** The registry
   marks the sensor failed, reports it in its descriptor, and never schedules
   it, so the host sees "sensor down" instead of a stream of zeros.
3. **`rateHz` is the rate you can actually sustain.** The host decides whether
   a fault band is observable from this number.

Values are integers scaled by `scaleExp` (`value = raw × 10^scaleExp`). Integer
transport keeps an AVR free of floating point and makes the wire format exact.

## Drive bases

| Class | Chassis | Holonomic |
|---|---|---|
| `DifferentialDrive` | two-wheel tank | no |
| `MecanumDrive` | four mecanum wheels | yes |

Both accept the same `DriveCommand`, so the host does not branch on chassis
type. A differential base **refuses** a lateral command rather than ignoring
it — a robot that quietly drops the sideways component ends up somewhere the
planner did not ask for.

`MotorChannel` covers the three common H-bridge wirings: two direction pins plus
PWM (L298N, TB6612), one direction pin plus PWM (DRV8871), and PWM-only.

### Calibrate `maxWheelMmPerS`

Every velocity command is scaled by it, so a guess here is a systematic error
in everything downstream — including the map.

> Run one wheel at full duty for five seconds. Measure the distance. Divide
> by five. That number, in mm/s, is `maxWheelMmPerS`.

Set `minEffectiveDuty` to the duty at which the motors stop buzzing and start
turning; commands below it are lifted rather than sent as a stall.

## Safety

The firmware holds these regardless of what the host does — which matters,
because "the host crashed" is exactly when they are needed.

- **Command watchdog.** Motors stop if no command arrives within
  `commandTimeoutMs` (default 500 ms). A kicked-out USB cable leaves the robot
  stationary, not running at its last commanded speed.
- **Latching e-stop.** Once engaged, drive commands are refused with
  `AckCode::EStopActive` until explicitly cleared.
- **Bounded work per loop.** Frame parsing and sample publishing are capped, so
  a flood of input cannot starve the watchdog.
- **CRC on every frame.** A corrupted drive command is dropped, not acted on.
- **Resynchronisation.** COBS framing means a receiver that joins mid-stream,
  or survives a glitch, recovers at the very next delimiter.

## Protocol

Little-endian, COBS-framed, CRC-16/CCITT-FALSE. Delimiter is `0x00`, which
cannot occur inside a frame.

```
+-----+------+-----+-----+----------+---------+
| ver | type | seq | len | payload  | crc16   |
|  1  |  1   |  1  |  1  |   len    |    2    |
+-----+------+-----+-----+----------+---------+
         CRC covers ver..payload
```

| Host → device | | Device → host | |
|---|---|---|---|
| `0x01` Hello | handshake | `0x81` HelloAck | version, sensor count, limits, board |
| `0x02` Describe | request descriptors | `0x82` Descriptor | one per sensor |
| `0x03` SetStream | rate + enable mask | `0x83` Sample | id, timestamp, values |
| `0x04` Drive | linear / angular / lateral | `0x85` Status | state, e-stop, battery, uptime |
| `0x05` DriveRaw | per-wheel per-mil | `0x86` Log | level + text |
| `0x06` Stop | | `0x87` Ack | of-type + code |
| `0x07` EStop | engage / clear | | |
| `0x08` SetAux | LEDs, servos, relays | | |
| `0x09` Ping | watchdog keepalive | | |
| `0x0A` ReadOnce | poll one sensor | | |

**Angular velocity is in milliradians per second**, not millidegrees. An
`int16` of millidegrees saturates at 32.7 °/s — slower than any real robot
turns. Milliradians give ±1877 °/s, and `v = ω·r` needs no π constant when ω is
in radians. The native unit test suite caught this; see
`test/test_drive/test_main.cpp`.

### Heading control

The firmware is a **velocity** interface: it has no heading loop of its own.
The host's `drive_with_heading(speed, heading)` converts heading error into an
angular rate and issues a single non-blocking command, so the chassis curves
onto the heading while still moving. With an odometry sensor registered the
error is measured; without one the heading is integrated from the commanded
rate and the pose stays flagged as estimated.

Register a sensor with `SensorKind::Odometry` reporting `x, y, heading` to close
that loop properly.

The Python mirror lives in `src/rover/wire.py`. `tests/test_wire.py` replays
golden frames emitted by `tools/gen_vectors.cpp`, so the two implementations
cannot drift apart without a test failing.

## Layout

```
firmware/
├── platformio.ini              board matrix
├── src/main.cpp                selects one example
├── lib/RedRover/src/
│   ├── redRover.h              umbrella header
│   ├── redrover/
│   │   ├── Types.h             SensorKind, Unit, descriptors, DriveCommand
│   │   ├── Hal.h               the one place the core touches a pin
│   │   ├── ISensor.h           implement this to add a sensor
│   │   ├── IDriveBase.h        implement this to add a chassis
│   │   ├── ITransport.h        implement this to add a link
│   │   ├── Protocol.h          frame layout and payload codecs
│   │   ├── Node.h              the loop: dispatch, stream, watchdog, e-stop
│   │   ├── drive/              differential, mecanum, motor channels
│   │   ├── sensors/            analog, digital, ultrasonic, accel, encoder
│   │   ├── transport/          serial, UDP
│   │   ├── platform/           Arduino HAL
│   │   └── testing/            FakeHal, LoopbackTransport
│   └── *.cpp
├── examples/                   Arduino IDE sketches
├── test/                       55 native unit tests
└── tools/gen_vectors.cpp       golden frames for the Python mirror
```

## Testing

```bash
pio test -d firmware -e native
```

55 tests covering COBS and CRC round trips and corruption, frame encode/decode,
registry scheduling (including the 49-day `millis()` rollover), differential and
mecanum kinematics, saturation scaling, the command watchdog, the e-stop latch,
stream resynchronisation after a truncated frame, sensor unit conversions, and
the interrupt guard on the encoder count. All on the host, with no board
attached.
