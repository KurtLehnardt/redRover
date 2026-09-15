# redRover

**Multi-Modal Facility Health Robot**

A Sphero RVR+ (or any Arduino-class robot — see [`firmware/`](firmware/)) that
autonomously patrols a facility, measuring vibration, acoustic emission, and
thermal signatures at each machine station. Local AI (no cloud dependency)
correlates the modalities into a single health verdict, escalating when a fault
persists across patrols. An optional piggyback drone deploys for overhead
inspection.

## Why "redRover"?

Combines **red list** (a facility's maintenance and equipment priority list)
with **rover** (autonomous mobile platform).

## Architecture

```
[Rover]  <--BLE / UART / USB serial / WiFi-->  [Compute platform]
                                                 |
                                                 +-- rover backends   (Sphero, firmware, simulator)
                                                 +-- sensor sources   (vibration, acoustic, thermal)
                                                 +-- fusion engine    (local LLM, rule-based fallback)
                                                 +-- patrol scheduler
                                                 +-- SQLite history   (trend escalation)
                                                 +-- dashboard        (FastAPI, token-guarded)
                                                 +-- drone orchestrator
```

## Which robot?

The patrol talks to a `RoverBackend`, never to a specific robot. Pick one with
`[rover].connection`:

| `connection` | Robot | Sensor rate | Bearing analysis |
|---|---|---|---|
| `ble` / `uart` | Sphero RVR+ | ~50 Hz | no — see bandwidth below |
| `serial` | Any Arduino / ESP32 / RP2040 / Teensy running [`firmware/`](firmware/) | kHz | yes |
| any, with `--simulate` | Simulator | n/a | n/a |

Adding a platform means implementing `src/rover/backends/base.py`, not editing
the patrol. There is also a [ROS 2 bridge](ros2/) that runs a firmware rover as
a standard ROS node (`/cmd_vel` in, `sensor_msgs` out).

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Run a full patrol against the simulator — no hardware, no AI needed
python -m src.main --simulate

# Faster: collapse the demo pacing
REDROVER_SIMULATION__TIME_SCALE=0 python -m src.main --simulate

# Local AI (optional; the fusion engine falls back to rules without it)
ollama pull gemma3           # must match [ai].model in config/default.toml

# Dashboard (see "Dashboard security" before exposing it)
REDROVER_DASHBOARD__AUTH_TOKEN=$(openssl rand -hex 16) python -m src.dashboard.app

pytest
```

## Simulated vs. real

`--simulate` and `--real` are strictly separated, and nothing in between.

| | `--simulate` | `--real` |
|---|---|---|
| Vibration | synthetic fault signatures | rover IMU stream (`[sensors].sensor_type = "imu"`) |
| Acoustic | synthetic emissions | system microphone via `sounddevice` |
| Thermal | synthetic frames | MLX90640 over I2C, if configured |
| Missing hardware | n/a | recorded as a **sensor failure** — never filled in with synthetic data |

Every measurement row carries `source_name` and `simulated`, so a simulated run
is distinguishable from a real one in the database and in the dashboard.

## Sensor bandwidth (read this before trusting a bearing verdict)

Fault signatures live in specific frequency bands, and a sensor that cannot
reach a band has not measured it.

| Fault family | Needs | RVR+ IMU (~50 Hz) | Laptop mic (44.1/48 kHz) | Firmware rover |
|---|---|---|---|---|
| Imbalance, misalignment, looseness | < 200 Hz | yes | — | yes |
| Bearing defects (BPFO/BPFI/BSF) | >= 2 kHz | **no** | — | yes (kHz ADC) |
| Ultrasonic air/gas leak | >= 96 kHz | — | **no** | with an ultrasonic mic |

No amount of host-side analysis recovers a band the sensor never saw. A
firmware rover reading an analog accelerometer is the supported path to real
bearing work — see [`firmware/README.md`](firmware/README.md).

The code enforces this rather than papering over it:

- Bands entirely above Nyquist are reported as `None` (**not measured**), never
  as `0.0`.
- Bands straddling Nyquist are reported with their value **and** listed in
  `partial_bands`, because the number under-reports the true energy.
- A bearing verdict is **suppressed** below 2 kHz: the impulsive energy is
  still reported, but it is not attributed to a bearing.
- A clean result from a bandwidth-limited sensor carries lower confidence than
  a clean result from a capable one.

## Fault vocabulary

Every layer names a fault with the same `FaultCode` (`src/ai/faults.py`), which
is what lets a diagnosis be matched against its own persisted history and
escalated when it recurs. Relationships *between* faults
(`mechanical_failure_with_heating`, `trending_worse`) are stored separately as
correlation tags so they never pollute that matching.

## Dashboard security

`/api/remap`, `/api/demo-patrol`, and `/api/estop` physically move the robot.

- All three require `[dashboard].auth_token`, sent as `X-RedRover-Token`,
  `Authorization: Bearer <token>`, or the session cookie. Prefer the
  environment: `REDROVER_DASHBOARD__AUTH_TOKEN=...`
- The browser UI unlocks by posting the token to `/api/session`, which returns
  it as an **HttpOnly, SameSite=strict** cookie — out of reach of page scripts,
  and unusable by a cross-site request. Until then the page shows an unlock
  form instead of buttons that would only 401.
- The page loads **no third-party scripts**. An appliance that advertises no
  cloud dependency should not go blank when the factory network does.
- With no token configured they return **503** — disabled, not open.
- CORS is restricted to `[dashboard].allowed_origins`; `"*"` is rejected at
  config-load time.
- The default bind is `127.0.0.1`.
- `POST /api/estop` latches an emergency stop; motors stop and further drive
  commands are refused until `clear_estop()`.

## Metrics

A patrol runs in its own process, so it serves its own scrape endpoint on
`[telemetry].prometheus_port` (9464 by default); the dashboard serves
`/metrics` from its own app on the dashboard port. `prometheus.yml` scrapes
both — with only the dashboard target, every patrol panel in
`grafana/dashboard.json` has no data source at all.

## Configuration

`config/default.toml` holds everything, including the patrol route — adding a
station does not require touching source. Any value can be overridden from the
environment with a `REDROVER_` prefix and `__` for nesting, and **the
environment wins over the file**:

```bash
REDROVER_DASHBOARD__PORT=9000
REDROVER_ALERTING__WEBHOOK_URL=https://hooks.slack.com/services/...
REDROVER_SIMULATION__TIME_SCALE=0
```

Calibrate `[rover].max_speed_mps` for your chassis: drive at `speed = 1.0` for
five seconds, measure the distance, divide by five. Navigation timing is
derived from it.

## Project structure

```
src/
├── rover/       Motor control, navigation, Sphero V2 BLE protocol
├── mapping/     Occupancy grid (ray-cast) + frontier explorer
├── sensors/     Acquisition sources, DSP, simulator
├── ai/          Fault vocabulary, fusion engine, Ollama client
├── drone/       Drone control, missions, orchestration, precision landing
├── scheduler/   Patrol timing and quiet hours
├── dashboard/   FastAPI UI + control API
└── rover/
    ├── backends/  One interface, several robots
    └── wire.py    Python mirror of the firmware protocol
firmware/        Portable C++ firmware for Arduino-class controllers
ros2/            ROS 2 bridge package
config/          Route, thresholds, connection settings
scripts/         CLI entry points (live patrol, room mapping)
tests/           Unit, integration, and regression tests
```

## Hardware

- **Rover**: Sphero RVR+ (BLE or UART), or any Arduino/ESP32/RP2040/Teensy
  robot running the firmware in [`firmware/`](firmware/)
- **Vibration**: rover IMU (low-frequency faults only) or a kHz-rate
  accelerometer on a firmware rover
- **Acoustic**: any system microphone; an ultrasonic mic (>= 96 kHz) for leak
  detection
- **Thermal**: MLX90640 (32x24) over I2C
- **Drone** (optional): DJI Tello EDU or Bitcraze Crazyflie 2.1
- **Compute**: MacBook (dev) or Jetson Orin Nano (deploy)

Optional dependencies are listed at the bottom of `requirements.txt`; install
only what your hardware needs.

## Testing

```bash
pytest                          # Python suite, offline, ~17s
pytest -m llm                   # opt in to tests that need a live Ollama
pio test -d firmware -e native  # 55 firmware tests, no board attached
```

The simulator draws from a seeded generator that `conftest.py` pins, so a
failing signal assertion reproduces instead of being a coin flip. Seed a
simulated patrol yourself with `src.sensors.simulator.set_seed()`.

The Python suite never reaches Ollama, real hardware, or `data/redRover.db`:
those are stubbed or redirected to a temp directory by `tests/conftest.py`.

`tests/test_wire.py` replays golden frames generated by the C++ firmware
(`firmware/tools/gen_vectors.cpp`), so the Python and C++ implementations of
the protocol cannot drift apart without a test failing.

## License

MIT
