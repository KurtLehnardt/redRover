# redRover

**Mobile Vibration Analyst for Predictive Maintenance**

A Sphero RVR+ robot that autonomously patrols factory floors, measuring machine vibration signatures at each station. Local AI (no cloud dependency) classifies bearing health, shaft alignment, and mechanical looseness — replacing $100K in fixed sensors and $50K/year in analyst visits.

## Why "redRover"?

Combines **red list** (a facility's maintenance and equipment priority list) with **rover** (autonomous mobile platform).

## Architecture

```
[Sphero RVR+] <--UART/BLE--> [Compute Platform]
                                  ├── Gemma 3/4 (vibration reasoning)
                                  ├── IMU / contact microphone
                                  ├── Patrol scheduler
                                  └── Local dashboard (WiFi AP)
```

## Compute Options

| Platform | Use Case |
|----------|----------|
| MacBook Pro M1 | Development, POC, tethered operation |
| Jetson Orin Nano 16GB | Untethered deployment on robot |

## Quick Start

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull local AI model
ollama pull gemma3

# Run with simulated data (no hardware needed)
python -m src.main --simulate

# Run dashboard
python -m src.dashboard.app
```

## Project Structure

```
src/
├── rover/       # RVR+ motor control, waypoint navigation
├── sensors/     # Vibration data acquisition (IMU + contact mic)
├── ai/          # Local LLM inference, spectrogram classification
├── scheduler/   # Patrol route planning + timing
└── dashboard/   # Local web UI (FastAPI + HTMX)
config/          # Machine waypoints, alert thresholds
models/          # Trained vibration classifiers
data/            # Sample datasets (CWRU bearing data)
scripts/         # Setup scripts, provisioning
tests/           # Unit + integration tests
```

## Vibration Fault Detection

The system classifies these fault types from vibration signatures:

- **Normal** — healthy baseline
- **Bearing wear** — inner/outer race defects, ball defects
- **Shaft misalignment** — angular or parallel
- **Mechanical looseness** — structural or component
- **Imbalance** — mass imbalance in rotating components

## Hardware Requirements

- Sphero RVR+ (with UART expansion port)
- USB accelerometer or contact microphone
- Compute: MacBook (dev) or Jetson Orin Nano (deploy)

## License

MIT
