"""Configuration management for redRover.

Values come from ``config/default.toml`` and may be overridden by environment
variables using a double-underscore path, e.g. ``REDROVER_DASHBOARD__PORT=9000``
or ``REDROVER_ALERTING__WEBHOOK_URL=https://hooks.example/...``.  Secrets
therefore never need to live in the checked-in TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class RoverConfig(BaseModel):
    # "ble" / "uart" -> Sphero RVR+; "serial" -> any board running firmware/
    connection: str = "ble"
    # Serial port for connection="serial". Empty means auto-detect.
    serial_port: str = ""
    serial_baud: int = 115200
    speed: float = 0.3
    dwell_time: int = 10
    # Measured ground speed at full throttle; used to convert a drive command
    # into a travel time instead of assuming a fixed 50 cm/s.
    max_speed_mps: float = 1.5
    # Physical half-width, used to place detected walls outside the chassis.
    robot_radius_m: float = 0.12
    # Hard ceiling on any single uninterrupted drive command.
    max_drive_seconds: float = 20.0

    @field_validator("speed")
    @classmethod
    def _speed_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("rover.speed must be between 0.0 and 1.0")
        return v


class SensorConfig(BaseModel):
    sample_rate: int = 4000
    measurement_duration: int = 5
    # "imu" (rover IMU stream), "usb_accel", "contact_mic", "firmware"
    sensor_type: str = "imu"
    # Seconds of accelerometer stream to collect per firmware measurement.
    firmware_capture_seconds: float = 2.0
    imu_period_ms: int = 20
    acoustic_sample_rate: int = 96000
    acoustic_duration: float = 3.0
    microphone_enabled: bool = True
    microphone_device: str = ""
    # "none" or "mlx90640"
    thermal_camera: str = "none"
    ambient_temp_c: float = 22.0
    machine_rpm: float = 1800.0


class AIConfig(BaseModel):
    model: str = "gemma3"
    ollama_host: str = "http://localhost:11434"
    alert_threshold: float = 0.75
    request_timeout_s: float = 120.0


class SchedulerConfig(BaseModel):
    patrol_interval: int = 60
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "06:00"


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    # Shared secret required by every state-changing endpoint.  Empty means
    # those endpoints are disabled rather than open.
    auth_token: str = ""
    # Exact origins allowed to call the API; "*" is rejected because the API
    # can physically drive the robot.
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    reload: bool = False

    @field_validator("allowed_origins")
    @classmethod
    def _no_wildcard(cls, v: list[str]) -> list[str]:
        if "*" in v:
            raise ValueError(
                "dashboard.allowed_origins must not contain '*': the API can "
                "drive the robot. List the Grafana/UI origins explicitly."
            )
        return v


class DroneConfig(BaseModel):
    # "mission_pad" | "aruco" | "plain" -- see src/drone/controller.py
    landing_mode: str = "mission_pad"
    marker_id: int = 42
    min_battery: int = 20


class AlertingConfig(BaseModel):
    webhook_url: str = ""
    min_health: str = "warning"  # "monitor" | "warning" | "critical"
    history_limit: int = 200


class TelemetryConfig(BaseModel):
    enabled: bool = True
    service_name: str = "redrover"
    endpoint: str = ""
    export_interval_ms: int = 5000
    console_export: bool = False
    environment: str = "development"
    # Port for this process to serve /metrics on. 0 disables it; the dashboard
    # leaves it at 0 because its own app already serves /metrics.
    prometheus_port: int = 9464


class DatabaseConfig(BaseModel):
    path: str = "data/redRover.db"


class SimulationConfig(BaseModel):
    """Controls how simulated runs spend wall-clock time.

    ``time_scale`` multiplies every artificial delay in simulate mode: 1.0
    plays a patrol at real-time pace for demos, 0.0 removes the waits entirely
    so the test suite and CI are not billed for dramatic pauses.
    """

    time_scale: float = 1.0

    @field_validator("time_scale")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("simulation.time_scale must be >= 0")
        return v


class WaypointConfig(BaseModel):
    station_id: str
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    machine_type: str = ""
    rpm: float = 1800.0
    has_overhead_equipment: bool = False


class RouteConfig(BaseModel):
    """The patrol route.

    Waypoints declared here are upserted into the ``stations`` table on
    startup, so a real facility is configured in TOML rather than by editing
    ``src/main.py``.
    """

    name: str = "Default Route"
    waypoints: list[WaypointConfig] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REDROVER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let the environment override the TOML file.

        ``load_config`` passes the parsed TOML as init kwargs, and pydantic
        ranks init highest by default — which would make
        ``REDROVER_DASHBOARD__AUTH_TOKEN`` silently lose to whatever the file
        says. Ordering env first is what makes secrets and per-host overrides
        actually work.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    rover: RoverConfig = RoverConfig()
    sensors: SensorConfig = SensorConfig()
    ai: AIConfig = AIConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    dashboard: DashboardConfig = DashboardConfig()
    drone: DroneConfig = DroneConfig()
    alerting: AlertingConfig = AlertingConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    database: DatabaseConfig = DatabaseConfig()
    simulation: SimulationConfig = SimulationConfig()
    route: RouteConfig = RouteConfig()


def default_config_path() -> Path:
    return Path(__file__).parent.parent / "config" / "default.toml"


def load_config(path: Path | None = None) -> Settings:
    """Load configuration from a TOML file, with environment overrides."""
    if path is None:
        path = default_config_path()

    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return Settings(**data)

    return Settings()
