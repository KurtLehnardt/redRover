"""Configuration management for redRover."""

import tomllib
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class RoverConfig(BaseModel):
    connection: str = "ble"
    speed: float = 0.3
    dwell_time: int = 10


class SensorConfig(BaseModel):
    sample_rate: int = 4000
    measurement_duration: int = 5
    sensor_type: str = "imu"


class AIConfig(BaseModel):
    model: str = "gemma3"
    ollama_host: str = "http://localhost:11434"
    alert_threshold: float = 0.75


class SchedulerConfig(BaseModel):
    patrol_interval: int = 60
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "06:00"


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class TelemetryConfig(BaseModel):
    enabled: bool = True
    service_name: str = "redrover"
    endpoint: str = ""
    export_interval_ms: int = 5000


class DatabaseConfig(BaseModel):
    path: str = "data/redRover.db"


class Settings(BaseSettings):
    rover: RoverConfig = RoverConfig()
    sensors: SensorConfig = SensorConfig()
    ai: AIConfig = AIConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    dashboard: DashboardConfig = DashboardConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    database: DatabaseConfig = DatabaseConfig()


def load_config(path: Path | None = None) -> Settings:
    """Load configuration from TOML file."""
    if path is None:
        path = Path(__file__).parent.parent / "config" / "default.toml"

    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return Settings(**data)

    return Settings()
