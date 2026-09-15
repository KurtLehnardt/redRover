"""Configuration loading, precedence, and validation."""

from __future__ import annotations

import pytest

from src.config import Settings, load_config


def test_route_is_loaded_from_toml():
    """The patrol route is configuration, not source code."""
    config = load_config()
    assert config.route.waypoints
    assert all(w.station_id for w in config.route.waypoints)


def test_env_overrides_the_toml_file(monkeypatch):
    """A value set in the environment must win over the checked-in file.

    Regression: load_config() passes the TOML as init kwargs, and pydantic
    ranks init above env by default, so REDROVER_* overrides were ignored for
    any key the file happened to set — which is every key that matters.
    """
    baseline = load_config()
    assert baseline.dashboard.port == 8080

    monkeypatch.setenv("REDROVER_DASHBOARD__PORT", "9999")
    monkeypatch.setenv("REDROVER_DASHBOARD__AUTH_TOKEN", "from-env")
    monkeypatch.setenv("REDROVER_SIMULATION__TIME_SCALE", "0")

    config = load_config()
    assert config.dashboard.port == 9999
    assert config.dashboard.auth_token == "from-env"
    assert config.simulation.time_scale == 0.0


def test_speed_must_be_a_fraction():
    with pytest.raises(ValueError, match="rover.speed"):
        Settings(rover={"speed": 1.5})


def test_time_scale_must_be_non_negative():
    with pytest.raises(ValueError, match="time_scale"):
        Settings(simulation={"time_scale": -1.0})


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    config = load_config(tmp_path / "does-not-exist.toml")
    assert config.dashboard.host == "127.0.0.1"
    assert config.route.waypoints == []
