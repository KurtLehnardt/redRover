"""Telemetry must actually be readable, not merely collected."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from src import telemetry


@pytest.fixture(autouse=True)
def _clean_telemetry():
    telemetry.shutdown_telemetry()
    yield
    telemetry.shutdown_telemetry()


def _scrape(port: int) -> str:
    return urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()


def _free_port() -> int:
    """Reserve a port the OS says is free.

    ``prometheus_port = 0`` means "disabled" in config, so it cannot double as
    the ephemeral-port sentinel here.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_metrics_are_served_when_a_port_is_configured():
    """Regression: a PrometheusMetricReader only *collects*.

    Without something serving a scrape endpoint the metrics existed in memory
    and nothing could ever read them, so every patrol panel in
    grafana/dashboard.json had no data source at all.
    """
    pytest.importorskip("prometheus_client")

    port = _free_port()
    telemetry.init_telemetry(
        service_name="redrover-test",
        enabled=True,
        prometheus_port=port,
    )
    if telemetry._prometheus_server is None:
        pytest.skip("prometheus exporter unavailable")

    counter = telemetry.get_meter("redrover.test").create_counter("redrover.test.hits")
    counter.add(3, {"where": "unit-test"})

    body = _scrape(port)
    assert "redrover_test_hits" in body


def test_no_server_is_started_when_the_port_is_zero_by_the_host():
    """The dashboard serves /metrics from its own app and must not double-bind."""
    telemetry.init_telemetry(service_name="redrover-test", enabled=True)
    # prometheus_port defaults to 0 on the function signature, so an embedding
    # host gets collection without a competing server.
    assert telemetry._prometheus_server is None


def test_shutdown_stops_the_metrics_server():
    pytest.importorskip("prometheus_client")

    port = _free_port()
    telemetry.init_telemetry(
        service_name="redrover-test",
        enabled=True,
        prometheus_port=port,
    )
    if telemetry._prometheus_server is None:
        pytest.skip("prometheus exporter unavailable")
    _scrape(port)  # alive

    telemetry.shutdown_telemetry()
    assert telemetry._prometheus_server is None
    with pytest.raises((urllib.error.URLError, OSError)):
        _scrape(port)


def test_patrol_config_defaults_to_serving_metrics():
    """A patrol runs in its own process, so it needs its own scrape endpoint."""
    from src.config import load_config

    assert load_config().telemetry.prometheus_port != 0


def test_prometheus_targets_cover_both_processes():
    """prometheus.yml must scrape the patrol as well as the dashboard."""
    from pathlib import Path

    body = (Path(__file__).resolve().parent.parent / "prometheus.yml").read_text()
    assert "localhost:8080" in body  # dashboard
    assert "localhost:9464" in body  # patrol / scheduler


def test_metrics_go_to_the_provider_this_process_installed():
    """Regression: OpenTelemetry's globals are set-once.

    After a second init, the global lookup still returns the *first* provider,
    so every metric created afterwards was exported by a provider this process
    was not serving — collected, and unreachable.
    """
    pytest.importorskip("prometheus_client")

    # First init, as some earlier component would have done.
    telemetry.init_telemetry(service_name="redrover-first", enabled=True)
    first_provider = telemetry._meter_provider

    # Second init, as a patrol starting inside the same process would do.
    telemetry.shutdown_telemetry()
    port = _free_port()
    telemetry.init_telemetry(
        service_name="redrover-second",
        enabled=True,
        prometheus_port=port,
    )
    if telemetry._prometheus_server is None:
        pytest.skip("prometheus exporter unavailable")

    assert telemetry._meter_provider is not first_provider
    counter = telemetry.get_meter("redrover.second").create_counter("redrover.second.hits")
    counter.add(1)

    assert "redrover_second_hits" in _scrape(port)
