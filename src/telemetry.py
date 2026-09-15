"""OpenTelemetry setup for redRover — traces, metrics, and structured logging."""

from __future__ import annotations

import atexit
import logging
from functools import wraps
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import StatusCode

logger = logging.getLogger("redRover.telemetry")

_initialized = False
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None
_prometheus_server = None


def init_telemetry(
    service_name: str = "redrover",
    endpoint: str | None = None,
    enabled: bool = True,
    export_interval_ms: int = 5000,
    console_export: bool = False,
    environment: str = "development",
    prometheus_port: int = 0,
) -> None:
    """Initialize OpenTelemetry providers with graceful fallback.

    With no OTLP ``endpoint`` and ``console_export=False`` (the default), only
    the Prometheus reader is installed — no exporter writes to stdout, so the
    patrol UI stays readable.  Set ``console_export=True`` to get the old
    behaviour when debugging instrumentation.

    ``prometheus_port`` starts an HTTP server that exposes those metrics.  A
    ``PrometheusMetricReader`` on its own only *collects*: without something
    serving a scrape endpoint the metrics exist in memory and nothing can ever
    read them, which is why every patrol panel in the Grafana dashboard sat
    empty.  The dashboard process leaves this at 0 because it serves
    ``/metrics`` from its own app.
    """
    global _initialized, _tracer_provider, _meter_provider
    if _initialized:
        return
    _initialized = True

    if not enabled:
        logger.info("Telemetry disabled by config")
        return

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": environment,
    })

    # --- Traces ---
    _tracer_provider = TracerProvider(resource=resource)
    span_exporter = _create_span_exporter(endpoint, console_export)
    if span_exporter is not None:
        _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(_tracer_provider)

    # --- Metrics ---
    metric_readers = []

    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        metric_readers.append(PrometheusMetricReader())
        if prometheus_port:
            _start_prometheus_server(prometheus_port)
        else:
            logger.info("Prometheus metrics collected (served by the host app)")
    except ImportError:
        logger.info("Prometheus exporter not available")

    metric_exporter = _create_metric_exporter(endpoint, console_export)
    if metric_exporter is not None:
        metric_readers.append(
            PeriodicExportingMetricReader(
                metric_exporter, export_interval_millis=export_interval_ms,
            )
        )

    _meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    metrics.set_meter_provider(_meter_provider)
    if metrics.get_meter_provider() is not _meter_provider:
        # OpenTelemetry installs its global providers once per process. A
        # second init here keeps its own provider (get_meter below prefers it),
        # but anything already holding a meter from the first one reports
        # somewhere this process is not serving.
        logger.warning(
            "a MeterProvider was already installed in this process; metrics "
            "created before this call are exported by the earlier provider"
        )

    atexit.register(shutdown_telemetry)
    logger.info("Telemetry initialized: endpoint=%s", endpoint or "local-only")


def shutdown_telemetry(timeout_millis: int = 5000) -> None:
    """Flush and stop exporter threads.

    Without this the BatchSpanProcessor and the periodic metric reader keep
    background threads alive after a patrol finishes, and buffered spans are
    lost on exit.
    """
    global _initialized, _tracer_provider, _meter_provider, _prometheus_server

    if _prometheus_server is not None:
        try:
            server, thread = _prometheus_server
            server.shutdown()
            thread.join(timeout=2)
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            logger.debug("prometheus server shutdown: %s", exc)
        _prometheus_server = None

    if _tracer_provider is not None:
        try:
            _tracer_provider.force_flush(timeout_millis)
            _tracer_provider.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            logger.debug("tracer provider shutdown: %s", exc)
        _tracer_provider = None

    if _meter_provider is not None:
        try:
            _meter_provider.force_flush(timeout_millis)
            _meter_provider.shutdown(timeout_millis)
        except Exception as exc:  # pragma: no cover - best-effort shutdown
            logger.debug("meter provider shutdown: %s", exc)
        _meter_provider = None

    _initialized = False


def _start_prometheus_server(port: int) -> None:
    """Expose the collected metrics so Prometheus has something to scrape."""
    global _prometheus_server
    if _prometheus_server is not None:
        return
    try:
        from prometheus_client import start_http_server
    except ImportError:
        logger.warning(
            "prometheus_port=%d requested but prometheus_client is not installed; "
            "metrics will be collected and never served", port,
        )
        return
    try:
        _prometheus_server = start_http_server(port)
        logger.info("Prometheus metrics served on http://0.0.0.0:%d/metrics", port)
    except OSError as exc:
        # A second patrol process, or the dashboard already holding the port.
        logger.warning("could not serve Prometheus metrics on port %d: %s", port, exc)


def _create_span_exporter(endpoint: str | None, console_export: bool):
    """Create span exporter — OTLP if configured, console if asked, else none."""
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(endpoint=endpoint)
        except Exception as e:
            logger.warning("OTLP span exporter failed: %s", e)
    return ConsoleSpanExporter() if console_export else None


def _create_metric_exporter(endpoint: str | None, console_export: bool):
    """Create metric exporter — OTLP if configured, console if asked, else none."""
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            return OTLPMetricExporter(endpoint=endpoint)
        except Exception as e:
            logger.warning("OTLP metric exporter failed: %s", e)
    return ConsoleMetricExporter() if console_export else None


def get_tracer(name: str = "redrover") -> trace.Tracer:
    """Get a tracer, preferring the provider this process installed.

    OpenTelemetry's globals are set-once, so after a re-init the global lookup
    still returns the *first* provider. Binding to ours keeps spans going to
    the exporters this process actually configured.
    """
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(name)
    return trace.get_tracer(name)


def get_meter(name: str = "redrover") -> metrics.Meter:
    """Get a meter bound to the provider this process installed.

    Same reason as :func:`get_tracer`: without this, metrics created after a
    second init land in a provider nothing is serving.
    """
    if _meter_provider is not None:
        return _meter_provider.get_meter(name)
    return metrics.get_meter(name)


def traced(span_name: str | None = None, attributes: dict[str, Any] | None = None):
    """Decorator to wrap async functions in a trace span."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as e:
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
