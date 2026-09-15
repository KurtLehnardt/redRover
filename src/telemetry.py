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


def init_telemetry(
    service_name: str = "redrover",
    endpoint: str | None = None,
    enabled: bool = True,
    export_interval_ms: int = 5000,
    console_export: bool = False,
    environment: str = "development",
) -> None:
    """Initialize OpenTelemetry providers with graceful fallback.

    With no OTLP ``endpoint`` and ``console_export=False`` (the default), only
    the Prometheus reader is installed — no exporter writes to stdout, so the
    patrol UI stays readable.  Set ``console_export=True`` to get the old
    behaviour when debugging instrumentation.
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
        logger.info("Prometheus metrics exporter enabled (scrape /metrics)")
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

    atexit.register(shutdown_telemetry)
    logger.info("Telemetry initialized: endpoint=%s", endpoint or "local-only")


def shutdown_telemetry(timeout_millis: int = 5000) -> None:
    """Flush and stop exporter threads.

    Without this the BatchSpanProcessor and the periodic metric reader keep
    background threads alive after a patrol finishes, and buffered spans are
    lost on exit.
    """
    global _initialized, _tracer_provider, _meter_provider

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
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_meter(name: str = "redrover") -> metrics.Meter:
    """Get a meter instance."""
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
