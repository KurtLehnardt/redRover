"""OpenTelemetry setup for redRover — traces, metrics, and structured logging."""

import logging
from functools import wraps
from typing import Any

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode

logger = logging.getLogger("redRover.telemetry")

_initialized = False


def init_telemetry(
    service_name: str = "redrover",
    endpoint: str | None = None,
    enabled: bool = True,
    export_interval_ms: int = 5000,
) -> None:
    """Initialize OpenTelemetry providers with graceful fallback.

    If OTLP endpoint is unreachable, falls back to console exporter.
    If enabled=False, uses no-op providers (zero overhead).
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    if not enabled:
        logger.info("Telemetry disabled by config")
        return

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": "development",
    })

    # --- Traces ---
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = _create_span_exporter(endpoint)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    metric_readers = []

    # Prometheus exporter (exposes /metrics for Grafana scraping)
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        prometheus_reader = PrometheusMetricReader()
        metric_readers.append(prometheus_reader)
        logger.info("Prometheus metrics exporter enabled on :8080/metrics")
    except ImportError:
        logger.info("Prometheus exporter not available, using periodic exporter only")

    # Periodic exporter (OTLP or console)
    metric_exporter = _create_metric_exporter(endpoint)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=export_interval_ms,
    )
    metric_readers.append(metric_reader)

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)

    logger.info("Telemetry initialized: endpoint=%s", endpoint or "console")


def _create_span_exporter(endpoint: str | None):
    """Create span exporter — OTLP if endpoint available, else console."""
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            return OTLPSpanExporter(endpoint=endpoint)
        except Exception as e:
            logger.warning("OTLP span exporter failed, using console: %s", e)
    return ConsoleSpanExporter()


def _create_metric_exporter(endpoint: str | None):
    """Create metric exporter — OTLP if endpoint available, else console."""
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            return OTLPMetricExporter(endpoint=endpoint)
        except Exception as e:
            logger.warning("OTLP metric exporter failed, using console: %s", e)
    return ConsoleMetricExporter()


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
