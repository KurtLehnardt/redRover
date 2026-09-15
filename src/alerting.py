"""Alerting system — notifications for critical faults."""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime

import httpx

from .ai.fusion import FusedDiagnosis, OverallHealth

logger = logging.getLogger("redRover.alerting")

# Ordered least to most severe.
_HEALTH_RANK = {
    OverallHealth.HEALTHY: 0,
    OverallHealth.MONITOR: 1,
    OverallHealth.WARNING: 2,
    OverallHealth.CRITICAL: 3,
}


class AlertManager:
    """Sends alerts when faults exceed severity thresholds.

    A patrol shares one instance; constructing a fresh manager per request was
    why ``/api/alerts`` always returned an empty list.  Use
    :func:`get_alert_manager` to reach the process-wide instance.
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        min_health: str = "warning",
        history_limit: int = 200,
    ):
        self.webhook_url = webhook_url or None
        try:
            self.min_health = OverallHealth(min_health)
        except ValueError:
            logger.warning("Unknown alerting.min_health=%r, defaulting to 'warning'", min_health)
            self.min_health = OverallHealth.WARNING
        self._alert_history: deque[dict] = deque(maxlen=history_limit)

    @classmethod
    def from_config(cls, config) -> AlertManager:
        return cls(
            webhook_url=config.alerting.webhook_url,
            min_health=config.alerting.min_health,
            history_limit=config.alerting.history_limit,
        )

    async def evaluate(self, diagnosis: FusedDiagnosis) -> bool:
        """Alert if the diagnosis meets the configured threshold.

        Returns True when an alert was raised.
        """
        if _HEALTH_RANK[diagnosis.overall_health] < _HEALTH_RANK[self.min_health]:
            return False
        await self._send_alert(diagnosis, level=diagnosis.overall_health.value.upper())
        return True

    async def _send_alert(self, diagnosis: FusedDiagnosis, level: str) -> None:
        """Send alert via logging and optional webhook."""
        alert = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "station_id": diagnosis.station_id,
            "health": diagnosis.overall_health.value,
            "confidence": diagnosis.overall_confidence,
            "faults": diagnosis.correlated_faults,
            "correlation_tags": diagnosis.correlation_tags,
            "recommendation": diagnosis.recommendation,
            "priority": diagnosis.priority,
            "inference_mode": diagnosis.inference_mode,
        }
        self._alert_history.append(alert)

        logger.warning(
            "ALERT [%s] Station %s: %s (P%d) — %s",
            level, diagnosis.station_id,
            ", ".join(diagnosis.correlated_faults) or "unspecified",
            diagnosis.priority,
            diagnosis.recommendation,
        )

        if self.webhook_url:
            await self._post_webhook(alert)

    async def _post_webhook(self, alert: dict) -> None:
        """POST alert to webhook endpoint (Slack, Teams, etc.)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.webhook_url, json=alert)
                response.raise_for_status()
                logger.info("Webhook delivered: %s", response.status_code)
        except Exception as e:
            logger.error("Webhook delivery failed: %s", e)

    @property
    def recent_alerts(self) -> list[dict]:
        """Return the retained alert history, newest last."""
        return list(self._alert_history)

    def clear(self) -> None:
        self._alert_history.clear()


_manager: AlertManager | None = None


def get_alert_manager(config=None) -> AlertManager:
    """Return the process-wide alert manager, creating it on first use."""
    global _manager
    if _manager is None:
        if config is None:
            from .config import load_config
            config = load_config()
        _manager = AlertManager.from_config(config)
    return _manager


def reset_alert_manager() -> None:
    """Drop the process-wide manager (used by tests)."""
    global _manager
    _manager = None
