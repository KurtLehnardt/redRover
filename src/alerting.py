"""Alerting system — notifications for critical faults."""

import logging
from datetime import datetime

import httpx

from .ai.fusion import FusedDiagnosis, OverallHealth

logger = logging.getLogger("redRover.alerting")


class AlertManager:
    """Sends alerts when faults exceed severity thresholds."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url
        self._alert_history: list[dict] = []

    async def evaluate(self, diagnosis: FusedDiagnosis) -> None:
        """Evaluate a diagnosis and alert if needed."""
        if diagnosis.overall_health == OverallHealth.CRITICAL:
            await self._send_alert(diagnosis, level="CRITICAL")
        elif diagnosis.overall_health == OverallHealth.WARNING:
            await self._send_alert(diagnosis, level="WARNING")

    async def _send_alert(self, diagnosis: FusedDiagnosis, level: str) -> None:
        """Send alert via logging and optional webhook."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "station_id": diagnosis.station_id,
            "health": diagnosis.overall_health.value,
            "confidence": diagnosis.overall_confidence,
            "faults": diagnosis.correlated_faults,
            "recommendation": diagnosis.recommendation,
            "priority": diagnosis.priority,
        }
        self._alert_history.append(alert)

        logger.warning(
            "ALERT [%s] Station %s: %s (P%d) — %s",
            level, diagnosis.station_id,
            ", ".join(diagnosis.correlated_faults),
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
        """Return last 50 alerts."""
        return self._alert_history[-50:]
