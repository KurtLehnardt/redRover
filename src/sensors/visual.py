"""Visual inspection — gauge reading, fluid level, and anomaly detection.

Targets:
- Analog gauge reading (pressure, temperature dials)
- Fluid level sight glasses
- Oil/coolant puddle detection
- General visual anomalies (loose bolts, corrosion, missing guards)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..ai.ollama import OllamaClient, OllamaUnavailable

logger = logging.getLogger(__name__)


class VisualFaultType(str, Enum):
    NORMAL = "normal"
    GAUGE_OUT_OF_RANGE = "gauge_out_of_range"
    LOW_FLUID = "low_fluid_level"
    LEAK_DETECTED = "leak_on_floor"
    VISUAL_ANOMALY = "visual_anomaly"


@dataclass
class GaugeReading:
    """Result of reading an analog gauge."""
    station_id: str
    gauge_id: str
    value: float
    unit: str
    min_normal: float
    max_normal: float
    is_in_range: bool
    confidence: float


@dataclass
class VisualInspection:
    """Result of visual inspection at a station."""
    station_id: str
    timestamp: float
    image_path: str | None
    gauge_readings: list[GaugeReading]
    fluid_levels: list[dict]
    anomalies: list[dict]
    floor_condition: str  # "clean", "oil_puddle", "coolant_puddle", "debris"


GAUGE_READING_PROMPT = """You are an industrial gauge reading system. Analyze this image of an analog gauge or meter.

Identify:
1. The type of gauge (pressure, temperature, flow, level)
2. The current reading (numeric value)
3. The unit of measurement
4. Whether the reading appears to be in the normal/green zone or in warning/danger zone

Respond ONLY with valid JSON:
{
    "gauge_type": "pressure|temperature|flow|level|unknown",
    "value": <numeric reading>,
    "unit": "psi|bar|celsius|fahrenheit|gpm|percent",
    "zone": "normal|warning|danger",
    "confidence": 0.0 to 1.0
}
"""

FLOOR_INSPECTION_PROMPT = """You are an industrial floor inspection system. Analyze this image of a factory floor near a machine.

Look for:
1. Oil or coolant puddles/drips
2. Water accumulation
3. Debris or loose parts
4. Staining patterns indicating chronic leaks

Respond ONLY with valid JSON:
{
    "floor_condition": "clean|oil_puddle|coolant_puddle|water|debris|staining",
    "severity": "none|minor|moderate|severe",
    "description": "brief description of what you see",
    "confidence": 0.0 to 1.0
}
"""

VISUAL_ANOMALY_PROMPT = """You are an industrial visual inspection system. Analyze this image of factory equipment.

Look for:
1. Loose or missing bolts/fasteners
2. Visible corrosion or rust
3. Cracked or damaged components
4. Missing safety guards
5. Unusual discoloration (heat damage, chemical exposure)
6. Misaligned components

Respond ONLY with valid JSON:
{
    "anomalies_found": true|false,
    "anomalies": [
        {"type": "description", "severity": "low|medium|high", "location": "description"}
    ],
    "overall_condition": "good|fair|poor|critical",
    "confidence": 0.0 to 1.0
}
"""


class VisualAnalyzer:
    """Analyzes visual data using a local VLM (Gemma vision or LLaVA)."""

    def __init__(
        self,
        model: str = "gemma3",
        ollama_host: str = "http://localhost:11434",
        client: OllamaClient | None = None,
    ):
        self.model = model
        self.ollama_host = ollama_host
        self._client = client or OllamaClient(host=ollama_host, model=model)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def read_gauge(self, station_id: str, image_path: str) -> GaugeReading | None:
        """Read an analog gauge from an image."""
        data = await self._ask(GAUGE_READING_PROMPT, image_path)
        if data is None:
            return None
        try:
            return GaugeReading(
                station_id=station_id,
                gauge_id=Path(image_path).stem,
                value=float(data.get("value", 0)),
                unit=str(data.get("unit", "unknown")),
                min_normal=0.0,  # Set from config
                max_normal=100.0,
                is_in_range=data.get("zone") == "normal",
                confidence=float(data.get("confidence", 0)),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("gauge reading unparseable for %s: %s", image_path, exc)
            return None

    async def inspect_floor(self, station_id: str, image_path: str) -> dict:
        """Inspect floor condition for leaks/debris."""
        data = await self._ask(FLOOR_INSPECTION_PROMPT, image_path)
        return data or {"floor_condition": "unknown", "confidence": 0}

    async def detect_anomalies(self, station_id: str, image_path: str) -> dict:
        """General visual anomaly detection."""
        data = await self._ask(VISUAL_ANOMALY_PROMPT, image_path)
        return data or {"anomalies_found": False, "confidence": 0}

    async def _ask(self, prompt: str, image_path: str) -> dict | None:
        image_b64 = self._load_image_b64(image_path)
        if not image_b64:
            logger.warning("visual: image not found at %s", image_path)
            return None
        try:
            return await self._client.vision_json(prompt, image_b64)
        except OllamaUnavailable as exc:
            logger.info("visual analysis unavailable: %s", exc)
            return None

    @staticmethod
    def _load_image_b64(image_path: str) -> str | None:
        """Load an image and convert it to base64."""
        path = Path(image_path)
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode("utf-8")
