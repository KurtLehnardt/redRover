"""Visual inspection — gauge reading, fluid level, and anomaly detection.

Targets:
- Analog gauge reading (pressure, temperature dials)
- Fluid level sight glasses
- Oil/coolant puddle detection
- General visual anomalies (loose bolts, corrosion, missing guards)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
    """Analyzes visual data using local VLM (Gemma vision or LLaVA)."""

    def __init__(self, model: str = "gemma3", ollama_host: str = "http://localhost:11434"):
        self.model = model
        self.ollama_host = ollama_host

    async def read_gauge(self, station_id: str, image_path: str) -> GaugeReading | None:
        """Read an analog gauge from an image."""
        import httpx
        import json
        import base64

        image_data = self._load_image_b64(image_path)
        if not image_data:
            return None

        response = await self._query_vision(GAUGE_READING_PROMPT, image_data)
        try:
            data = json.loads(self._extract_json(response))
            return GaugeReading(
                station_id=station_id,
                gauge_id=Path(image_path).stem,
                value=float(data.get("value", 0)),
                unit=data.get("unit", "unknown"),
                min_normal=0.0,  # Set from config
                max_normal=100.0,
                is_in_range=data.get("zone") == "normal",
                confidence=float(data.get("confidence", 0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    async def inspect_floor(self, station_id: str, image_path: str) -> dict:
        """Inspect floor condition for leaks/debris."""
        import json

        image_data = self._load_image_b64(image_path)
        if not image_data:
            return {"floor_condition": "unknown", "confidence": 0}

        response = await self._query_vision(FLOOR_INSPECTION_PROMPT, image_data)
        try:
            return json.loads(self._extract_json(response))
        except json.JSONDecodeError:
            return {"floor_condition": "unknown", "confidence": 0}

    async def detect_anomalies(self, station_id: str, image_path: str) -> dict:
        """General visual anomaly detection."""
        import json

        image_data = self._load_image_b64(image_path)
        if not image_data:
            return {"anomalies_found": False, "confidence": 0}

        response = await self._query_vision(VISUAL_ANOMALY_PROMPT, image_data)
        try:
            return json.loads(self._extract_json(response))
        except json.JSONDecodeError:
            return {"anomalies_found": False, "confidence": 0}

    async def _query_vision(self, prompt: str, image_b64: str) -> str:
        """Query Ollama with a vision prompt and image."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 300},
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def _load_image_b64(self, image_path: str) -> str | None:
        """Load image and convert to base64."""
        import base64

        path = Path(image_path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially markdown-wrapped response."""
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text
