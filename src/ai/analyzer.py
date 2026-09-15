"""Local AI vibration analysis using Ollama + Gemma.

Single-modality analysis. The patrol path uses
:class:`~src.ai.fusion.FusionAnalyzer` instead; this remains for tools that
want a vibration-only verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..sensors.vibration import FaultType, VibrationSample, extract_features
from .ollama import OllamaClient, OllamaUnavailable

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Result of AI vibration analysis."""

    station_id: str
    fault_type: FaultType
    confidence: float
    severity: str  # "none", "incipient", "moderate", "severe"
    recommendation: str
    reasoning: str


SYSTEM_PROMPT = """You are a vibration analysis expert for industrial predictive maintenance.
You receive vibration signal features from a machine and must diagnose its condition.

Respond ONLY with valid JSON in this exact format:
{
    "fault_type": "normal|bearing_inner_race|bearing_outer_race|bearing_ball|misalignment|looseness|imbalance",
    "confidence": 0.0 to 1.0,
    "severity": "none|incipient|moderate|severe",
    "recommendation": "brief actionable recommendation",
    "reasoning": "brief technical explanation citing specific features"
}

Key diagnostic rules:
- High kurtosis (>4) + high crest factor → bearing fault (impulsive)
- Strong 2x shaft frequency energy → misalignment
- Dominant 1x with low harmonics → imbalance
- Many harmonics + sub-harmonics → looseness
- RMS < 2.0 mm/s with low kurtosis → normal operation
- Bearing faults: check envelope spectrum for defect frequencies (BPFO, BPFI, BSF)
"""


class VibrationAnalyzer:
    """Analyzes vibration data using a local LLM."""

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

    async def analyze(self, sample: VibrationSample) -> DiagnosisResult:
        """Run AI diagnosis on a vibration sample."""
        features = extract_features(sample)
        prompt = self._build_prompt(sample.station_id, features)
        try:
            data = await self._client.chat_json(SYSTEM_PROMPT, prompt)
        except OllamaUnavailable as exc:
            logger.info("vibration analyzer fell back: %s", exc)
            return DiagnosisResult(
                station_id=sample.station_id,
                fault_type=FaultType.NORMAL,
                confidence=0.0,
                severity="unknown",
                recommendation="AI unavailable. Manual inspection recommended.",
                reasoning=str(exc)[:200],
            )

        try:
            fault_type = FaultType(data.get("fault_type", "normal"))
        except ValueError:
            fault_type = FaultType.NORMAL

        return DiagnosisResult(
            station_id=sample.station_id,
            fault_type=fault_type,
            confidence=float(data.get("confidence", 0.0)),
            severity=str(data.get("severity", "unknown")),
            recommendation=str(data.get("recommendation", "")),
            reasoning=str(data.get("reasoning", "")),
        )

    def _build_prompt(self, station_id: str, features: dict) -> str:
        """Build the analysis prompt with extracted features."""

        def fmt(key: str, digits: int = 4) -> str:
            value = features.get(key)
            return "NOT MEASURED" if value is None else f"{value:.{digits}f}"

        return f"""Analyze the following vibration data from machine station {station_id}:

Signal Statistics:
- Sample rate: {features.get("sample_rate_hz", 0):.0f} Hz
- Bearing analysis available: {features.get("bearing_analysis_available")}
- RMS amplitude: {fmt("rms")}
- Peak amplitude: {fmt("peak")}
- Crest factor: {fmt("crest_factor", 2)}
- Kurtosis: {fmt("kurtosis", 2)}
- Dominant frequency: {fmt("dominant_frequency_hz", 1)} Hz

Frequency Band Energy (NOT MEASURED means above the sensor Nyquist limit):
- 0-100 Hz: {fmt("energy_0_100hz", 6)}
- 100-500 Hz: {fmt("energy_100_500hz", 6)}
- 500-1000 Hz: {fmt("energy_500_1000hz", 6)}
- 1000-2000 Hz: {fmt("energy_1000_2000hz", 6)}

Diagnose this machine's condition."""
