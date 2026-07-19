"""Local AI vibration analysis using Ollama + Gemma."""

import json
import httpx
from dataclasses import dataclass

from ..sensors.vibration import VibrationSample, FaultType, extract_features


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
    """Analyzes vibration data using local LLM."""

    def __init__(self, model: str = "gemma3", ollama_host: str = "http://localhost:11434"):
        self.model = model
        self.ollama_host = ollama_host

    async def analyze(self, sample: VibrationSample) -> DiagnosisResult:
        """Run AI diagnosis on a vibration sample."""
        features = extract_features(sample)

        prompt = self._build_prompt(sample.station_id, features)

        response = await self._query_ollama(prompt)
        return self._parse_response(sample.station_id, response)

    def _build_prompt(self, station_id: str, features: dict) -> str:
        """Build the analysis prompt with extracted features."""
        return f"""Analyze the following vibration data from machine station {station_id}:

Signal Statistics:
- RMS amplitude: {features['rms']:.4f}
- Peak amplitude: {features['peak']:.4f}
- Crest factor: {features['crest_factor']:.2f}
- Kurtosis: {features['kurtosis']:.2f}
- Dominant frequency: {features['dominant_frequency_hz']:.1f} Hz

Frequency Band Energy:
- 0-100 Hz: {features['energy_0_100hz']:.6f}
- 100-500 Hz: {features['energy_100_500hz']:.6f}
- 500-1000 Hz: {features['energy_500_1000hz']:.6f}
- 1000-2000 Hz: {features['energy_1000_2000hz']:.6f}

Diagnose this machine's condition."""

    async def _query_ollama(self, prompt: str) -> str:
        """Query local Ollama instance."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def _parse_response(self, station_id: str, response: str) -> DiagnosisResult:
        """Parse LLM JSON response into structured result."""
        # Extract JSON from response (handle markdown code blocks)
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return clean JSON
            return DiagnosisResult(
                station_id=station_id,
                fault_type=FaultType.NORMAL,
                confidence=0.0,
                severity="unknown",
                recommendation="AI response could not be parsed. Manual inspection recommended.",
                reasoning=f"Raw response: {response[:200]}",
            )

        return DiagnosisResult(
            station_id=station_id,
            fault_type=FaultType(data.get("fault_type", "normal")),
            confidence=float(data.get("confidence", 0.0)),
            severity=data.get("severity", "unknown"),
            recommendation=data.get("recommendation", ""),
            reasoning=data.get("reasoning", ""),
        )
