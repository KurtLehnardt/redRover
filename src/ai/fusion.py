"""Multi-modal sensor fusion — combines vibration, acoustic, thermal, and visual
into a unified facility health diagnosis.

The fusion engine correlates signals across modalities to increase confidence
and catch faults that single-sensor analysis would miss.
"""

import json
from dataclasses import dataclass, field
from enum import Enum

import httpx

from ..sensors.vibration import VibrationSample, FaultType, extract_features
from ..sensors.acoustic import AcousticSample, AcousticFaultType, extract_acoustic_features
from ..sensors.thermal import ThermalFrame, ThermalFaultType, extract_thermal_features, classify_thermal_severity


class OverallHealth(str, Enum):
    HEALTHY = "healthy"
    MONITOR = "monitor"  # Early signs, watch closely
    WARNING = "warning"  # Action needed within weeks
    CRITICAL = "critical"  # Action needed within days/hours


@dataclass
class ModalityResult:
    """Result from a single sensor modality."""
    modality: str
    fault_detected: bool
    fault_type: str
    confidence: float
    severity: str
    details: dict = field(default_factory=dict)


@dataclass
class FusedDiagnosis:
    """Combined diagnosis from all sensor modalities."""
    station_id: str
    overall_health: OverallHealth
    overall_confidence: float
    modality_results: list[ModalityResult]
    correlated_faults: list[str]  # Faults confirmed by multiple modalities
    recommendation: str
    priority: int  # 1=highest priority (critical), 4=lowest (healthy)
    reasoning: str
    inference_mode: str = "llm"  # "llm", "rule_based", "degraded"


FUSION_PROMPT = """You are a multi-modal predictive maintenance expert. You receive data from
multiple sensors at a single machine station and must provide a unified diagnosis.

Sensor correlation rules:
- Bearing fault (vibration) + elevated temperature (thermal) + high-freq noise (acoustic) = CONFIRMED bearing failure, critical priority
- Vibration normal + ultrasonic energy detected (acoustic) = AIR LEAK near machine, warning priority
- Misalignment (vibration) + elevated temperature = misalignment causing overheating, critical priority
- Single sensor anomaly with low confidence = MONITOR, schedule re-check
- Multiple sensors agreeing = HIGH confidence in diagnosis

Respond ONLY with valid JSON:
{
    "overall_health": "healthy|monitor|warning|critical",
    "correlated_faults": ["list of confirmed fault types"],
    "recommendation": "specific actionable recommendation",
    "priority": 1-4 (1=critical, 4=healthy),
    "reasoning": "brief explanation of cross-sensor correlation"
}
"""


class FusionAnalyzer:
    """Fuses multi-modal sensor data into unified health assessment."""

    def __init__(self, model: str = "gemma3", ollama_host: str = "http://localhost:11434"):
        self.model = model
        self.ollama_host = ollama_host

    async def analyze(
        self,
        station_id: str,
        vibration: VibrationSample | None = None,
        acoustic: AcousticSample | None = None,
        thermal: ThermalFrame | None = None,
        station_history: list[dict] | None = None,
    ) -> FusedDiagnosis:
        """Run fused analysis across all available sensor data."""
        modality_results = []

        # Process each modality independently first
        if vibration:
            vib_result = self._analyze_vibration(vibration)
            modality_results.append(vib_result)

        if acoustic:
            aco_result = self._analyze_acoustic(acoustic)
            modality_results.append(aco_result)

        if thermal:
            therm_result = self._analyze_thermal(thermal)
            modality_results.append(therm_result)

        # Run AI fusion if we have multiple modalities
        if len(modality_results) >= 2:
            return await self._ai_fusion(station_id, modality_results, station_history=station_history)
        elif len(modality_results) == 1:
            # Single modality — just wrap it
            r = modality_results[0]
            health = self._severity_to_health(r.severity)
            return FusedDiagnosis(
                station_id=station_id,
                overall_health=health,
                overall_confidence=r.confidence,
                modality_results=modality_results,
                correlated_faults=[r.fault_type] if r.fault_detected else [],
                recommendation=f"Single-sensor detection: {r.fault_type}" if r.fault_detected else "All clear",
                priority=self._health_to_priority(health),
                reasoning=f"Based on {r.modality} only",
                inference_mode="degraded",
            )
        else:
            return FusedDiagnosis(
                station_id=station_id,
                overall_health=OverallHealth.HEALTHY,
                overall_confidence=0.0,
                modality_results=[],
                correlated_faults=[],
                recommendation="No sensor data available",
                priority=4,
                reasoning="No modalities provided",
                inference_mode="degraded",
            )

    def _analyze_vibration(self, sample: VibrationSample) -> ModalityResult:
        """Rule-based vibration assessment."""
        features = extract_features(sample)
        kurtosis = features["kurtosis"]
        crest = features["crest_factor"]
        rms = features["rms"]

        if kurtosis > 4 and crest > 4:
            return ModalityResult("vibration", True, "bearing_fault", 0.85, "moderate",
                                  details=features)
        elif kurtosis > 6:
            return ModalityResult("vibration", True, "bearing_fault", 0.90, "severe",
                                  details=features)
        elif features["energy_0_100hz"] > 0.5 and features["dominant_frequency_hz"] < 100:
            # Strong low-frequency — check for misalignment or imbalance
            if features["energy_0_100hz"] > 1.0:
                return ModalityResult("vibration", True, "misalignment", 0.70, "moderate",
                                      details=features)
            else:
                return ModalityResult("vibration", True, "imbalance", 0.65, "incipient",
                                      details=features)
        elif rms > 1.5:
            return ModalityResult("vibration", True, "looseness", 0.60, "moderate",
                                  details=features)
        else:
            return ModalityResult("vibration", False, "normal", 0.90, "none",
                                  details=features)

    def _analyze_acoustic(self, sample: AcousticSample) -> ModalityResult:
        """Rule-based acoustic assessment."""
        features = extract_acoustic_features(sample)
        ultrasonic = features.get("ultrasonic_energy", 0)
        rms_variance = features["rms_variance"]

        if ultrasonic > 0.05:
            return ModalityResult("acoustic", True, "air_leak", 0.80, "moderate",
                                  details=features)
        elif ultrasonic > 0.02:
            return ModalityResult("acoustic", True, "air_leak", 0.60, "incipient",
                                  details=features)
        elif rms_variance > 0.01:
            return ModalityResult("acoustic", True, "electrical_arcing", 0.70, "warning",
                                  details=features)
        elif features.get("acoustic_audible_high", 0) > 0.001:
            return ModalityResult("acoustic", True, "metal_friction", 0.55, "incipient",
                                  details=features)
        else:
            return ModalityResult("acoustic", False, "normal", 0.85, "none",
                                  details=features)

    def _analyze_thermal(self, frame: ThermalFrame) -> ModalityResult:
        """Rule-based thermal assessment."""
        features = extract_thermal_features(frame)
        fault_type, severity = classify_thermal_severity(frame)

        if fault_type == ThermalFaultType.OVERHEATING:
            return ModalityResult("thermal", True, "overheating", 0.90, severity,
                                  details=features)
        elif fault_type == ThermalFaultType.HOTSPOT:
            return ModalityResult("thermal", True, "hotspot", 0.75, severity,
                                  details=features)
        else:
            return ModalityResult("thermal", False, "normal", 0.85, "none",
                                  details=features)

    async def _ai_fusion(
        self,
        station_id: str,
        results: list[ModalityResult],
        station_history: list[dict] | None = None,
    ) -> FusedDiagnosis:
        """Use LLM to correlate cross-modal signals."""
        prompt = self._build_fusion_prompt(station_id, results, station_history=station_history)

        try:
            response = await self._query_ollama(prompt)
            data = json.loads(self._extract_json(response))

            return FusedDiagnosis(
                station_id=station_id,
                overall_health=OverallHealth(data.get("overall_health", "healthy")),
                overall_confidence=self._compute_fused_confidence(results),
                modality_results=results,
                correlated_faults=data.get("correlated_faults", []),
                recommendation=data.get("recommendation", ""),
                priority=int(data.get("priority", 4)),
                reasoning=data.get("reasoning", ""),
                inference_mode="llm",
            )
        except Exception:
            # Fallback: rule-based fusion without LLM
            result = self._rule_based_fusion(station_id, results, station_history=station_history)
            result.inference_mode = "rule_based"
            return result

    def _rule_based_fusion(
        self, station_id: str, results: list[ModalityResult],
        station_history: list[dict] | None = None,
    ) -> FusedDiagnosis:
        """Fallback fusion without LLM — pure rule-based correlation."""
        faults = [r for r in results if r.fault_detected]
        n_faults = len(faults)

        if n_faults == 0:
            return FusedDiagnosis(
                station_id=station_id,
                overall_health=OverallHealth.HEALTHY,
                overall_confidence=min(r.confidence for r in results),
                modality_results=results,
                correlated_faults=[],
                recommendation="All sensors nominal",
                priority=4,
                reasoning="No faults detected across any modality",
                inference_mode="rule_based",
            )

        # Check for corroborating evidence
        has_vibration_fault = any(r.modality == "vibration" and r.fault_detected for r in results)
        has_thermal_fault = any(r.modality == "thermal" and r.fault_detected for r in results)
        has_acoustic_fault = any(r.modality == "acoustic" and r.fault_detected for r in results)

        correlated = []
        if has_vibration_fault and has_thermal_fault:
            correlated.append("mechanical_failure_with_heating")
        if has_acoustic_fault and not has_vibration_fault:
            correlated.append("air_leak_or_electrical")

        if n_faults >= 2:
            health = OverallHealth.CRITICAL if any(r.severity in ("severe", "critical") for r in faults) else OverallHealth.WARNING
        else:
            health = OverallHealth.WARNING if faults[0].severity in ("moderate", "severe") else OverallHealth.MONITOR

        fault_types = [r.fault_type for r in faults]
        recommendation = self._generate_recommendation(faults)

        # Trend escalation: if fault type persists across recent history, escalate
        if station_history:
            current_fault_types = set(fault_types)
            recent_history = station_history[-3:]  # Last 3 readings
            for ft in current_fault_types:
                historical_count = sum(
                    1 for h in recent_history
                    if h.get("fault_type") == ft and h["fault_type"] != "normal"
                )
                if historical_count > 0:
                    # Escalate severity: incipient→moderate, moderate→severe
                    severity_escalation = {
                        OverallHealth.MONITOR: OverallHealth.WARNING,
                        OverallHealth.WARNING: OverallHealth.CRITICAL,
                    }
                    if health in severity_escalation:
                        health = severity_escalation[health]
                    recommendation += " TRENDING: fault persistent across multiple patrols"
                    break  # Only escalate once

        return FusedDiagnosis(
            station_id=station_id,
            overall_health=health,
            overall_confidence=self._compute_fused_confidence(results),
            modality_results=results,
            correlated_faults=correlated or fault_types,
            recommendation=recommendation,
            priority=self._health_to_priority(health),
            reasoning=f"Faults from {n_faults} modalities: {', '.join(fault_types)}",
            inference_mode="rule_based",
        )

    def _generate_recommendation(self, faults: list[ModalityResult]) -> str:
        """Generate actionable recommendation from fault list."""
        if any("bearing" in r.fault_type for r in faults):
            if any(r.modality == "thermal" and r.fault_detected for r in faults):
                return "URGENT: Bearing failure with thermal confirmation. Schedule immediate replacement."
            return "Bearing wear detected. Schedule replacement within 2 weeks."
        if any("air_leak" in r.fault_type for r in faults):
            return "Compressed air leak detected. Locate and seal — estimated $3K-8K/year energy waste."
        if any("arcing" in r.fault_type for r in faults):
            return "URGENT: Electrical arcing detected. De-energize and inspect immediately."
        if any("overheating" in r.fault_type for r in faults):
            return "Thermal anomaly detected. Check cooling, lubrication, and load conditions."
        return "Anomaly detected. Schedule manual inspection."

    def _compute_fused_confidence(self, results: list[ModalityResult]) -> float:
        """Bayesian-inspired confidence boost when multiple sensors agree."""
        fault_results = [r for r in results if r.fault_detected]
        if not fault_results:
            return min(r.confidence for r in results)
        # Multiple confirming sensors boost confidence
        base = max(r.confidence for r in fault_results)
        bonus = 0.05 * (len(fault_results) - 1)
        return min(base + bonus, 0.99)

    def _build_fusion_prompt(
        self, station_id: str, results: list[ModalityResult],
        station_history: list[dict] | None = None,
    ) -> str:
        """Build prompt for LLM fusion analysis."""
        lines = [f"Station: {station_id}\n\nSensor Readings:"]
        for r in results:
            status = f"FAULT: {r.fault_type} (severity: {r.severity}, confidence: {r.confidence:.0%})" if r.fault_detected else "NORMAL"
            lines.append(f"\n[{r.modality.upper()}] {status}")
            # Include key metrics
            for key, val in list(r.details.items())[:5]:
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")

        # Append historical trend context if available
        if station_history:
            lines.append("\n\nHistorical Trend:")
            lines.append("Previous measurements at this station:")
            for h in station_history:
                ts = h.get("measured_at", "?")[:16]  # Trim to minute precision
                rms = h.get("rms", 0) or 0
                kurtosis = h.get("kurtosis", 0) or 0
                fault = h.get("fault_type", "unknown") or "unknown"
                conf = h.get("confidence", 0) or 0
                lines.append(f"  {ts} RMS={rms:.2f} Kurtosis={kurtosis:.1f} -> {fault} ({conf:.0%})")
            lines.append("  [current reading]")
            lines.append("\nAnalyze the trend: is this getting worse, stable, or improving?")

        lines.append("\nProvide a fused diagnosis correlating all sensor data.")
        return "\n".join(lines)

    async def _query_ollama(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "system": FUSION_PROMPT,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 400},
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text

    def _severity_to_health(self, severity: str) -> OverallHealth:
        return {
            "none": OverallHealth.HEALTHY,
            "incipient": OverallHealth.MONITOR,
            "moderate": OverallHealth.WARNING,
            "severe": OverallHealth.CRITICAL,
            "critical": OverallHealth.CRITICAL,
        }.get(severity, OverallHealth.MONITOR)

    def _health_to_priority(self, health: OverallHealth) -> int:
        return {
            OverallHealth.CRITICAL: 1,
            OverallHealth.WARNING: 2,
            OverallHealth.MONITOR: 3,
            OverallHealth.HEALTHY: 4,
        }[health]
