"""Multi-modal sensor fusion — combines vibration, acoustic, thermal, and visual
into a unified facility health diagnosis.

The fusion engine correlates signals across modalities to increase confidence
and catch faults that single-sensor analysis would miss.

Two invariants hold throughout:

* Fault names are :class:`~src.ai.faults.FaultCode` values everywhere, so a
  diagnosis can be matched against its own persisted history.
* A modality that could not observe a band reports ``None`` for it, and the
  rules treat that as "not measured", never as "measured zero".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from ..sensors.acoustic import AcousticSample, extract_acoustic_features
from ..sensors.thermal import (
    ThermalFaultType,
    ThermalFrame,
    classify_thermal_severity,
    extract_thermal_features,
)
from ..sensors.vibration import VibrationSample, extract_features
from ..telemetry import get_meter, get_tracer
from .faults import CorrelationTag, FaultCode
from .ollama import OllamaClient, OllamaUnavailable

logger = logging.getLogger(__name__)


class OverallHealth(str, Enum):
    HEALTHY = "healthy"
    MONITOR = "monitor"  # Early signs, watch closely
    WARNING = "warning"  # Action needed within weeks
    CRITICAL = "critical"  # Action needed within days/hours


SEVERITY_TO_HEALTH = {
    "none": OverallHealth.HEALTHY,
    "incipient": OverallHealth.MONITOR,
    "moderate": OverallHealth.WARNING,
    "severe": OverallHealth.CRITICAL,
    "critical": OverallHealth.CRITICAL,
}

HEALTH_TO_PRIORITY = {
    OverallHealth.CRITICAL: 1,
    OverallHealth.WARNING: 2,
    OverallHealth.MONITOR: 3,
    OverallHealth.HEALTHY: 4,
}

ESCALATION = {
    OverallHealth.MONITOR: OverallHealth.WARNING,
    OverallHealth.WARNING: OverallHealth.CRITICAL,
}


@dataclass
class ModalityResult:
    """Result from a single sensor modality."""

    modality: str
    fault_detected: bool
    fault_type: str
    confidence: float
    severity: str
    details: dict = field(default_factory=dict)
    # Bands this modality could not observe at all at its configured rate.
    unobservable: list[str] = field(default_factory=list)
    # Bands observed only in part (the value under-reports the true energy).
    partial: list[str] = field(default_factory=list)

    @property
    def code(self) -> FaultCode:
        return FaultCode.parse(self.fault_type)


@dataclass
class FusedDiagnosis:
    """Combined diagnosis from all sensor modalities."""

    station_id: str
    overall_health: OverallHealth
    overall_confidence: float
    modality_results: list[ModalityResult]
    correlated_faults: list[str]  # canonical FaultCode values
    recommendation: str
    priority: int  # 1=highest priority (critical), 4=lowest (healthy)
    reasoning: str
    inference_mode: str = "llm"  # "llm", "rule_based", "degraded"
    correlation_tags: list[str] = field(default_factory=list)
    # Modalities that were requested but produced no usable data.
    unobservable: list[str] = field(default_factory=list)

    @property
    def primary_fault(self) -> str:
        return self.correlated_faults[0] if self.correlated_faults else FaultCode.NORMAL.value


FUSION_PROMPT = """You are a multi-modal predictive maintenance expert. You receive data from
multiple sensors at a single machine station and must provide a unified diagnosis.

Sensor correlation rules:
- Bearing fault (vibration) + elevated temperature (thermal) + high-freq noise (acoustic) = CONFIRMED bearing failure, critical priority
- Vibration normal + ultrasonic energy detected (acoustic) = AIR LEAK near machine, warning priority
- Misalignment (vibration) + elevated temperature = misalignment causing overheating, critical priority
- Single sensor anomaly with low confidence = MONITOR, schedule re-check
- Multiple sensors agreeing = HIGH confidence in diagnosis

A reading marked "NOT MEASURED" means the sensor could not observe that band.
Never treat it as evidence of absence.

Use ONLY these fault identifiers in correlated_faults:
normal, bearing_fault, misalignment, imbalance, looseness, air_leak, gas_leak,
electrical_arcing, metal_friction, hotspot, overheating, gauge_out_of_range

Respond ONLY with valid JSON:
{
    "overall_health": "healthy|monitor|warning|critical",
    "correlated_faults": ["list of confirmed fault identifiers"],
    "recommendation": "specific actionable recommendation",
    "priority": 1-4 (1=critical, 4=healthy),
    "reasoning": "brief explanation of cross-sensor correlation"
}
"""


class FusionAnalyzer:
    """Fuses multi-modal sensor data into unified health assessment."""

    def __init__(
        self,
        model: str = "gemma3",
        ollama_host: str = "http://localhost:11434",
        client: OllamaClient | None = None,
    ):
        self.model = model
        self.ollama_host = ollama_host
        self._client = client or OllamaClient(host=ollama_host, model=model)
        self._tracer = get_tracer("redrover.fusion")
        self._meter = get_meter("redrover.fusion")
        self._inference_duration = self._meter.create_histogram(
            "redrover.ai.inference_seconds",
            description="Duration of fusion inference",
            unit="s",
        )
        self._inference_mode_counter = self._meter.create_counter(
            "redrover.ai.inference_count",
            description="Inference count by mode (llm, rule_based, degraded)",
        )
        self._confidence_hist = self._meter.create_histogram(
            "redrover.ai.confidence",
            description="Overall confidence of fused diagnosis",
        )
        self._fault_counter = self._meter.create_counter(
            "redrover.ai.faults_detected",
            description="Faults detected by type",
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- public API ---------------------------------------------------------

    async def analyze(
        self,
        station_id: str,
        vibration: VibrationSample | None = None,
        acoustic: AcousticSample | None = None,
        thermal: ThermalFrame | None = None,
        station_history: list[dict] | None = None,
    ) -> FusedDiagnosis:
        """Run fused analysis across all available sensor data."""
        start = time.time()
        modality_results: list[ModalityResult] = []

        if vibration is not None:
            modality_results.append(self._analyze_vibration(vibration))
        if acoustic is not None:
            modality_results.append(self._analyze_acoustic(acoustic))
        if thermal is not None:
            modality_results.append(self._analyze_thermal(thermal))

        if len(modality_results) >= 2:
            result = await self._ai_fusion(
                station_id, modality_results, station_history=station_history
            )
        elif len(modality_results) == 1:
            result = self._single_modality(station_id, modality_results[0])
        else:
            result = FusedDiagnosis(
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

        result.unobservable = sorted({band for mr in modality_results for band in mr.unobservable})

        duration = time.time() - start
        attrs = {"station.id": station_id, "inference.mode": result.inference_mode}
        self._inference_duration.record(duration, attrs)
        self._inference_mode_counter.add(1, {"inference.mode": result.inference_mode})
        self._confidence_hist.record(result.overall_confidence, attrs)
        for fault in result.correlated_faults:
            self._fault_counter.add(1, {"fault.type": fault, "station.id": station_id})

        return result

    def _single_modality(self, station_id: str, r: ModalityResult) -> FusedDiagnosis:
        health = SEVERITY_TO_HEALTH.get(r.severity, OverallHealth.MONITOR)
        return FusedDiagnosis(
            station_id=station_id,
            overall_health=health,
            overall_confidence=r.confidence,
            modality_results=[r],
            correlated_faults=[r.code.value] if r.fault_detected else [],
            recommendation=(
                f"Single-sensor detection: {r.code.value}" if r.fault_detected else "All clear"
            ),
            priority=HEALTH_TO_PRIORITY[health],
            reasoning=f"Based on {r.modality} only",
            inference_mode="degraded",
        )

    # -- per-modality rules -------------------------------------------------

    @staticmethod
    def _num(features: dict, key: str) -> float | None:
        """Read a numeric feature, preserving None for unobservable bands."""
        value = features.get(key)
        return None if value is None else float(value)

    def _analyze_vibration(self, sample: VibrationSample) -> ModalityResult:
        """Rule-based vibration assessment."""
        features = extract_features(sample)
        kurtosis = features["kurtosis"]
        crest = features["crest_factor"]
        rms = features["rms"]
        low_energy = self._num(features, "energy_0_100hz")
        dominant = features["dominant_frequency_hz"]
        bearing_ok = bool(features.get("bearing_analysis_available"))

        unobservable = [
            key for key, value in features.items() if key.startswith("energy_") and value is None
        ]
        partial = list(features.get("partial_bands", []))
        if not bearing_ok:
            unobservable.append("bearing_resonance_band")

        impulsive = (kurtosis > 4 and crest > 4) or kurtosis > 6 or crest > 4.0
        if impulsive:
            if not bearing_ok:
                # Impulsive energy is real, but at this sample rate it cannot be
                # attributed to a bearing.  Report what was actually observed.
                return ModalityResult(
                    "vibration",
                    True,
                    FaultCode.LOOSENESS.value,
                    0.50,
                    "incipient",
                    details=features,
                    unobservable=unobservable,
                    partial=partial,
                )
            if kurtosis > 6:
                severity, confidence = "severe", 0.90
            elif kurtosis > 4 and crest > 4:
                severity, confidence = "moderate", 0.85
            else:
                severity, confidence = "incipient", 0.70
            return ModalityResult(
                "vibration",
                True,
                FaultCode.BEARING_FAULT.value,
                confidence,
                severity,
                details=features,
                unobservable=unobservable,
                partial=partial,
            )

        if low_energy is not None and low_energy > 0.5 and dominant < 100:
            if low_energy > 1.0:
                return ModalityResult(
                    "vibration",
                    True,
                    FaultCode.MISALIGNMENT.value,
                    0.70,
                    "moderate",
                    details=features,
                    unobservable=unobservable,
                    partial=partial,
                )
            return ModalityResult(
                "vibration",
                True,
                FaultCode.IMBALANCE.value,
                0.65,
                "incipient",
                details=features,
                unobservable=unobservable,
                partial=partial,
            )

        if rms > 1.5:
            return ModalityResult(
                "vibration",
                True,
                FaultCode.LOOSENESS.value,
                0.60,
                "moderate",
                details=features,
                unobservable=unobservable,
                partial=partial,
            )

        # A clean result is only as confident as the bands we could see.
        confidence = 0.90 if bearing_ok else 0.60
        return ModalityResult(
            "vibration",
            False,
            FaultCode.NORMAL.value,
            confidence,
            "none",
            details=features,
            unobservable=unobservable,
            partial=partial,
        )

    def _analyze_acoustic(self, sample: AcousticSample) -> ModalityResult:
        """Rule-based acoustic assessment."""
        features = extract_acoustic_features(sample)
        ultrasonic = self._num(features, "ultrasonic_energy")
        rms_variance = features["rms_variance"]
        audible_high = self._num(features, "acoustic_audible_high")

        unobservable = [
            key for key, value in features.items() if key.startswith("acoustic_") and value is None
        ]
        partial = list(features.get("partial_bands", []))
        if ultrasonic is None:
            unobservable.append("ultrasonic_band")

        if ultrasonic is not None:
            if ultrasonic > 0.05:
                return ModalityResult(
                    "acoustic",
                    True,
                    FaultCode.AIR_LEAK.value,
                    0.80,
                    "moderate",
                    details=features,
                    unobservable=unobservable,
                    partial=partial,
                )
            if ultrasonic > 0.02:
                return ModalityResult(
                    "acoustic",
                    True,
                    FaultCode.AIR_LEAK.value,
                    0.60,
                    "incipient",
                    details=features,
                    unobservable=unobservable,
                    partial=partial,
                )

        if rms_variance > 0.01:
            return ModalityResult(
                "acoustic",
                True,
                FaultCode.ELECTRICAL_ARCING.value,
                0.70,
                "moderate",
                details=features,
                unobservable=unobservable,
                partial=partial,
            )

        if audible_high is not None and audible_high > 0.001:
            return ModalityResult(
                "acoustic",
                True,
                FaultCode.METAL_FRICTION.value,
                0.55,
                "incipient",
                details=features,
                unobservable=unobservable,
                partial=partial,
            )

        confidence = 0.85 if ultrasonic is not None else 0.55
        return ModalityResult(
            "acoustic",
            False,
            FaultCode.NORMAL.value,
            confidence,
            "none",
            details=features,
            unobservable=unobservable,
            partial=partial,
        )

    def _analyze_thermal(self, frame: ThermalFrame) -> ModalityResult:
        """Rule-based thermal assessment."""
        features = extract_thermal_features(frame)
        fault_type, severity = classify_thermal_severity(frame)

        if fault_type == ThermalFaultType.OVERHEATING:
            return ModalityResult(
                "thermal",
                True,
                FaultCode.OVERHEATING.value,
                0.90,
                severity,
                details=features,
            )
        if fault_type == ThermalFaultType.HOTSPOT:
            return ModalityResult(
                "thermal",
                True,
                FaultCode.HOTSPOT.value,
                0.75,
                severity,
                details=features,
            )
        return ModalityResult(
            "thermal",
            False,
            FaultCode.NORMAL.value,
            0.85,
            "none",
            details=features,
        )

    # -- fusion -------------------------------------------------------------

    async def _ai_fusion(
        self,
        station_id: str,
        results: list[ModalityResult],
        station_history: list[dict] | None = None,
    ) -> FusedDiagnosis:
        """Use the LLM to correlate cross-modal signals, falling back to rules."""
        prompt = self._build_fusion_prompt(station_id, results, station_history=station_history)

        try:
            data = await self._client.chat_json(FUSION_PROMPT, prompt)
            health = OverallHealth(str(data.get("overall_health", "healthy")).lower())
            codes = [
                FaultCode.parse(f).value
                for f in data.get("correlated_faults", [])
                if FaultCode.parse(f) is not FaultCode.UNKNOWN
            ]
            priority = int(data.get("priority", HEALTH_TO_PRIORITY[health]))
            if priority not in (1, 2, 3, 4):
                priority = HEALTH_TO_PRIORITY[health]

            diagnosis = FusedDiagnosis(
                station_id=station_id,
                overall_health=health,
                overall_confidence=self._compute_fused_confidence(results),
                modality_results=results,
                correlated_faults=codes or self._detected_codes(results),
                recommendation=str(data.get("recommendation", "")),
                priority=priority,
                reasoning=str(data.get("reasoning", "")),
                inference_mode="llm",
            )
        except (OllamaUnavailable, ValueError, KeyError, TypeError) as exc:
            logger.info("LLM fusion unavailable (%s) — using rule-based fusion", exc)
            diagnosis = self._rule_based_fusion(station_id, results)

        return self._apply_trend(diagnosis, station_history)

    @staticmethod
    def _detected_codes(results: list[ModalityResult]) -> list[str]:
        seen: list[str] = []
        for r in results:
            if r.fault_detected and r.code.value not in seen:
                seen.append(r.code.value)
        return seen

    def _rule_based_fusion(
        self,
        station_id: str,
        results: list[ModalityResult],
    ) -> FusedDiagnosis:
        """Fallback fusion without LLM — pure rule-based correlation."""
        faults = [r for r in results if r.fault_detected]
        n_faults = len(faults)

        if n_faults == 0:
            return FusedDiagnosis(
                station_id=station_id,
                overall_health=OverallHealth.HEALTHY,
                overall_confidence=self._compute_fused_confidence(results),
                modality_results=results,
                correlated_faults=[],
                recommendation="All sensors nominal",
                priority=4,
                reasoning="No faults detected across any modality",
                inference_mode="rule_based",
            )

        has_vibration = any(r.modality == "vibration" and r.fault_detected for r in results)
        has_thermal = any(r.modality == "thermal" and r.fault_detected for r in results)
        has_acoustic = any(r.modality == "acoustic" and r.fault_detected for r in results)

        tags: list[str] = []
        if has_vibration and has_thermal:
            tags.append(CorrelationTag.MECHANICAL_WITH_HEATING.value)
        if has_acoustic and not has_vibration:
            tags.append(CorrelationTag.LEAK_OR_ELECTRICAL.value)
        if n_faults >= 2:
            tags.append(CorrelationTag.MULTI_MODAL_AGREEMENT.value)

        if n_faults >= 2:
            health = (
                OverallHealth.CRITICAL
                if any(r.severity in ("severe", "critical") for r in faults)
                else OverallHealth.WARNING
            )
        else:
            health = (
                OverallHealth.WARNING
                if faults[0].severity in ("moderate", "severe")
                else OverallHealth.MONITOR
            )

        codes = self._detected_codes(results)
        return FusedDiagnosis(
            station_id=station_id,
            overall_health=health,
            overall_confidence=self._compute_fused_confidence(results),
            modality_results=results,
            correlated_faults=codes,
            recommendation=self._generate_recommendation(faults),
            priority=HEALTH_TO_PRIORITY[health],
            reasoning=f"Faults from {n_faults} modalities: {', '.join(codes)}",
            inference_mode="rule_based",
            correlation_tags=tags,
        )

    def _apply_trend(
        self,
        diagnosis: FusedDiagnosis,
        station_history: list[dict] | None,
    ) -> FusedDiagnosis:
        """Escalate when the same canonical fault persists across patrols.

        History rows carry canonical :class:`FaultCode` values because
        ``Database.log_diagnosis`` writes ``primary_fault``; matching therefore
        compares like with like.
        """
        if not station_history or not diagnosis.correlated_faults:
            return diagnosis

        current = {FaultCode.parse(f) for f in diagnosis.correlated_faults}
        recent = station_history[-3:]
        historical = {
            FaultCode.parse(row.get("fault_type")) for row in recent if row.get("fault_type")
        }
        historical.discard(FaultCode.NORMAL)
        historical.discard(FaultCode.UNKNOWN)

        if not (current & historical):
            return diagnosis

        escalated = ESCALATION.get(diagnosis.overall_health)
        if escalated is not None:
            diagnosis.overall_health = escalated
            diagnosis.priority = HEALTH_TO_PRIORITY[escalated]
        diagnosis.correlation_tags = [
            *diagnosis.correlation_tags,
            CorrelationTag.TRENDING_WORSE.value,
        ]
        diagnosis.recommendation = (
            f"{diagnosis.recommendation} TRENDING: fault persistent across "
            f"{len(current & historical)} of the last {len(recent)} patrols."
        ).strip()
        return diagnosis

    def _generate_recommendation(self, faults: list[ModalityResult]) -> str:
        """Generate actionable recommendation from fault list."""
        codes = {r.code for r in faults}
        if FaultCode.BEARING_FAULT in codes:
            if any(r.modality == "thermal" and r.fault_detected for r in faults):
                return (
                    "URGENT: Bearing failure with thermal confirmation. "
                    "Schedule immediate replacement."
                )
            return "Bearing wear detected. Schedule replacement within 2 weeks."
        if codes & {FaultCode.AIR_LEAK, FaultCode.GAS_LEAK}:
            return (
                "Compressed air leak detected. Locate and seal — "
                "estimated $3K-8K/year energy waste."
            )
        if FaultCode.ELECTRICAL_ARCING in codes:
            return "URGENT: Electrical arcing detected. De-energize and inspect immediately."
        if codes & {FaultCode.OVERHEATING, FaultCode.HOTSPOT}:
            return "Thermal anomaly detected. Check cooling, lubrication, and load conditions."
        if FaultCode.MISALIGNMENT in codes:
            return "Shaft misalignment detected. Schedule laser alignment."
        return "Anomaly detected. Schedule manual inspection."

    def _compute_fused_confidence(self, results: list[ModalityResult]) -> float:
        """Bayesian-inspired confidence boost when multiple sensors agree."""
        if not results:
            return 0.0
        fault_results = [r for r in results if r.fault_detected]
        if not fault_results:
            return min(r.confidence for r in results)
        base = max(r.confidence for r in fault_results)
        bonus = 0.05 * (len(fault_results) - 1)
        return min(base + bonus, 0.99)

    def _build_fusion_prompt(
        self,
        station_id: str,
        results: list[ModalityResult],
        station_history: list[dict] | None = None,
    ) -> str:
        """Build prompt for LLM fusion analysis."""
        lines = [f"Station: {station_id}\n\nSensor Readings:"]
        for r in results:
            status = (
                f"FAULT: {r.code.value} (severity: {r.severity}, confidence: {r.confidence:.0%})"
                if r.fault_detected
                else "NORMAL"
            )
            lines.append(f"\n[{r.modality.upper()}] {status}")
            for key, val in list(r.details.items())[:6]:
                if val is None:
                    lines.append(f"  {key}: NOT MEASURED")
                elif isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")
            if r.unobservable:
                lines.append(f"  NOT MEASURED: {', '.join(sorted(set(r.unobservable)))}")
            if r.partial:
                lines.append(
                    f"  PARTIALLY MEASURED (value under-reports): "
                    f"{', '.join(sorted(set(r.partial)))}"
                )

        if station_history:
            lines.append("\n\nHistorical Trend:")
            lines.append("Previous measurements at this station:")
            for h in station_history:
                ts = (h.get("measured_at") or "?")[:16]
                rms = h.get("rms") or 0
                kurtosis = h.get("kurtosis") or 0
                fault = h.get("fault_type") or "unknown"
                conf = h.get("confidence") or 0
                lines.append(
                    f"  {ts} RMS={rms:.2f} Kurtosis={kurtosis:.1f} -> {fault} ({conf:.0%})"
                )
            lines.append("  [current reading]")
            lines.append("\nAnalyze the trend: is this getting worse, stable, or improving?")

        lines.append("\nProvide a fused diagnosis correlating all sensor data.")
        return "\n".join(lines)
