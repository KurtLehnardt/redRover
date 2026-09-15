"""Canonical fault vocabulary.

Every layer — modality analysers, the fusion engine, the database, the
dashboard — must name a fault the same way.  They previously did not: the
database stored composite narrative labels like
``mechanical_failure_with_heating`` while the rule engine compared against
modality labels like ``bearing_fault``, so trend escalation could never match
its own history.  This module is the single source of truth.
"""

from __future__ import annotations

from enum import Enum


class FaultCode(str, Enum):
    """Canonical fault identifiers persisted to the database."""

    NORMAL = "normal"
    BEARING_FAULT = "bearing_fault"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LOOSENESS = "looseness"
    AIR_LEAK = "air_leak"
    GAS_LEAK = "gas_leak"
    ELECTRICAL_ARCING = "electrical_arcing"
    METAL_FRICTION = "metal_friction"
    HOTSPOT = "hotspot"
    OVERHEATING = "overheating"
    GAUGE_OUT_OF_RANGE = "gauge_out_of_range"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | None) -> FaultCode:
        """Map an arbitrary string onto a canonical code, never raising."""
        if not value:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        try:
            return cls(text)
        except ValueError:
            pass
        # Vibration module uses finer-grained bearing races; they collapse to
        # a single actionable code.
        if "bearing" in text:
            return cls.BEARING_FAULT
        if "arc" in text:
            return cls.ELECTRICAL_ARCING
        if "friction" in text:
            return cls.METAL_FRICTION
        if "leak" in text:
            return cls.AIR_LEAK
        if "overheat" in text:
            return cls.OVERHEATING
        if "hotspot" in text or "hot_spot" in text:
            return cls.HOTSPOT
        if "align" in text:
            return cls.MISALIGNMENT
        if "balance" in text:
            return cls.IMBALANCE
        if "loose" in text:
            return cls.LOOSENESS
        return cls.UNKNOWN


# Correlation tags describe *relationships between* faults.  They are narrative
# and are stored separately from the codes so they never pollute trend matching.
class CorrelationTag(str, Enum):
    MECHANICAL_WITH_HEATING = "mechanical_failure_with_heating"
    LEAK_OR_ELECTRICAL = "air_leak_or_electrical"
    MULTI_MODAL_AGREEMENT = "multi_modal_agreement"
    TRENDING_WORSE = "trending_worse"


# Faults whose verdict depends on a measurement band the sensor may not cover.
BANDWIDTH_DEPENDENT = {
    FaultCode.BEARING_FAULT: "bearing_analysis_available",
    FaultCode.AIR_LEAK: "supports_ultrasonic",
    FaultCode.GAS_LEAK: "supports_ultrasonic",
}
