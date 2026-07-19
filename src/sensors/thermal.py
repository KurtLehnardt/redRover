"""Thermal anomaly detection — IR camera for heat-based fault identification.

Targets:
- Electrical cabinet hotspots (blown fuses, loose connections)
- Motor/bearing overheating
- Steam/fluid leaks (thermal contrast)
- HVAC anomalies
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum


class ThermalFaultType(str, Enum):
    NORMAL = "normal"
    HOTSPOT = "hotspot"
    OVERHEATING = "overheating"
    COLD_SPOT = "cold_spot"  # Insulation failure or leak
    THERMAL_GRADIENT = "abnormal_gradient"


@dataclass
class ThermalFrame:
    """A single thermal camera frame (e.g., MLX90640 = 32x24 pixels)."""
    station_id: str
    timestamp: float
    pixels: NDArray[np.float32]  # Temperature in Celsius, shape (height, width)
    ambient_temp: float  # Room ambient temperature

    @property
    def max_temp(self) -> float:
        return float(np.max(self.pixels))

    @property
    def min_temp(self) -> float:
        return float(np.min(self.pixels))

    @property
    def mean_temp(self) -> float:
        return float(np.mean(self.pixels))

    @property
    def temp_range(self) -> float:
        return self.max_temp - self.min_temp

    @property
    def delta_above_ambient(self) -> float:
        """Max temperature rise above ambient."""
        return self.max_temp - self.ambient_temp


def detect_hotspots(
    frame: ThermalFrame,
    threshold_delta: float = 15.0,
) -> list[dict]:
    """Detect hotspots that exceed threshold above ambient.

    Args:
        threshold_delta: Temperature rise above ambient to flag (Celsius)

    Returns:
        List of hotspot detections with location and temperature.
    """
    threshold = frame.ambient_temp + threshold_delta
    hot_mask = frame.pixels > threshold

    if not np.any(hot_mask):
        return []

    # Find connected regions (simple approach)
    hotspots = []
    coords = np.argwhere(hot_mask)

    if len(coords) == 0:
        return []

    # Cluster nearby hot pixels (simple grid-based)
    visited = set()
    for y, x in coords:
        if (y, x) in visited:
            continue

        # Flood fill to find connected region
        region_pixels = []
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            if (cy, cx) in visited:
                continue
            if cy < 0 or cy >= frame.pixels.shape[0] or cx < 0 or cx >= frame.pixels.shape[1]:
                continue
            if frame.pixels[cy, cx] <= threshold:
                continue
            visited.add((cy, cx))
            region_pixels.append((cy, cx, frame.pixels[cy, cx]))
            stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])

        if region_pixels:
            temps = [p[2] for p in region_pixels]
            ys = [p[0] for p in region_pixels]
            xs = [p[1] for p in region_pixels]
            hotspots.append({
                "center_y": int(np.mean(ys)),
                "center_x": int(np.mean(xs)),
                "max_temp": float(max(temps)),
                "mean_temp": float(np.mean(temps)),
                "pixel_count": len(region_pixels),
                "delta_above_ambient": float(max(temps) - frame.ambient_temp),
            })

    return sorted(hotspots, key=lambda h: h["max_temp"], reverse=True)


def extract_thermal_features(frame: ThermalFrame) -> dict:
    """Extract features for thermal fault classification."""
    hotspots = detect_hotspots(frame)

    # Gradient analysis (sharp gradients indicate faults)
    grad_y, grad_x = np.gradient(frame.pixels)
    max_gradient = float(np.max(np.sqrt(grad_y**2 + grad_x**2)))

    # Quadrant analysis (localize heat distribution)
    h, w = frame.pixels.shape
    quadrants = {
        "top_left": frame.pixels[:h//2, :w//2],
        "top_right": frame.pixels[:h//2, w//2:],
        "bottom_left": frame.pixels[h//2:, :w//2],
        "bottom_right": frame.pixels[h//2:, w//2:],
    }

    return {
        "max_temp_c": frame.max_temp,
        "min_temp_c": frame.min_temp,
        "mean_temp_c": frame.mean_temp,
        "temp_range_c": frame.temp_range,
        "delta_above_ambient_c": frame.delta_above_ambient,
        "max_gradient_c_per_pixel": max_gradient,
        "hotspot_count": len(hotspots),
        "hotspot_max_temp_c": hotspots[0]["max_temp"] if hotspots else frame.mean_temp,
        "hotspot_total_pixels": sum(h["pixel_count"] for h in hotspots),
        "quadrant_temps": {k: float(np.mean(v)) for k, v in quadrants.items()},
    }


def classify_thermal_severity(frame: ThermalFrame) -> tuple[ThermalFaultType, str]:
    """Quick rule-based thermal classification for alerting."""
    delta = frame.delta_above_ambient
    hotspots = detect_hotspots(frame)

    if delta > 50:
        return ThermalFaultType.OVERHEATING, "critical"
    elif delta > 30:
        return ThermalFaultType.HOTSPOT, "severe"
    elif delta > 15:
        return ThermalFaultType.HOTSPOT, "moderate"
    elif len(hotspots) > 0:
        return ThermalFaultType.HOTSPOT, "incipient"
    else:
        return ThermalFaultType.NORMAL, "none"
