"""Drone-Rover orchestrator — coordinates ground robot and aerial drone.

The orchestrator manages the full lifecycle:
1. RVR+ drives to station, performs ground-level sensing
2. If anomaly detected that needs aerial confirmation → deploy drone
3. RVR+ stays still (landing pad) while drone inspects
4. Drone returns and docks
5. RVR+ continues patrol

Safety rules:
- RVR+ must be stationary during drone flight
- Drone must have sufficient battery for mission + return
- If drone loses contact, RVR+ stays in place as landing beacon
- Emergency land if battery drops below threshold
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime

from .controller import DroneController, DroneState, DroneType, AerialCapture
from .mission import DroneMission, should_deploy_drone, FAULT_TO_MISSION
from ..rover.controller import RoverController, RoverState

logger = logging.getLogger(__name__)


@dataclass
class DeploymentResult:
    """Result of a drone deployment."""
    mission_id: str
    station_id: str
    success: bool
    captures: list[AerialCapture]
    flight_duration: float
    battery_used: int
    error: str | None = None


class DroneRoverOrchestrator:
    """Coordinates the ground robot and piggyback drone."""

    def __init__(
        self,
        rover: RoverController,
        drone: DroneController,
        min_deploy_battery: int = 40,
        max_mission_duration: float = 90.0,
    ):
        self.rover = rover
        self.drone = drone
        self.min_deploy_battery = min_deploy_battery
        self.max_mission_duration = max_mission_duration
        self._deployments: list[DeploymentResult] = []

    @property
    def can_deploy(self) -> bool:
        """Check if drone deployment is possible."""
        return (
            self.drone.is_flight_ready
            and self.drone.battery >= self.min_deploy_battery
            and self.rover.state == RoverState.DWELLING
        )

    async def evaluate_and_deploy(
        self,
        station_id: str,
        fault_type: str,
        station_config: dict | None = None,
    ) -> DeploymentResult | None:
        """Evaluate if drone should deploy, and execute if so.

        Args:
            station_id: Current station ID
            fault_type: Detected fault type from ground sensors
            station_config: Station-specific configuration

        Returns:
            DeploymentResult if deployed, None if deployment not warranted.
        """
        # Check if fault warrants drone deployment
        if not should_deploy_drone(fault_type, station_config):
            logger.info("[ORCH] Fault '%s' does not require aerial inspection", fault_type)
            return None

        # Check if drone is available
        if not self.can_deploy:
            logger.warning("[ORCH] Drone not available for deployment "
                          "(state=%s, battery=%d%%, rover=%s)",
                          self.drone.state, self.drone.battery, self.rover.state)
            return None

        # Generate appropriate mission
        mission = self._generate_mission(station_id, fault_type)
        if mission is None:
            return None

        # Check if battery is sufficient for this mission
        estimated_duration = mission.estimated_duration
        if estimated_duration > self.max_mission_duration:
            logger.warning("[ORCH] Mission too long (%.0fs > %.0fs limit)",
                          estimated_duration, self.max_mission_duration)
            return None

        # Execute deployment
        return await self.deploy(mission)

    async def deploy(self, mission: DroneMission) -> DeploymentResult:
        """Execute a drone mission.

        The RVR+ remains stationary as the landing pad.
        """
        logger.info("")
        logger.info("  ╔══════════════════════════════════════════╗")
        logger.info("  ║  DRONE DEPLOYMENT: %s", mission.mission_type.value.ljust(20) + "║")
        logger.info("  ║  Station: %s", (mission.station_id + " " * 20)[:31] + "║")
        logger.info("  ║  Targets: %d | Est: %.0fs", mission.total_targets, mission.estimated_duration)
        logger.info("  ║  Reason: %s", mission.reason[:32] + "║" if len(mission.reason) <= 32 else mission.reason[:29] + "...║")
        logger.info("  ╚══════════════════════════════════════════╝")

        start_time = time.time()
        start_battery = self.drone.battery
        captures = []
        error = None

        try:
            # Phase 1: Launch
            logger.info("  [ORCH] Phase 1: Launch from cradle")
            success = await self.drone.launch()
            if not success:
                return DeploymentResult(
                    mission_id=mission.mission_id,
                    station_id=mission.station_id,
                    success=False,
                    captures=[],
                    flight_duration=0,
                    battery_used=0,
                    error="Launch failed",
                )

            # Phase 2: Execute inspection targets
            logger.info("  [ORCH] Phase 2: Executing %d inspection targets", mission.total_targets)
            for i, target in enumerate(mission.targets):
                # Safety check: battery
                if self.drone.battery < self.min_deploy_battery // 2:
                    logger.warning("  [ORCH] Low battery (%d%%), aborting mission",
                                   self.drone.battery)
                    error = "Low battery abort"
                    break

                # Safety check: timeout
                elapsed = time.time() - start_time
                if elapsed > mission.timeout:
                    logger.warning("  [ORCH] Mission timeout (%.0fs), aborting", elapsed)
                    error = "Timeout abort"
                    break

                # Fly to target
                await self.drone.fly_to_target(target)

                # Inspect (capture data)
                capture = await self.drone.inspect(target)
                captures.append(capture)

                logger.info("  [ORCH]   Target %d/%d complete: %s (%d images)",
                            i + 1, mission.total_targets, target.name, len(capture.images))

            # Phase 3: Return and land
            logger.info("  [ORCH] Phase 3: Return to cradle")
            await self.drone.return_to_cradle()

        except Exception as e:
            logger.error("  [ORCH] Mission error: %s", e)
            error = str(e)
            # Emergency land
            try:
                await self.drone.emergency_land()
            except Exception:
                pass

        flight_duration = time.time() - start_time
        battery_used = start_battery - self.drone.battery

        result = DeploymentResult(
            mission_id=mission.mission_id,
            station_id=mission.station_id,
            success=error is None,
            captures=captures,
            flight_duration=flight_duration,
            battery_used=battery_used,
            error=error,
        )
        self._deployments.append(result)

        logger.info("  [ORCH] Deployment complete: %s | Duration: %.1fs | Battery used: %d%%",
                    "SUCCESS" if result.success else f"FAILED ({error})",
                    flight_duration, battery_used)

        return result

    def _generate_mission(self, station_id: str, fault_type: str) -> DroneMission | None:
        """Generate a mission based on fault type."""
        if fault_type in FAULT_TO_MISSION:
            _, generator = FAULT_TO_MISSION[fault_type]
            reason = f"Ground sensors detected: {fault_type}"
            return generator(station_id=station_id, reason=reason)

        logger.warning("[ORCH] No mission template for fault: %s", fault_type)
        return None

    @property
    def total_deployments(self) -> int:
        return len(self._deployments)

    @property
    def successful_deployments(self) -> int:
        return sum(1 for d in self._deployments if d.success)
