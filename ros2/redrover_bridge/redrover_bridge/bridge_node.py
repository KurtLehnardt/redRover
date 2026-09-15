"""ROS 2 bridge for a redRover firmware rover.

Publishes whatever the board advertises as standard ROS messages and accepts
``/cmd_vel``, so a redRover node drops into an existing ROS 2 stack —
Nav2, RViz, rosbag — without either side knowing about the other.

Run it::

    ros2 run redrover_bridge bridge --ros-args \
        -p port:=/dev/ttyACM0 -p baud:=115200

Topic mapping is driven by each sensor's self-description, not by a hard-coded
table, so a sensor you add to the sketch appears on a topic without touching
this file.

======================  ==========================================
Sensor kind             Topic
======================  ==========================================
ACCELERATION            ``~/imu`` (sensor_msgs/Imu)
ANGULAR_RATE            ``~/imu`` (sensor_msgs/Imu)
DISTANCE                ``~/range/<name>`` (sensor_msgs/Range)
TEMPERATURE             ``~/temperature/<name>`` (sensor_msgs/Temperature)
BATTERY                 ``~/battery`` (sensor_msgs/BatteryState)
BUMPER                  ``~/bumper/<name>`` (std_msgs/Bool)
ENCODER                 ``~/joint_states`` (sensor_msgs/JointState)
anything else           ``~/sensor/<name>`` (std_msgs/Float32MultiArray)
======================  ==========================================
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState, Imu, JointState, Range, Temperature
    from std_msgs.msg import Bool, Float32MultiArray
    from std_srvs.srv import SetBool
    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - the module is importable without ROS
    ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

from src.rover import wire
from src.rover.backends.firmware import FirmwareRover


class RedRoverBridge(Node):
    """Adapts the redRover serial protocol to ROS 2 topics."""

    def __init__(self):
        super().__init__("redrover_bridge")

        self.declare_parameter("port", "")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("max_speed_mps", 1.0)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("stream_rate_hz", 50)

        self.frame_id = self.get_parameter("frame_id").value
        self.rover = FirmwareRover(
            port=self.get_parameter("port").value,
            baud=self.get_parameter("baud").value,
            max_speed_mps=self.get_parameter("max_speed_mps").value,
        )

        self._publishers: dict[int, object] = {}
        self._imu_publisher = None
        self._joint_publisher = None

        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.create_service(SetBool, "~/emergency_stop", self._on_estop)

        # The rover's asyncio machinery runs on its own loop in a background
        # thread; rclpy owns the main thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._submit(self._start())

    # -- lifecycle ----------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _start(self) -> None:
        try:
            await self.rover.connect()
        except Exception as exc:
            self.get_logger().error(f"could not reach the rover: {exc}")
            return

        caps = self.rover.capabilities
        self.get_logger().info(
            f"connected to '{caps.name}': {len(self.rover.sensors)} sensors, "
            f"max {caps.max_sensor_rate_hz:.0f} Hz"
        )
        self._advertise()

        self.rover.add_sensor_callback(self._on_sensor_data)
        rate_hz = int(self.get_parameter("stream_rate_hz").value)
        await self.rover.start_sensor_streaming(period_ms=max(1, 1000 // max(1, rate_hz)))

    def destroy_node(self):
        self._submit(self.rover.disconnect()).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        super().destroy_node()

    # -- publishers ---------------------------------------------------------

    def _advertise(self) -> None:
        """Create one publisher per sensor, typed from its descriptor."""
        for sensor in self.rover.sensors.values():
            if sensor.descriptor.failed:
                self.get_logger().warning(
                    f"sensor '{sensor.name}' failed to start on the board; not publishing"
                )
                continue

            kind = sensor.kind
            topic = _sanitise(sensor.name)

            if kind in (wire.SensorKind.ACCELERATION, wire.SensorKind.ANGULAR_RATE):
                if self._imu_publisher is None:
                    self._imu_publisher = self.create_publisher(Imu, "~/imu", 10)
                self._publishers[sensor.descriptor.id] = ("imu", kind)
            elif kind is wire.SensorKind.DISTANCE:
                self._publishers[sensor.descriptor.id] = (
                    "range", self.create_publisher(Range, f"~/range/{topic}", 10),
                )
            elif kind is wire.SensorKind.TEMPERATURE:
                self._publishers[sensor.descriptor.id] = (
                    "temperature",
                    self.create_publisher(Temperature, f"~/temperature/{topic}", 10),
                )
            elif kind is wire.SensorKind.BATTERY:
                self._publishers[sensor.descriptor.id] = (
                    "battery", self.create_publisher(BatteryState, "~/battery", 10),
                )
            elif kind is wire.SensorKind.BUMPER:
                self._publishers[sensor.descriptor.id] = (
                    "bumper", self.create_publisher(Bool, f"~/bumper/{topic}", 10),
                )
            elif kind is wire.SensorKind.ENCODER:
                if self._joint_publisher is None:
                    self._joint_publisher = self.create_publisher(
                        JointState, "~/joint_states", 10
                    )
                self._publishers[sensor.descriptor.id] = ("joint", sensor.name)
            else:
                self._publishers[sensor.descriptor.id] = (
                    "generic",
                    self.create_publisher(Float32MultiArray, f"~/sensor/{topic}", 10),
                )

            self.get_logger().info(
                f"publishing '{sensor.name}' ({kind.name}, {sensor.descriptor.rate_hz} Hz)"
            )

    async def _on_sensor_data(self, _snapshot: dict) -> None:
        """Called on every sample; publishes the sensors that changed."""
        stamp = self.get_clock().now().to_msg()
        joint_names: list[str] = []
        joint_positions: list[float] = []

        for sensor_id, entry in self._publishers.items():
            sensor = self.rover.sensors.get(sensor_id)
            if sensor is None or not sensor.last_values:
                continue
            kind, target = entry
            values = sensor.last_values

            if kind == "imu" and self._imu_publisher is not None:
                msg = Imu()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                if target is wire.SensorKind.ACCELERATION and len(values) >= 3:
                    # Values arrive in g; ROS wants m/s^2.
                    msg.linear_acceleration.x = values[0] * 9.80665
                    msg.linear_acceleration.y = values[1] * 9.80665
                    msg.linear_acceleration.z = values[2] * 9.80665
                elif len(values) >= 3:
                    msg.angular_velocity.x = math.radians(values[0])
                    msg.angular_velocity.y = math.radians(values[1])
                    msg.angular_velocity.z = math.radians(values[2])
                # No orientation estimate is produced here; -1 is the ROS
                # convention for "this field is not supplied", which is better
                # than publishing an identity quaternion that looks like data.
                msg.orientation_covariance[0] = -1.0
                self._imu_publisher.publish(msg)

            elif kind == "range":
                msg = Range()
                msg.header.stamp = stamp
                msg.header.frame_id = _sanitise(sensor.name)
                msg.radiation_type = Range.ULTRASOUND
                msg.field_of_view = 0.26
                msg.min_range = 0.02
                msg.max_range = 4.0
                msg.range = float(values[0])
                target.publish(msg)

            elif kind == "temperature":
                msg = Temperature()
                msg.header.stamp = stamp
                msg.header.frame_id = _sanitise(sensor.name)
                msg.temperature = float(values[0])
                msg.variance = 0.0
                target.publish(msg)

            elif kind == "battery":
                msg = BatteryState()
                msg.header.stamp = stamp
                msg.voltage = float(values[0])
                msg.present = True
                target.publish(msg)

            elif kind == "bumper":
                msg = Bool()
                msg.data = bool(values[0])
                target.publish(msg)

            elif kind == "joint":
                joint_names.append(target)
                joint_positions.append(float(values[0]))

            elif kind == "generic":
                msg = Float32MultiArray()
                msg.data = [float(v) for v in values]
                target.publish(msg)

        if joint_names and self._joint_publisher is not None:
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = joint_names
            msg.position = joint_positions
            self._joint_publisher.publish(msg)

    # -- subscriptions ------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Translate a ROS Twist into a redRover Drive frame."""
        linear_mm_s = int(msg.linear.x * 1000)
        angular_mrad_s = int(msg.angular.z * 1000)
        lateral_mm_s = int(msg.linear.y * 1000)

        caps = self.rover.capabilities
        if lateral_mm_s and not caps.holonomic:
            # Dropping the component silently would drive the robot somewhere
            # the planner did not ask for.
            self.get_logger().warning(
                "ignoring cmd_vel.linear.y: this chassis cannot strafe"
            )
            lateral_mm_s = 0

        self._submit(self.rover.drive(linear_mm_s, angular_mrad_s, lateral_mm_s))

    def _on_estop(self, request, response):
        if request.data:
            self._submit(self.rover.emergency_stop()).result(timeout=2)
            response.message = "emergency stop engaged"
        else:
            self.rover.clear_estop()
            response.message = "emergency stop cleared"
        response.success = True
        return response


def _sanitise(name: str) -> str:
    """Make a sensor name safe to use in a topic."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_")
    return cleaned.lower() or "unnamed"


def main(args=None):
    if not ROS_AVAILABLE:
        print(
            "rclpy is not available. Source a ROS 2 environment "
            "(e.g. `source /opt/ros/jazzy/setup.bash`) and try again.",
            file=sys.stderr,
        )
        return 1

    rclpy.init(args=args)
    node = RedRoverBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
