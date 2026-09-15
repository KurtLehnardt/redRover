# ROS 2 bridge

Runs a redRover firmware rover as a ROS 2 node, so it works with Nav2, RViz,
`teleop_twist_keyboard`, and `ros2 bag` without either side being aware of the
other.

```bash
# The bridge imports src.rover from this repository, so put it on PYTHONPATH.
export PYTHONPATH=$PWD:$PYTHONPATH

cd ros2 && colcon build --packages-select redrover_bridge
source install/setup.bash

ros2 run redrover_bridge bridge --ros-args -p port:=/dev/ttyACM0 -p baud:=115200
```

## Topics

Sensor topics are created from each sensor's self-description, so a sensor you
add to the sketch shows up on a topic without editing the bridge.

| Direction | Topic | Type |
|---|---|---|
| in | `/cmd_vel` | `geometry_msgs/Twist` |
| out | `~/imu` | `sensor_msgs/Imu` |
| out | `~/range/<name>` | `sensor_msgs/Range` |
| out | `~/temperature/<name>` | `sensor_msgs/Temperature` |
| out | `~/battery` | `sensor_msgs/BatteryState` |
| out | `~/bumper/<name>` | `std_msgs/Bool` |
| out | `~/joint_states` | `sensor_msgs/JointState` |
| out | `~/sensor/<name>` | `std_msgs/Float32MultiArray` |
| srv | `~/emergency_stop` | `std_srvs/SetBool` |

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `port` | `""` | Serial device; empty auto-detects |
| `baud` | `115200` | Must match the sketch's `Serial.begin()` |
| `max_speed_mps` | `1.0` | Calibrated full-throttle ground speed |
| `frame_id` | `base_link` | Frame stamped on IMU messages |
| `stream_rate_hz` | `50` | Requested sensor streaming rate |

## Notes

`cmd_vel.linear.y` is refused with a warning on a non-holonomic chassis rather
than being dropped silently — a planner that thinks the robot strafed when it
did not will accumulate pose error it cannot explain.

The firmware's command watchdog still applies: if the bridge stops publishing,
the board stops the motors on its own.
