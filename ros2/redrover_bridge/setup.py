from setuptools import find_packages, setup

package_name = "redrover_bridge"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="redRover",
    maintainer_email="noreply@example.com",
    description=(
        "ROS 2 bridge for redRover firmware rovers: publishes self-described "
        "sensors as standard messages and accepts cmd_vel."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            "bridge = redrover_bridge.bridge_node:main",
        ],
    },
)
