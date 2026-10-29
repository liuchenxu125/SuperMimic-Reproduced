#!/usr/bin/env bash
# Native host build only. For ARM cross compilation use build_in_docker.sh arm64.
set -eo pipefail
SM_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
SM_NATIVE_CMAKE_ARGS=()
# A user's active Conda base environment may otherwise override the ROS host
# yaml-cpp with a library unavailable on the deployment machine.
SM_SYSTEM_YAML="/usr/lib/$(c++ -dumpmachine)/cmake/yaml-cpp"
if [[ -f "$SM_SYSTEM_YAML/yaml-cpp-config.cmake" ]]; then
    SM_NATIVE_CMAKE_ARGS+=("-Dyaml-cpp_DIR=$SM_SYSTEM_YAML")
fi
SM_MSG_PREFIX="${HL_MOTION_PREFIX:-/workspace/hl_motion}"
if [[ ! -f "$SM_MSG_PREFIX/include/crb_ros_msg/crb_ros_msg/msg/joint_state_data.hpp" || ! -f "$SM_MSG_PREFIX/include/crb_ros_msg/crb_ros_msg/srv/get_robot_mode.hpp" || ! -f "$SM_MSG_PREFIX/include/crb_ros_msg/crb_ros_msg/msg/joystick_cmd_report.hpp" ]]; then
    SM_MSG_PREFIX="$SM_SOURCE_ROOT/build/supermimic_msgs"
    cmake -S "$SM_SOURCE_ROOT/cmake/crb_ros_msg_compat" \
        -B "$SM_SOURCE_ROOT/build/supermimic_msgs_build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SM_MSG_PREFIX" \
        -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
    cmake --build "$SM_SOURCE_ROOT/build/supermimic_msgs_build" -j2
    cmake --install "$SM_SOURCE_ROOT/build/supermimic_msgs_build"
fi
cmake -S "$SM_SOURCE_ROOT/supermimic" -B "$SM_SOURCE_ROOT/build/supermimic_ros" \
    -DSUPERMIMIC_BUILD_ROS=ON -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -Dcrb_ros_msg_DIR="$SM_MSG_PREFIX/share/crb_ros_msg/cmake" \
    "${SM_NATIVE_CMAKE_ARGS[@]}"
cmake --build "$SM_SOURCE_ROOT/build/supermimic_ros" -j2
