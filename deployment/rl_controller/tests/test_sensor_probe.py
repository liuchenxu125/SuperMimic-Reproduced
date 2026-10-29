"""Read-only probe test on an isolated ROS domain and remapped synthetic topics."""

import argparse
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import rclpy
from crb_ros_msg.msg import JointStateData
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Int32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = "187"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    # The synthetic sender keeps the default transport in both comparisons.
    for key in ("FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        os.environ.pop(key, None)
    rclpy.init()
    node = rclpy.create_node("passive_probe_synthetic_test")
    specs = [("/motion/joint_state", JointState),
             ("/motion/debug/joint_state", JointStateData),
             ("/motion/imu", Imu), ("/motion/current_mode", Int32)]
    topics = ["/supermimic_passive_test/stream_" + str(i) for i in range(4)]
    pubs = [node.create_publisher(kind, topic, 10) for topic, (_, kind) in zip(topics, specs)]
    try:
        for udp in (False, True):
            command = ["bash", str(args.script.resolve())] + (["--udp"] if udp else [])
            command += ["--ros-args", "-p", "seconds:=5.0"]
            for topic, (original, _) in zip(topics, specs):
                command += ["-r", original + ":=" + topic]
            with tempfile.TemporaryFile(mode="w+") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 12
                publisher_checked = False
                try:
                    while process.poll() is None and time.monotonic() < deadline:
                        stamp = node.get_clock().now().to_msg()
                        for publisher, (_, kind) in zip(pubs, specs):
                            message = kind()
                            if kind is Int32:
                                message.data = 3
                            else:
                                message.header.stamp = stamp
                                if kind is Imu:
                                    message.orientation.w = 0.
                                else:
                                    message.name = ["deliberately_incomplete_joint"]
                                    message.position = [0.]
                                    message.velocity = [0.]
                                # Intentionally leave IMU frame empty and quaternion zero:
                                # a passive probe counts delivery without payload filtering.
                            publisher.publish(message)
                        rclpy.spin_once(node, timeout_sec=0.02)
                        if any(name == "supermimic_sensor_probe" for name, _ in node.get_node_names_and_namespaces()):
                            outputs = node.get_publisher_names_and_types_by_node("supermimic_sensor_probe", "/")
                            assert all(name in ("/rosout", "/parameter_events") for name, _ in outputs), outputs
                            publisher_checked = True
                    assert process.poll() == 0, "Probe did not exit normally"
                    log.seek(0)
                    output = log.read()
                    assert publisher_checked, output
                    for stream in ("standard_joints", "debug_joints", "imu", "mode"):
                        assert re.search(stream + r": pubs=1 rx=[1-9][0-9]*", output), output
                    assert "frame='' quat_norm=0.000000" in output, output
                    assert "value=3" in output, output
                    assert "XMLPARSER Error" not in output, output
                    print(f"probe transport={'udp' if udp else 'default'}: receive checks passed; no motor publishers")
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        process.wait(timeout=5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
