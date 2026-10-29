# SDK message compatibility package

`JoystickCmdReport.msg` is copied verbatim from the `hl_motion` 2.2.37
submodule `crb_ros_msg`, commit `3228d83f5629727eb2a8440bf9fd46c9f9fcedbf`,
path `msg/JoystickCmdReport.msg`. Its wire layout has no Header; default source
frequency is 20 Hz. No guessed sensor_msgs/Joy button mapping is used.

The SDK consumes `/joystick_events`. Button IDs: A=0, B=1, X=2, Y=3,
LB=4, RB=5, BACK=6, START=7, LT=11, RT=12. The SDK owns modifier chords
and mode changes. SuperMimic uses unmodified X (advance and play next clip) and B (return) release events. D-pad input does not select motions.

The build uses this package when an external SDK installation is missing any
required JointStateData, GetRobotMode or JoystickCmdReport C++ interface.
Existing JointStateData and GetRobotMode definitions are preserved.
