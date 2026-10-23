# Casbot02 27 DOF asset

Copied from the user's local ScaleBFM/ScaleTrack Casbot02 assets. The copied
meshes and adapted MJCF are independent of that project's Python environment.
See `source_manifest.json` for the exact source and changes.

The source MJCF and URDF agree on common body transforms at zero joint angles.
Position limits follow the 27 DOF URDF used by ScaleTrack training; the source
MJCF had different elbow limits. Motors, PD gains, armature, and action scale
are configured in `protomotions/robot_configs/casbot02.py` and the Casbot02
tracking experiment. Embedded terrain/lights are removed. Passive joint
damping and frictionloss are zero, with active PD damping set by the config.
