# Casbot02 standalone SuperMimic sim2sim bundle

From this directory, using Python with `numpy`, `torch` and `mujoco` installed:

```bash
MUJOCO_GL=glfw python sim2sim.py --clip walk4_subject1 --render
python sim2sim.py --clip walk3_subject5 --seconds 10 --report check.json
python sim2sim.py --list-clips
```

Runtime does not import SuperMimic, mjlab or ProtoMotions. `actor.pt` contains
only the actor; `manifest.json` records the motor/physics/input contract and
source checkpoint. `scene/` contains the complete portable XML and mesh assets.

Space pauses; R resets; N/B or 1–9 selects another clip and resets the robot.
This is not a continuous motion-transition controller. Runs finish on clip end,
height/tilt failure, the requested duration, or closing the viewer.

The renderer shows the policy robot and a translucent blue reference. Root pose
is read from MuJoCo ground truth, not a real-robot state estimator. This bundle
is a simulation test, not a hardware control program.

See `docs/CASBOT02_SIM2SIM.md` in the source project for results and re-export
commands. MuJoCo/PyTorch versions and source-file hashes are in the manifest.
