# Adapted from lzyang2000/morphmimic (Apache-2.0); see NOTICE and LICENSE.
import math

GOAL_K = 10
FPS = 50
KP_GOAL_VERSION = 10
HUMAN_JOINTS: tuple[str, ...] = (
  "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2",
  "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot", "Neck",
  "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
  "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)  # fmt: skip
# The SOMA joint that fills each slot at training time.
HUMAN_SOMA_SOURCE: tuple[str, ...] = (
  "Hips", "LeftLeg", "RightLeg", "Spine1", "LeftShin", "RightShin", "Spine2",
  "LeftFoot", "RightFoot", "Chest", "LeftToeBase", "RightToeBase", "Neck1",
  "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm", "LeftForeArm",
  "RightForeArm", "LeftHand", "RightHand", "LeftHandMiddle1", "RightHandMiddle1",
)  # fmt: skip
NUM_HUMAN = 24
HUMAN_PELVIS = 0
HUMAN_HEAD = 15
HUMAN_SPINE3 = 9
HUMAN_L_ANKLE = 7
HUMAN_R_ANKLE = 8
HUMAN_L_WRIST = 20
HUMAN_R_WRIST = 21
# Slot sets of the settings (`full` = all 24).
HUMAN_FIVE_IDX: tuple[int, ...] = (0, 7, 8, 20, 21)
HUMAN_SIX_IDX: tuple[int, ...] = (0, 9, 7, 8, 20, 21)
HUMAN_THREE_IDX: tuple[int, ...] = (15, 20, 21)
# Position anchors in order of preference: the Pelvis when its slot is on, else the
# Head.
HUMAN_ANCHORS: tuple[int, ...] = (0, 15)
# Features per slot per frame: [pos 3 | rot6d 6].
HUMAN_POINT_DIM = 9
HUMAN_FRAME_DIM = 216  # NUM_HUMAN * HUMAN_POINT_DIM
HUMAN_FEAT_DIM = 2160  # GOAL_K * HUMAN_FRAME_DIM
# Mask bits appended to the features: [slot mask 24 | frame mask GOAL_K].
HUMAN_MASK_DIM = 34
# The human Spine3's orientation relative to the robot anchor body (torso_link), rot6d
# per future frame, after the mask bits (keypoint goal version 10; versions 6-7 measured
# the human anchor against the robot pelvis; a row without the Spine3 slot measures
# its anchor, Pelvis else Head, against the robot torso), standardised by its own
# normaliser; all zero when the heading is masked.
HUMAN_HEADING_DIM = 60  # GOAL_K * 6
HUMAN_GOAL_DIM = 2254  # HUMAN_FEAT_DIM + HUMAN_MASK_DIM + HUMAN_HEADING_DIM
# Raw goal the env carries: per frame and slot the world pose [pos 3 | quat 4].
HUMAN_POSE_DIM = 7
HUMAN_RAW_DIM = 1680  # GOAL_K * NUM_HUMAN * HUMAN_POSE_DIM
# Raw poses then the robot anchor body's (torso_link) quaternion (wxyz, world).
HUMAN_ENV_DIM = 1684  # HUMAN_RAW_DIM + 4
# The SMPL rest frame against the robot pelvis frame on the retargeted references: an
# aligned human anchor is the robot pelvis turned by Rz(HUMAN_YAW_OFFSET) then
# Rx(HUMAN_ROLL_OFFSET) about its own x (SMPL joint frames are y-up). The yaw part is
# 90.0° on 221115/idle_loop_001__A061_M; with both, the relative orientation of the
# Pelvis on the fixture reset poses is within 1.1° of the identity.
HUMAN_YAW_OFFSET = math.pi / 2
HUMAN_ROLL_OFFSET = math.pi / 2
# Target-jitter half-widths per joint and frame: position in m, orientation in rad.
# HEFT's sizes (github.com/Axellwppr/motion_tracking, G1_tracking.yaml); SONIC's
# Table S4 values (5 cm / 1 cm / 0.1 / 0.2 rad) until 2026-10-27.
HUMAN_JITTER = {"pos_xy": 0.005, "pos_z": 0.005, "rot_rp": 0.01, "rot_yaw": 0.01}
# Slow drift of the whole human skeleton (HEFT's student motion randomisation, the
# headset's wandering pelvis and chest): per episode, sinusoids at a log-uniform
# frequency in `freq` Hz with a uniform phase. xy: along a random direction, peak
# speed U[0, xy_speed] m/s; z: a constant U[-z_offset, z_offset] m plus an amplitude
# U[0, z_amp] m; rotation about the pelvis: a random axis, amplitude U[0, rot_amp]
# rad. On top, per frame: a shift of radius U[0, pos_noise] m in a random direction
# and a turn of U[-rot_noise, rot_noise] rad about a random axis.
HUMAN_DRIFT = {
    "xy_speed": 0.2,
    "z_offset": 0.05,
    "z_amp": 0.025,
    "rot_amp": 0.1,
    "freq": (0.25, 1.0),
    "pos_noise": 0.005,
    "rot_noise": 0.02,
}
