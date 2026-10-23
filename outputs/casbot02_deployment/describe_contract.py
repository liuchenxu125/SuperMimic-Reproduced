"""Describe the current trained Casbot02 actor's hardware integration contract.

Read-only with respect to the checkpoint, robot, and training source. This does
not export a controller, contact hardware, or establish deployment readiness.
Run from the repository root with .venv/bin/python.
"""
from pathlib import Path
import json
import torch
from mjlab.entity import Entity
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.robots import get_robot

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/'logs/casbot02_lafan1/ckpt_latest.pt'
torch.set_num_threads(2)
policy,cfg=load_policy_from_checkpoint(CHECKPOINT,'cpu')
assert cfg['robot']=='casbot02' and cfg['env']['anchor_obs']
assert cfg['policy']['goal_dim']==649 and cfg['policy']['history']==10
profile=get_robot('casbot02')
entity=Entity(profile.entity_cfg())
model=entity.spec.compile()
joint_ids,action_names=entity.find_joints_by_actuator_names(('.*',))
joint_names=list(entity.joint_names)
assert joint_names==action_names and len(joint_names)==27
motors=profile.motor_parameters()['motors']
action_scales=profile.action_scale()
rows=[]
for i,name in enumerate(action_names):
    joint=model.joint(name)
    qadr=int(model.jnt_qposadr[joint.id])
    p=motors[name]
    rows.append(dict(index=i,joint_name=name,default_joint_pos_rad=float(model.key_qpos[0,qadr]),
        action_scale_rad=action_scales[name],kp=p['stiffness'],kd=p['damping'],
        effort_limit_nm=p['effort_limit'],source_velocity_limit_rad_s=p['velocity_limit'],
        joint_hard_limits_rad=model.jnt_range[joint.id].tolist()))
with torch.inference_mode():
    y=policy(torch.zeros(1,870),torch.zeros(1,649))
assert y.shape==(1,27) and torch.isfinite(y).all()
contract={
 'checkpoint':str(CHECKPOINT.relative_to(ROOT)),'robot':'casbot02',
 'control_hz':50,'control_dt_s':0.02,'simulator_physics_hz':200,
 'inputs':{
  'obs_hist':{'shape':[1,870],'dtype':'float32','packing':'term-major; each term oldest to newest',
    'slices':[
     {'slice':[0,270],'term':'joint_pos - default_joint_pos','history':10,'per_frame':27,'units':'rad'},
     {'slice':[270,540],'term':'joint_vel','history':10,'per_frame':27,'units':'rad/s'},
     {'slice':[540,570],'term':'base_link angular velocity in base_link frame','history':10,'per_frame':3,'units':'rad/s'},
     {'slice':[570,600],'term':'R_world_from_base.T @ [0,0,-1]','history':10,'per_frame':3,'units':'unit vector'},
     {'slice':[600,870],'term':'previous normalized clean policy action','history':10,'per_frame':27,'units':'dimensionless'},
    ],'reset':'Repeat first measured frame for all 10 history slots; previous actions zero.'},
  'goal':{'shape':[1,649],'dtype':'float32','packing':'10 frame-major reference frames, then 3 relative-position and 6 relative-rotation elements',
    'reference_offsets_s':[round(i*.02,2) for i in range(10)],
    'per_reference_frame_64':[
     {'slice':[0,27],'term':'reference joint_pos - default_joint_pos','units':'rad'},
     {'slice':[27,54],'term':'reference joint_vel','units':'rad/s'},
     {'slice':[54,57],'term':'reference gravity in reference base frame'},
     {'slice':[57,60],'term':'reference angular velocity in reference base frame','units':'rad/s'},
     {'slice':[60,63],'term':'reference linear velocity in reference yaw-only heading frame','units':'m/s'},
     {'slice':[63,64],'term':'reference base_link height','units':'m'},
    ],'anchor_slices':[
     {'slice':[640,643],'term':'R_robot.T @ (p_reference - p_robot)','units':'m'},
     {'slice':[643,649],'term':'(R_robot.T @ R_reference)[:, :2].reshape(6)','packing':'R00,R01,R10,R11,R20,R21'},
    ],'anchor_body':'base_link','requires_real_base_pose_estimate':True},
 },
 'normalization':'Policy contains fixed input scaling buffers. Supply raw values in stated units; do not scale again outside the loaded Policy.',
 'output':{'shape':[1,27],'term':'normalized clean actions','bounded_by_tanh':False,
   'joint_target':'q_des[i] = default_joint_pos[i] + action_scale[i] * action[i]',
   'pd_reference':'Kp*(q_des-q) - Kd*qdot, desired velocity zero; simulator uses implicit PD and effort limits.',
   'deployment_noise':'No synthetic action or observation noise.'},
 'quaternion_convention_in_project':'wxyz, rotation from base frame to world frame',
 'motor_rows_in_policy_order':rows,
 'joint_obs_order':joint_names,
 'warning':'Policy action targets use entity natural joint order; raw MJCF actuator enumeration has waist last and must not be substituted for the policy order.',
 'excluded_from_actor_inference':['world model','critic','replay buffer','domain-randomization context xi'],
 'validation':{'cpu_shape_check_passed':True,'zero_inputs_used_only_for_shape_check':True,
               'hardware_deployment_tested':False,
               'exported_runtime_model':(ROOT/'outputs/casbot02_sim2sim/actor.pt').is_file()},
}
(OUT/'io_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
print(json.dumps({'obs_hist':[1,870],'goal':[1,649],'action':[1,27],
                  'policy_action_order':action_names,'output':str(OUT/'io_contract.json')},indent=2))
