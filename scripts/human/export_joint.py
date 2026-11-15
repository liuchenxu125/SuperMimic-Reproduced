#!/usr/bin/env python3
"""Export both actors from one joint checkpoint; never installs to rl_controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from export_human import HumanActor
from supermimic.human.constants import HUMAN_GOAL_DIM, KP_GOAL_VERSION
from supermimic.human.features import fixed_human_masks
from supermimic.human.joint_training import FORMAT
from supermimic.human.model import HumanEncoder
from supermimic.human.teacher import digest, policy_config
from supermimic.policy.nets import Policy


def export(checkpoint, output):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('joint_format')!=FORMAT or ck['joint_config']['human_goal_version']!=KP_GOAL_VERSION:
        raise ValueError('Expected current motion+human joint checkpoint')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    cfg=policy_config(ck['config']['policy'])
    policy=Policy(cfg).eval();policy.load_state_dict(ck['policy'])
    human=HumanEncoder(cfg).eval();human.load_state_dict(ck['human'])
    actor=HumanActor(SimpleNamespace(policy=policy),human).eval()
    def inputs(n,points):
        masks=fixed_human_masks(n,points,10,torch.device('cpu'))
        quat=torch.zeros(n,10,24,4);quat[...,0]=1
        torso=torch.zeros(n,4);torso[...,0]=1
        return (torch.randn(n,930)*.1,torch.randn(n,10,24,3)*.2,quat,torso,*masks)
    with torch.no_grad():
        traced_h=torch.jit.trace(actor,inputs(1,'full'),check_trace=False)
        traced_m=torch.jit.trace(policy,(torch.zeros(1,930),torch.zeros(1,690)),check_trace=False)
        errors=[]
        for n,points in ((1,'full'),(4,'five'),(2,'full')):
            x=inputs(n,points)
            a,b=actor(*x),traced_h(*x)
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            g=torch.randn(n,690)*.1
            c,d=policy(x[0],g),traced_m(x[0],g)
            torch.testing.assert_close(c,d,rtol=2e-5,atol=2e-6)
            errors.append({'batch':n,'points':points,'human_max_abs':float((a-b).abs().max()),
                           'motion_max_abs':float((c-d).abs().max())})
    output.mkdir(parents=True,exist_ok=True)
    traced_h.save(str(output/'human_actor.pt'));traced_m.save(str(output/'motion_actor.pt'))
    manifest={'format':'supermimic-motion-human-actors-v1','checkpoint':str(checkpoint.resolve()),
        'checkpoint_sha256':digest(checkpoint),'iteration':ck['iteration'],'human_goal_version':KP_GOAL_VERSION,
        'obs_dim':930,'motion_goal_dim':690,'human_goal_dim':HUMAN_GOAL_DIM,'actions':27,
        'control_dt':.02,'shared_decoder':True,'robot_contract':ck['robot_contract'],
        'human_inputs':['obs_hist[B,930]','human_pos[B,10,24,3]','human_quat_wxyz[B,10,24,4]',
                        'robot_torso_quat_wxyz[B,4]','point_mask[B,24] bool','frame_mask[B,10] bool','heading_mask[B] bool'],
        'human_coordinates':'SOMA uniform / SMPL rest frames, metres, z-up, aligned heading',
        'native_history_reference_pelvis_rot6d_required':True,'validation':errors,
        'normal_deployment_changed':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    print(json.dumps(export(a.checkpoint,a.output_dir),indent=2))
