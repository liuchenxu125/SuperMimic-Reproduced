"""Joint-only data alignment, replay starts and optimizer isolation regressions."""
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.config import TrainConfig
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.human.constants import HUMAN_JOINTS
from supermimic.human.data import Pair, load_pair
from supermimic.human.joint_data import HumanWindows, stage_robot_pairs
from supermimic.human.joint_student import JointCopyStep, TorsoFrame, frozen
from supermimic.human.joint_training import restore_config, robot_contract, check_warmstart_contract
from supermimic.human.model import HumanEncoder
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


@pytest.fixture
def paired(tmp_path):
    meta, model = robot_contract()
    pairs = []
    for name, n in (("z_clip", 64), ("a_clip", 70)):
        robot, human = tmp_path/f"{name}.npz", tmp_path/f"{name}_human.npz"
        np.savez(robot, fps=50., joint_names=meta["joint_names"], body_names=["base_link"],
                 joint_pos=np.zeros((n,27)),joint_vel=np.zeros((n,27)),
                 body_pos_w=np.tile([[[0,0,.9]]],(n,1,1)),
                 body_quat_w=np.tile([[[1,0,0,0]]],(n,1,1)),
                 body_lin_vel_w=np.zeros((n,1,3)),body_ang_vel_w=np.zeros((n,1,3)))
        pos=np.zeros((n,24,3));pos[...,2]=1.;pos[:,:,0]=np.arange(n)[:,None]*.01
        np.savez(human,fps=50.,human_joint_names=HUMAN_JOINTS,human_pos_w=pos,
                 human_quat_w=np.tile([1,0,0,0],(n,24,1)))
        pairs.append(Pair(name,robot,human))
    # Deliberately reversed pair order vs native sorted clip filenames.
    stage=tmp_path/"staged"
    stage_robot_pairs(pairs,stage,meta)
    lib=MotionLibrary(sorted(stage.glob("*.npz")),("base_link",),np.zeros(27),device="cpu",
        compact=True,root_body_name="base_link",robot_name="casbot02",joint_names=meta["joint_names"])
    return pairs,meta,model,lib


def test_joint_windows_use_name_mapping_and_clip_end(paired):
    pairs,meta,_,lib=paired
    windows=HumanWindows(pairs,meta,lib,cache_clips=1)
    frames=torch.tensor([0,68,70,133])
    result=windows.future(frames,10)
    by_name={p.name:p for p in pairs}
    for i,frame in enumerate(frames.tolist()):
        clip=lib.clips[int(lib.frame_clip[frame])]
        _,poses=load_pair(by_name[clip.name],meta)
        expected=poses[np.minimum(np.arange(10)+frame-clip.start,len(poses)-1)]
        np.testing.assert_allclose(result[i],expected)
    assert len(windows.data.cache)==1


def test_preloaded_human_cpu_storage_matches_bounded_cache(paired):
    pairs,meta,_,lib=paired
    packed=HumanWindows(pairs,meta,lib,cache_clips=0)
    cached=HumanWindows(pairs,meta,lib,cache_clips=1)
    frames=torch.tensor([0,68,70,133,69])
    torch.testing.assert_close(packed.future(frames,10),cached.future(frames,10))
    assert packed._packed.device.type=='cpu' and packed.data is None


def test_staging_honors_verified_crop_and_preserves_source(paired,tmp_path):
    pairs,meta,_,_=paired
    pair=pairs[0]
    before=pair.robot_npz.read_bytes()
    crop=Pair(pair.name,pair.robot_npz,pair.human_npz,3,3,52)
    stage_robot_pairs([crop],tmp_path/"crop",meta)
    with np.load(tmp_path/"crop"/f"{pair.name}.npz") as data:
        assert len(data["joint_pos"])==52
    assert pair.robot_npz.read_bytes()==before
    stage_robot_pairs([crop],tmp_path/"crop",meta)  # validated resume, not overwrite


def test_replay_copy_updates_human_only_and_restores_flags(paired):
    torch.manual_seed(1)
    pairs,meta,model,lib=paired
    cfg=PolicyCfg(num_joints=27,goal_dim=690,obs_term_scale=(1.,5.,2.,1.,1.,1.),
                  encoder_hidden=(16,12),decoder_hidden=(16,12))
    policy=Policy(cfg);human=HumanEncoder(cfg)
    wm=WorldModel(WorldModelCfg(hidden=(16,16),state_spec=StateSpecCfg(num_joints=27)),None,"cpu")
    replay=ReplayBuffer(2,64,num_joints=27,device="cpu")
    for t in range(24):
        state=torch.zeros(2,67);state[:,2]=.9;state[:,3]=1.
        replay.add_step(state,torch.zeros(2,27),torch.zeros(2,27),wm.spec.xi_offset.expand(2,-1),
            torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long),
            torch.full((2,),t),torch.full((2,),t==0))
    step=JointCopyStep(wm,policy,human,lib,HumanWindows(pairs,meta,lib),TorsoFrame(model,"cpu"),
                       batch_size=4,horizon=3,obs_noise=False)
    old=[p.detach().clone() for m in (wm,policy) for p in m.parameters()]
    human_old=[p.detach().clone() for p in human.parameters()]
    metrics=step.update(replay)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(a,b) for a,b in zip(old,[p for m in (wm,policy) for p in m.parameters()]))
    assert any(not torch.equal(a,b) for a,b in zip(human_old,human.parameters()))
    assert all(p.requires_grad for m in (wm,policy) for p in m.parameters())
    assert all(p.grad is None for m in (wm,policy) for p in m.parameters())


def test_freeze_restores_original_flags_on_failure():
    module=torch.nn.Linear(2,2)
    module.bias.requires_grad_(False)
    with pytest.raises(RuntimeError):
        with frozen(module):
            assert not any(p.requires_grad for p in module.parameters())
            raise RuntimeError("test")
    assert module.weight.requires_grad and not module.bias.requires_grad


def test_native_nested_config_is_independent():
    original=TrainConfig()
    values=asdict(original)
    restored=restore_config(values)
    restored.policy.num_joints=27
    assert original.policy.num_joints==29
    assert restored.world_model.state_spec.dt==original.world_model.state_spec.dt
    assert isinstance(restored.data.npz_dir,Path)
    assert asdict(original)==values


def test_joint_resume_allows_only_explicit_human_batch_change():
    from supermimic.human.joint_training import check_joint_resume_config
    saved={'human_batch':512,'horizon':32,'masking':'mixed'}
    assert check_joint_resume_config(saved,dict(saved))=={}
    assert check_joint_resume_config(saved,saved|{'human_batch':4096})=={'human_batch':[512,4096]}
    with pytest.raises(ValueError,match='horizon'):
        check_joint_resume_config(saved,saved|{'human_batch':4096,'horizon':16})
    with pytest.raises(ValueError,match='masking'):
        check_joint_resume_config(saved,saved|{'masking':'five'})


def test_warmstart_rejects_motor_change():
    meta,_=robot_contract()
    import copy
    exported=copy.deepcopy(meta)
    check_warmstart_contract(meta,exported)
    exported["motors"][0]["kp"]*=2
    with pytest.raises(ValueError,match="Warm-start motor contract differs"):
        check_warmstart_contract(meta,exported)


def test_full_training_refuses_partial_preparation(tmp_path):
    import json
    from supermimic.human.joint_training import require_prepared_pairs
    path=tmp_path/'pairs.csv'
    report=tmp_path/'preparation_report.json'
    report.write_text(json.dumps({'requested_limit':0,'pending':10}))
    with pytest.raises(ValueError,match='still incomplete'):
        require_prepared_pairs(path)
    report.write_text(json.dumps({'requested_limit':0,'pending':0}))
    require_prepared_pairs(path)
    report.write_text(json.dumps({'requested_limit':1000,'pending':100000}))
    require_prepared_pairs(path)


def test_torso_frame_matches_mujoco(paired):
    _,_,model,_=paired
    data=mujoco.MjData(model);data.qpos[:]=model.qpos0;data.qpos[2]=.9
    j=model.joint("waist_yaw_joint");data.qpos[j.qposadr[0]]=.4
    mujoco.mj_forward(model,data)
    state=torch.tensor(np.r_[data.qpos,np.zeros(33)],dtype=torch.float32)[None]
    actual=TorsoFrame(model,"cpu")(state)[0].numpy()
    expected=data.xquat[model.body("waist_yaw_link").id]
    assert abs(np.dot(actual,expected))==pytest.approx(1.,abs=1e-6)


def test_streaming_preparation_resumes_verified_pairs(paired,tmp_path,monkeypatch):
    import csv,importlib.util,io,json,shutil,sys,tarfile
    from supermimic.human import soma
    pairs,_,_,_=paired
    source=tmp_path/'source';source.mkdir()
    name='example__A001'
    shutil.copy2(pairs[0].robot_npz,source/f'{name}.npz')
    manifest=source/'manifest.csv';manifest.write_text(f'name,keep\n{name},1\n')
    archive=tmp_path/'soma.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        payload=b'synthetic test BVH placeholder'
        member=tarfile.TarInfo(f'soma_uniform/bvh/date/{name}.bvh');member.size=len(payload)
        tar.addfile(member,io.BytesIO(payload))
        link=tarfile.TarInfo(f'soma_uniform/bvh/date/ignored__A002.bvh');link.type=tarfile.SYMTYPE
        link.linkname='/tmp/ignored';tar.addfile(link)
    calls=[]
    def convert(inp,out):
        calls.append(inp)
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(pairs[0].human_npz,out)
    monkeypatch.setattr(soma,'convert_human_clip',convert)
    script=Path(__file__).resolve().parents[1]/'scripts/human/prepare_seed_joint.py'
    spec=importlib.util.spec_from_file_location('stream_seed_joint',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'output'
    argv=[str(script),'--archive',str(archive),'--robot-dir',str(source),
          '--manifest',str(manifest),'--output-dir',str(out),'--limit','1']
    monkeypatch.setattr(sys,'argv',argv)
    module.main();module.main()
    assert len(calls)==1
    assert not (out/'.current.bvh').exists()
    with (out/'pairs.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['name']==name
    assert json.loads((out/'preparation_report.json').read_text())['ready']==1
    # EOF must classify genuinely absent source BVHs, rather than leaving the
    # full-dataset trainer waiting for an impossible pending=0 forever.
    shutil.copy2(pairs[0].robot_npz,source/'missing__A003.npz')
    manifest.write_text(f'name,keep\n{name},1\nmissing__A003,1\n')
    all_out=tmp_path/'all_output'
    monkeypatch.setattr(sys,'argv',[str(script),'--archive',str(archive),'--robot-dir',str(source),
        '--manifest',str(manifest),'--output-dir',str(all_out),'--limit','0'])
    module.main()
    report=json.loads((all_out/'preparation_report.json').read_text())
    assert report['ready']==1 and report['rejected']==1 and report['pending']==0
    assert report['audit'][-1]['reason']=='missing_human_in_archive'


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_joint_cache_handoff_tracks_updates_not_logging_or_physics(monkeypatch, device):
    from types import SimpleNamespace

    from supermimic.human.joint_training import JointTrainer
    from supermimic.training.loop import Trainer

    trainer = object.__new__(JointTrainer)
    trainer.device = device
    trainer._joint_ready = True
    trainer._reported_update = trainer._cache_released_update = 12
    trainer.policy_trainer = SimpleNamespace(human_updates=12)
    trainer.cfg = SimpleNamespace(schedule=SimpleNamespace(log_interval=50))
    events = []
    monkeypatch.setattr(Trainer, "_synchronize", lambda self: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))

    # Restored counters during warmup/WM do not constitute a new joint update.
    trainer._synchronize()
    assert events == ["sync"]
    for count in (13, 14):
        events.clear()
        trainer.policy_trainer.human_updates = count
        trainer._synchronize()
        assert events == (["sync", "release"] if device.startswith("cuda") else ["sync"])
        assert trainer._cache_released_update == count
        assert trainer._reported_update == 12  # No log record at these counts.
        events.clear()
        trainer._synchronize()
        assert events == ["sync"]


def test_joint_cache_handoff_follows_optional_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from supermimic.human.joint_training import JointTrainer
    from supermimic.training.loop import Trainer

    trainer = object.__new__(JointTrainer)
    trainer.device, trainer._joint_ready = "cuda:0", True
    trainer._reported_update = trainer._cache_released_update = 49
    trainer.policy_trainer = SimpleNamespace(human_updates=50, latest={})
    trainer.cfg = SimpleNamespace(schedule=SimpleNamespace(log_interval=50))
    trainer.iteration, trainer.validate_every = 49, 50
    trainer.collection_counts, trainer.log_dir = {}, tmp_path
    trainer._initial_versions = {name: [] for name in ("wm", "motion", "human")}
    trainer.wm = trainer.policy = trainer.human = torch.nn.Identity()
    trainer.writer = SimpleNamespace(add_scalar=lambda *args: None)
    events = []
    trainer.validate_human = lambda: events.append("validate") or {}
    monkeypatch.setattr(Trainer, "_synchronize", lambda self: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))
    trainer._synchronize()
    assert events == ["sync", "validate", "release"]
    assert trainer._reported_update == trainer._cache_released_update == 50
