"""Regression checks for one-shot motion imports and preserving the active actor."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('motion_deploy', ROOT/'scripts/deploy_casbot02_motion.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_sdk_header_import_uses_filename_100hz_and_preserves_motion(tmp_path):
    source = tmp_path/'不再犹豫_100hz.data'
    raw = np.zeros((101, 61))
    raw[:, 1] = np.arange(101)*.001
    raw[:, 2] = .9
    raw[:, 12] = .1
    header = ['unused']*61
    header[0], header[60] = 'Base_Latation_X', 'waist_yaw_joint'
    with source.open('w') as stream:
        stream.write(','.join(header)+'\n')
        for row in raw:
            stream.write(','.join(map(str,row))+',\n')
    target = tmp_path/'npz'/'named.npz'
    deploy.convert_csv(source,target,'casbot02',30.,50.)
    with np.load(target) as data:
        assert data['joint_pos'].shape == (51,27)  # 1 second, not the fallback 30 Hz.
        assert float(data['input_fps'][0]) == 100.
        np.testing.assert_allclose(data['joint_pos'][:,0],.1)
        np.testing.assert_allclose(data['body_pos_w'][:,0,0],np.arange(51)*.002,atol=1e-7)
    report = json.loads(target.with_suffix('.conversion.json').read_text())
    assert report['clips'][0]['input_fps_source'] == 'filename'


def test_active_reference_record_matches_installed_actor_not_newest_export(tmp_path,monkeypatch):
    config = tmp_path/'config'/'deploy.yaml'
    bundle = config.parent/'model';bundle.mkdir(parents=True)
    config.write_text('bundle: model\n')
    (bundle/'manifest.json').write_text(json.dumps({'checkpoint_sha256':'cpu7','actor_sha256':'actor7'}))
    records = tmp_path/'outputs/casbot02_deployment'
    expected = {'applied':True,'checkpoint_sha256':'cpu7','actor_sha256':'actor7',
                'checkpoint':'cpu7/ckpt_0096000.pt','npz_dir':'current_refs','robot_assets_dir':'cpu7_assets'}
    for name,record in [('correct',expected),('wrong',dict(expected,checkpoint_sha256='old88k',actor_sha256='actor88'))]:
        folder = records/name;folder.mkdir(parents=True)
        (folder/'deployment.json').write_text(json.dumps(record))
    monkeypatch.setattr(deploy,'ROOT',tmp_path)
    assert deploy.active_deployment(config) == expected


def test_missing_menu_anchor_does_not_edit_config_or_copy_data(tmp_path,monkeypatch):
    config = tmp_path/'deploy.yaml'
    original = 'joystick_motion_order: [existing]\nmotions:\n  existing: model/existing\n'
    config.write_text(original)
    source = tmp_path/'new.npz';source.write_bytes(b'not read before menu validation')
    store = tmp_path/'converted'
    record = {'checkpoint':str(tmp_path/'actor.pt'),'npz_dir':str(tmp_path/'staging'),'robot_assets_dir':None}
    monkeypatch.setattr(deploy,'active_deployment',lambda config:record)
    monkeypatch.setattr(sys,'argv',['deploy','--npz',str(source),'--config',str(config),
                                    '--npz-store',str(store),'--after','missing'])
    with pytest.raises(ValueError,match='not in joystick_motion_order'):
        deploy.main()
    assert config.read_text() == original
    assert not store.exists()


def test_default_export_stays_cpu7_even_when_active_actor_is_cpu8(tmp_path,monkeypatch):
    config = tmp_path/'deploy.yaml'
    config.write_text('joystick_motion_order: [existing]\nmotions:\n  existing: model/existing\n')
    source = tmp_path/'new.npz';source.write_bytes(b'copied without parsing before export')
    record = {'checkpoint':str(tmp_path/'cpu8.pt'),'npz_dir':str(tmp_path/'refs'),
              'robot_assets_dir':str(tmp_path/'cpu8_assets')}
    monkeypatch.setattr(deploy,'active_deployment',lambda config:record)
    captured = {}

    class StopBeforeExport(Exception):
        pass

    def updater(ckpt,npz_dir,config,order,output,**kwargs):
        captured.update(checkpoint=ckpt,**kwargs)
        raise StopBeforeExport

    monkeypatch.setattr(deploy,'run_updater',updater)
    monkeypatch.setattr(sys,'argv',['deploy','--npz',str(source),'--config',str(config),
        '--npz-store',str(tmp_path/'store'),'--after','existing'])
    with pytest.raises(StopBeforeExport):
        deploy.main()
    assert captured['checkpoint'] == deploy.DEFAULT_CKPT.resolve()
    assert captured['robot_assets'] == deploy.DEFAULT_ROBOT_ASSETS
