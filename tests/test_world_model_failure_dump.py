"""Failure diagnostics must identify the batch and preserve the optimizer guard."""
import json
from types import SimpleNamespace

import pytest
import torch

from supermimic.data.replay_buffer import WindowBatch
from supermimic.envs.mdp.motion_library import ClipInfo
from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer
from supermimic.world_model.world_model import WorldModelCfg


class TinyModel(torch.nn.Linear):
    cfg = WorldModelCfg(hidden=(4,))

    def enable_compile(self, mode):
        pass


def make_trainer(tmp_path, device="cpu"):
    model = TinyModel(1, 1).to(device)
    library = SimpleNamespace(clips=[ClipInfo('suspect_motion', 0, 10)])
    trainer = WorldModelTrainer(model, library, WMTrainerCfg(batches_per_update=1), tmp_path)
    batch = WindowBatch(states=torch.zeros(2, 1, 67), a_clean=torch.zeros(1, 1, 27),
                        a_exec=torch.zeros(1, 1, 27), xi=torch.zeros(1, 31),
                        done=torch.zeros(1, 1, dtype=torch.bool), frame=torch.tensor([[4]]))
    batch = WindowBatch(**{key: getattr(batch, key).to(device) for key in batch.__dataclass_fields__})
    replay = SimpleNamespace(sample_windows=lambda *_: batch)
    return trainer, batch, replay


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nan_batch_is_saved_before_weights_change(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    trainer, batch, replay = make_trainer(tmp_path, device)
    batch.states[0, 0, 3] = float('nan')
    before = {key: val.clone() for key, val in trainer.wm.state_dict().items()}
    trainer.loss_on = lambda _: (torch.tensor(float('nan'), device=device, requires_grad=True), {})
    with pytest.raises(FloatingPointError, match='World model loss'):
        trainer.update(replay)
    path, = tmp_path.glob('*.json')
    report = json.loads(path.read_text())
    assert report['inputs']['states']['nonfinite'] == 1
    assert report['all_model_tensors_finite']
    assert report['frames'] == [{'global_frame':4, 'clip':'suspect_motion', 'local_frame':4}]
    saved = torch.load(path.with_suffix('.pt'), weights_only=False)
    assert torch.isnan(saved['batch']['states'][0, 0, 3])
    for key, val in trainer.wm.state_dict().items():
        torch.testing.assert_close(val, before[key])


def test_dump_error_does_not_mask_original_failure(tmp_path, capsys):
    target = tmp_path/'not_a_directory'
    target.write_text('keep')
    trainer, _, replay = make_trainer(target)
    trainer.loss_on = lambda _: (torch.tensor(float('nan'), requires_grad=True), {})
    with pytest.raises(FloatingPointError, match='World model loss'):
        trainer.update(replay)
    assert target.read_text() == 'keep'
    assert 'Could not save world model diagnostics' in capsys.readouterr().out


def test_healthy_update_does_not_create_diagnostics(tmp_path):
    trainer, _, replay = make_trainer(tmp_path)
    trainer.loss_on = lambda _: (trainer.wm.weight.square().sum(), {})
    metrics = trainer.update(replay)
    assert metrics['loss'] >= 0
    assert not list(tmp_path.iterdir())
