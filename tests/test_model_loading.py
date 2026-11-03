"""Regression check for the real nested-ZIP checkpoint reader failure."""
import hashlib
import io
from pathlib import Path
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models/dqn_traffic_sumo_v1_20000_steps.zip'


@pytest.mark.skipif(not MODEL.exists(),reason='Original trained checkpoint is not installed')
def test_real_checkpoint_load_preserves_every_tensor_and_original_file():
    torch=pytest.importorskip('torch')
    pytest.importorskip('stable_baselines3')
    from stable_baselines3.common import save_util
    from traffic_eval.model_loading import load_dqn
    original_zip_module=save_util.zipfile
    before=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    model=load_dqn(MODEL)
    with zipfile.ZipFile(MODEL) as source:
        weights=torch.load(io.BytesIO(source.read('policy.pth')),weights_only=True)
    loaded=model.policy.state_dict()
    assert set(loaded)==set(weights)
    assert all(torch.equal(loaded[key],weights[key]) for key in weights)
    assert model.num_timesteps==20_000
    assert model.observation_space.shape==(13,)
    assert model.action_space.n==2
    assert save_util.zipfile is original_zip_module
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==before
