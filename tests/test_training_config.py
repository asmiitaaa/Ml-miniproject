"""V2 training runs must not overwrite source scenarios or inherit hidden defaults."""
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest


def training_module(monkeypatch):
    agent=Path(__file__).resolve().parents[1]/'rl_agent'
    monkeypatch.syspath_prepend(str(agent))
    spec=importlib.util.spec_from_file_location('v2_train_test',agent/'train.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_training_configuration_records_seed_and_strict_inputs(tmp_path,monkeypatch):
    module=training_module(monkeypatch)
    source=Path(module.SUMO_CONFIG['sumocfg_path'])
    before=source.read_bytes()
    target,hashes,resolution=module.prepare_training_inputs(tmp_path)
    config=ET.parse(target)
    assert source.read_bytes()==before
    assert resolution==.4
    assert config.find('./processing/lateral-resolution').get('value')=='0.4'
    assert config.find('./processing/ignore-route-errors').get('value')=='false'
    assert config.find('./processing/collision.action').get('value')=='teleport'
    assert config.find('./random_number/seed').get('value')=='42'
    assert config.find('./random_number/random').get('value')=='false'
    assert 'network/corridor.net.xml' in hashes
    assert 'simulation/corridor.sumocfg' in hashes
    for option in config.findall('./input/*'):
        assert all(Path(name).is_absolute() for name in option.get('value').split(','))
    assert config.find('.//collision-output').get('value')==str(tmp_path/'collisions.xml')
    assert config.find('.//lanedata-output').get('value')==str(tmp_path/'lanedata-output.xml')


def test_trainer_preserves_existing_checkpoints(tmp_path,monkeypatch):
    module=training_module(monkeypatch)
    checkpoint=tmp_path/(module.MODEL_NAME+'.zip')
    checkpoint.write_bytes(b'preserve this checkpoint')
    with pytest.raises(FileExistsError,match='fresh --output-dir'):
        module.train(timesteps=24,output_dir=tmp_path)
    assert checkpoint.read_bytes()==b'preserve this checkpoint'
