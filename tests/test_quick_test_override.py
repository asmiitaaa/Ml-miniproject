from pathlib import Path
import subprocess
import sys


def test_explicit_checkpoint_cannot_be_hidden_by_scenario_checkpoint(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/quick_test.py'),
        '--config',str(root/'configs/evaluation.corridor.json'),'--checkpoint',str(tmp_path/'missing.zip'),
        '--output',str(tmp_path/'run')],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'real trained DQN checkpoint is required' in result.stderr
