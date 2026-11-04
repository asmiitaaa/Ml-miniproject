"""Package validation-selected models with provenance, verifying unchanged tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))


def weights_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.policy.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(args):
    import torch
    from traffic_eval.model_loading import load_dqn
    from traffic_eval.runner import sha256
    from select_checkpoints import convergence
    torch.set_num_threads(1)
    folder=Path(args.selection).resolve().parent
    selection=json.loads(Path(args.selection).read_text())
    plan=json.loads((folder/'selection_plan.json').read_text())
    if selection['failures']:raise ValueError('Validation failures must be resolved before publishing')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    records={}
    for scenario,info in selection['scenarios'].items():
        selected=info['selected']
        if selected is None:raise ValueError(f'No eligible validation checkpoint for {scenario}')
        candidate=next(c for c in plan['candidates'] if c['id']==selected['candidate'])
        full_steps=max(c['steps'] for c in plan['candidates'] if c['monitor']==candidate['monitor'])
        review={'scenario':scenario,'source_checkpoint_sha256':selected['checkpoint_sha256'],
            'validation_seeds':selection['validation_seeds'],'reserved_test_seeds':selection['test_seeds'],
            'selection_rule':selection['selection_rule'],'validation_mean_waiting_s':selected['mean_waiting_s'],
            'validation_collision_events':selected['collision_events'],'validation_teleports':selected['teleports'],
            'full_training_curve':convergence(ROOT/candidate['monitor'],full_steps),
            'selected_checkpoint_curve':selected['convergence']}
        source=ROOT/selected['checkpoint']
        if sha256(source)!=selected['checkpoint_sha256']:raise ValueError('Candidate changed after selection')
        name=('dqn_traffic_sumo_v2_' if args.family=='v2' else 'corridor_')+scenario+'_best.zip'
        target=output/name
        if target.exists():raise FileExistsError(f'Preserve existing published model: {target}')
        model=load_dqn(source);before=weights_digest(model)
        model.selection_review=review;model.training_status='validation-reviewed'
        model.save(target)
        reloaded=load_dqn(target)
        if weights_digest(reloaded)!=before:raise RuntimeError('Tensor mismatch after attaching validation metadata')
        records[scenario]=dict(checkpoint=str(target.relative_to(ROOT)).replace('\\','/'),
            checkpoint_sha256=sha256(target),policy_tensors_sha256=before,num_timesteps=int(model.num_timesteps),
            tensor_roundtrip_identical=True,review=review)
    (folder/'published_models.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',required=True);parser.add_argument('--family',choices=['v2','corridor'],required=True)
    parser.add_argument('--output',default='models');main(parser.parse_args())
