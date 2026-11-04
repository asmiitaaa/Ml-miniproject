"""Export portable training histories, empirical convergence reviews and figures."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from select_checkpoints import convergence


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    output=ROOT/'report/training';output.mkdir(parents=True,exist_ok=True)
    reviews={}
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,scenario in zip(axes,['peak','offpeak']):
        for trial,color in [('lr1e3','#2563eb'),('lr3e4','#d97706')]:
            key=scenario+'_'+trial;folder=ROOT/'results/agent_training'/key
            manifest=json.loads((folder/'training_manifest.json').read_text())
            if manifest.get('status')!='complete':raise ValueError(f'Incomplete training: {key}')
            frame=pd.read_csv(folder/'monitor.csv',comment='#')
            frame['decisions']=frame.l.cumsum()
            frame.drop(columns=['t']).to_csv(output/(key+'_episodes.csv'),index=False)
            frame['episode']=range(1,len(frame)+1)
            label=f"lr={manifest['hyperparameters']['learning_rate']:g}, exploration={manifest['hyperparameters']['exploration_fraction']:g}"
            ax.plot(frame.decisions,frame.r,color=color,alpha=.22,lw=.8)
            ax.plot(frame.decisions,frame.r.rolling(10,min_periods=10).mean(),color=color,label=label,lw=2)
            reviews[key]={'manifest':manifest,'curve':convergence(folder/'monitor.csv',manifest['actual_timesteps'])}
        ax.set(title=scenario,xlabel='Training decisions',ylabel='Episode return (higher is better)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Single-intersection training; all-lane corridor feature contract\nThin: episode return. Thick: trailing 10-episode mean. One hour per complete episode.',fontsize=11)
    fig.savefig(output/'corridor_training_curves.png',dpi=160);plt.close(fig)
    legacy=list((ROOT/'rl_agent/models/training_runs').glob('*/monitor.csv'))[0]
    frame=pd.read_csv(legacy,comment='#');frame['decisions']=frame.l.cumsum()
    frame.drop(columns=['t']).to_csv(output/'v2_episodes.csv',index=False)
    reviews['v2']={'curve':convergence(legacy,100000),'feature_note':'Legacy 13-feature primary-lane reward; absolute returns are not comparable with the new all-lane models.'}
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.plot(frame.decisions,frame.r,color='#94a3b8',lw=.8)
    ax.plot(frame.decisions,frame.r.rolling(10,min_periods=10).mean(),color='#4f46e5',lw=2)
    ax.set(title='Legacy v2 full training: 100,000 decisions',xlabel='Training decisions',ylabel='Episode return (higher is better)');ax.grid(alpha=.2)
    fig.savefig(output/'v2_training_curve.png',dpi=160);plt.close(fig)
    (output/'training_review.json').write_text(json.dumps(reviews,indent=2))
    print(json.dumps({k:v['curve'] for k,v in reviews.items()},indent=2))


if __name__=='__main__':main()
