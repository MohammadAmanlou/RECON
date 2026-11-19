"""Regenerate only derived report figures from committed RECON pilot result files.

Data provenance: results/summary.json, results/per_world.csv,
results/robustness/stress_{summary.json,per_world.csv},
results/neural/neural_{module_summary.json,module_per_seed.csv}.
No model is trained here; figure generation is descriptive post-processing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAVY='#13293D'; BLUE='#326AA8'; TEAL='#0A827A'; ORANGE='#DA8D39'; RED='#B8595A'; PURPLE='#655F99'; GRAY='#5C6C7A'; PALE='#F2F6F8'; BORDER='#DDE5EA'
COLORS={'random':GRAY,'d_optimal':TEAL,'adaptive_margin':ORANGE}
LABELS={'random':'Uniform random','d_optimal':'D-optimal (fixed)','adaptive_margin':'Adaptive (exploratory)'}
SC={'calibrated':'Calibrated','low_replication':'Low replication','weak_signal':'Weak signal','recovery_interference':'Recovery interference','unmodeled_four_way':'Unmodeled order-4'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':12,'axes.labelsize':9,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GRAY,'ytick.color':GRAY,'grid.alpha':0.17,'pdf.fonttype':42})


def save(fig,path):
    fig.savefig(path,transparent=False,bbox_inches='tight',pad_inches=0.15)
    plt.close(fig)


def plot_main(out,initial,st,neural,neural_df):
    policies=['random','d_optimal','adaptive_margin']
    scenarios=['calibrated','low_replication','weak_signal','recovery_interference','unmodeled_four_way']
    fig,ax=plt.subplots(figsize=(9.3,2.7)); x=np.arange(3);w=.34
    vals=[initial.query('policy==@p')['tri_f1'].mean() for p in ['random','d_optimal']]
    ax.bar(x[:2]-.1, vals,width=.34,color=[GRAY,TEAL],label='Tri-channel')
    for k,p in enumerate(['random','d_optimal']):
        sub=initial.query('policy==@p'); ax.bar(x[k]+.26,sub['acquisition_only_f1'].mean(),width=.25,color=[ '#C5CDD3', '#79C2B8'][k],label='Acquisition only' if k==0 else None)
    ax.set_xticks([0.04,1.04]);ax.set_xticklabels(['Uniform random','D-optimal']);ax.set_ylabel('Edge-recovery F1');ax.set_ylim(0,1.12);ax.axhline(1,c=BORDER,lw=.8)
    for i,v in enumerate(vals): ax.text(i-.1,v+.025,f'{v:.3f}',ha='center',fontsize=8)
    ax.set_title('Initial synthetic pilot | 12 worlds, matched packet counts')
    ax.legend(frameon=False,loc='upper center',ncol=2);fig.tight_layout();save(fig,out/'01_pilot_support.pdf')

    fig,ax=plt.subplots(figsize=(9.2,3.0)); wide=initial.pivot(index='world_seed',columns='policy',values='tri_f1').sort_index()
    for _,row in wide.iterrows(): ax.plot([0,1],row[['random','d_optimal']].values,c='#AFC8C7',lw=1,alpha=.85)
    ax.scatter(np.zeros(len(wide)),wide['random'],s=42,c=GRAY,zorder=3,label='Random');ax.scatter(np.ones(len(wide)),wide['d_optimal'],s=42,c=TEAL,zorder=3,label='D-optimal');ax.set_xticks([0,1],['Uniform random','D-optimal']);ax.set_xlim(-.4,1.4);ax.set_ylim(-.04,1.07);ax.set_ylabel('Support F1 per planted world');ax.set_title('Paired comparison | each line is one independent synthetic world');ax.grid(axis='y');fig.tight_layout();save(fig,out/'02_paired_worlds.pdf')

    def scenario_bars(key,title,ylabel,fmt='{:.2f}',ylim=None,file=''): 
        fig,ax=plt.subplots(figsize=(10,3.5));x=np.arange(len(scenarios));w=.23
        for j,p in enumerate(policies):
            means=[st['aggregate'][s][p][key]['mean'] for s in scenarios]
            ax.bar(x+(j-1)*w,means,width=w,label=LABELS[p],color=COLORS[p])
        ax.set_xticks(x,[SC[s] for s in scenarios],rotation=0);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y');
        if ylim:ax.set_ylim(*ylim)
        ax.legend(frameon=False,ncol=3,loc='upper right',fontsize=8);fig.tight_layout();save(fig,out/file)
    scenario_bars('support_f1','Hypergraph support recovery across five stress conditions','Mean support F1',ylim=(0,1.14),file='03_stress_support.pdf')
    scenario_bars('minimal_packet_f1','Identifying ALL minimal sufficient packets is harder','Mean family-recovery F1',ylim=(0,1.14),file='04_stress_minimal_packets.pdf')
    scenario_bars('heldout_3channel_rmse','Interventional response prediction on held-out packets','Three-channel RMSE',file='05_stress_rmse.pdf')
    scenario_bars('heldout_4th_action_mae','Fourth-action prediction error | simulator-calibrated','Fourth-action MAE',file='06_stress_fourth_action.pdf')

    fig,axs=plt.subplots(1,2,figsize=(10.5,3.55),gridspec_kw={'wspace':.38});
    for ax,key,title,vmax,cmap in [(axs[0],'support_f1','Mean support F1',1,'YlGnBu'),(axs[1],'minimal_packet_f1','Minimal-packet F1',1,'YlOrBr')]:
        a=np.array([[st['aggregate'][s][p][key]['mean'] for p in policies] for s in scenarios]);im=ax.imshow(a,vmin=0,vmax=vmax,cmap=cmap,aspect='auto');ax.set_xticks(range(3),['Random','D-optimal','Adaptive'],rotation=20,ha='right');ax.set_yticks(range(5),[SC[s] for s in scenarios]);ax.set_title(title)
        for i in range(len(scenarios)):
            for j in range(3):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if a[i,j]>.67 else NAVY)
    fig.tight_layout();save(fig,out/'07_recovery_heatmaps.pdf')

    fig,ax=plt.subplots(figsize=(9.3,3.3));positions=np.arange(5);offsets=[-.11,.11]
    for j,comparison in enumerate(['d_optimal_minus_random','d_optimal_minus_adaptive_margin']):
        entry=[st['paired_world_bootstrap'][s][comparison] for s in scenarios];mean=np.array([e['mean_paired_delta_f1'] for e in entry]);low=np.array([e['bootstrap_world_ci_95'][0] for e in entry]);high=np.array([e['bootstrap_world_ci_95'][1] for e in entry]);ax.errorbar(mean,positions+offsets[j],xerr=np.vstack([mean-low,high-mean]),fmt='o',capsize=3,color=[TEAL,ORANGE][j],label=['D-optimal – random','D-optimal – adaptive'][j],lw=1.9)
    ax.axvline(0,color=RED,ls='--',lw=1.2);ax.set_yticks(positions,[SC[s] for s in scenarios]);ax.invert_yaxis();ax.set_xlabel('Paired difference in mean F1 (world bootstrap 95% interval)');ax.set_title('Uncertainty matters: not every apparent advantage is resolved');ax.grid(axis='x');ax.legend(frameon=False,ncol=2,loc='lower center',fontsize=8);fig.tight_layout();save(fig,out/'08_paired_bootstrap.pdf')

    conf=st['discovered_edge_confirmation'];fig,axs=plt.subplots(1,2,figsize=(9.8,3.25));
    axs[0].barh(['Selected edges certified','Negative controls falsely certified'],[conf['certified_positive'],conf['negative_controls_certified_positive']],color=[TEAL,RED],height=.56)
    axs[0].set_xlim(0,40);axs[0].set_title('Fresh, model-selected confirmation');axs[0].set_xlabel('Count across 12 synthetic worlds');
    for i,v in enumerate([conf['certified_positive'],conf['negative_controls_certified_positive']]):axs[0].text(v+.6,i,str(v),va='center')
    axs[1].bar(['Within bound','Outside bound'],[conf['true_effects_covered'],conf['total_tested']-conf['true_effects_covered']],color=[TEAL,RED],width=.55);axs[1].set_title('Planted effect interval coverage');axs[1].set_ylabel('Tested hyperedges and controls');axs[1].set_ylim(0,55)
    for i,v in enumerate([conf['true_effects_covered'],conf['total_tested']-conf['true_effects_covered']]):axs[1].text(i,v+.9,str(v),ha='center')
    fig.tight_layout();save(fig,out/'09_confirmation.pdf')

    fals=st['fourth_action_falsification'];fig,ax=plt.subplots(figsize=(8.8,2.6));x=['Calibrated null','Injected +0.08 violation'];v=[fals['calibrated_null']['rejections'],fals['deliberate_violation_plus_0.08']['rejections']];ax.bar(x,v,color=[TEAL,RED],width=.47);ax.set_ylim(0,13);ax.set_ylabel('Rejections / 12 planted worlds');ax.set_title('Pre-registered fourth-action falsification (oracle-calibrated)');
    for i,k in enumerate(v):ax.text(i,k+.4,f'{k}/12',ha='center',fontsize=11,fontweight='bold');fig.tight_layout();save(fig,out/'10_falsification.pdf')

    m=neural['metrics']; labels=['Untrained','Both skills\ntrained','Reversible\nmask','Mask removed\n(no updates)','Projection\nreset','Relearned\nafter reset','Held-out\nOOD']; keys=['in_domain_empty_score','in_domain_both_score','reversible_mask_score','mask_restored_zero_update_score','destructive_output_projection_reset_score','retrained_after_reset_score','heldout_ood_both_score'];vals=[m[k]['mean'] for k in keys]; fig,ax=plt.subplots(figsize=(10.2,3.45));x=np.arange(len(labels));bars=ax.bar(x,vals,color=[GRAY,TEAL,RED,TEAL,RED,TEAL,BLUE],width=.68);ax.set_ylim(0,1.1);ax.set_xticks(x,labels);ax.set_ylabel('Held-out score = 1 / (1 + MSE)');ax.set_title('Actual trained PyTorch modular pilot | 8 CPU seeds; NOT an LLM');ax.axhline(1,c=BORDER,lw=.8)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.025,f'{v:.3f}',ha='center',fontsize=8);fig.tight_layout();save(fig,out/'11_neural_lesion.pdf')

    fig,ax=plt.subplots(figsize=(9.2,3.25)); xs=np.arange(8)
    for key,color,label in [('in_domain_both_score',TEAL,'Fully trained'),('reversible_mask_score',RED,'Masked'),('mask_restored_zero_update_score',BLUE,'Restored, zero updates')]:
        ax.plot(xs,neural_df[key],marker='o',lw=1.65,ms=4,color=color,label=label)
    ax.set_xticks(xs,[str(i) for i in xs]);ax.set_ylim(0.42,1.08);ax.set_ylabel('Held-out score');ax.set_xlabel('Training seed');ax.set_title('Every seed recovers performance immediately when the mask is removed');ax.legend(frameon=False,ncol=3,fontsize=8,loc='lower left');fig.tight_layout();save(fig,out/'12_neural_seeds.pdf')

    fig,axs=plt.subplots(1,2,figsize=(9.9,3.3)); w=initial.pivot(index='world_seed',columns='policy',values=['tri_f1','heldout_fourth_action_mae']);
    axs[0].hist(initial.query("policy=='random'")['tri_f1'],bins=np.linspace(0,1,9),alpha=.65,color=GRAY,label='Random');axs[0].hist(initial.query("policy=='d_optimal'")['tri_f1'],bins=np.linspace(0,1,9),alpha=.70,color=TEAL,label='D-optimal');axs[0].set_xlabel('Support F1');axs[0].set_ylabel('World count');axs[0].set_title('Pilot variability across worlds');axs[0].legend(frameon=False,fontsize=8)
    axs[1].scatter(initial.query("policy=='random'")['tri_f1'],initial.query("policy=='random'")['heldout_fourth_action_mae'],color=GRAY,label='Random',alpha=.75,s=36);axs[1].scatter(initial.query("policy=='d_optimal'")['tri_f1'],initial.query("policy=='d_optimal'")['heldout_fourth_action_mae'],color=TEAL,label='D-optimal',alpha=.85,s=36);axs[1].set_xlabel('Support F1');axs[1].set_ylabel('Fourth-action MAE');axs[1].set_title('Good support need not imply low 4th-action error');axs[1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,out/'13_pilot_distributions.pdf')

    fig,ax=plt.subplots(figsize=(9.3,3.15));p=['random','d_optimal','adaptive_margin'];x=np.arange(3);w=.29
    base=[st['aggregate']['calibrated'][z]['support_f1']['mean'] for z in p];weak=[st['aggregate']['weak_signal'][z]['support_f1']['mean'] for z in p];ax.bar(x-w/2,base,width=w,color=TEAL,label='Calibrated / standard effects');ax.bar(x+w/2,weak,width=w,color=RED,label='Weak effects / fixed 0.03 threshold');ax.set_ylim(0,1.13);ax.set_xticks(x,['Random','D-optimal','Adaptive']);ax.set_ylabel('Support F1');ax.set_title('Known failure: weak true effects fall below the decision threshold');ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=8);fig.tight_layout();save(fig,out/'14_failure_signal.pdf')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo-root',type=Path,required=True);parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'figures');args=parser.parse_args();d=args.repo_root/'results';args.out.mkdir(exist_ok=True,parents=True)
    initial=pd.read_csv(d/'per_world.csv');st=json.loads((d/'robustness/stress_summary.json').read_text());neural=json.loads((d/'neural/neural_module_summary.json').read_text());neural_df=pd.read_csv(d/'neural/neural_module_per_seed.csv')
    plot_main(args.out,initial,st,neural,neural_df)
    print('Rendered',len(list(args.out.glob('*.pdf'))),'vector charts')