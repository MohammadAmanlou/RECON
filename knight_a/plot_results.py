"""Plot saved world-averaged Hamming error; never invokes simulator."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt

def plot(root):
    data=json.loads((root/'heldout_summary.json').read_text(encoding='utf-8'))
    labels={'random':'Random','pairwise_static_bkt':'Static pairwise BKT-style','pairwise_ig':'Pairwise IG','hyperedge_ig':'Hyperedge IG','mixture_ig':'Mixture IG','irt_cat':'IRT-CAT (proxy)'}
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),sharey=True)
    for ax,(regime,v) in zip(axes,data['summary'].items()):
        for m,k in v['means'].items(): ax.plot(range(9),k['hamming'],label=labels[m],marker='o',markersize=2,linewidth=1.5)
        ax.set_title(regime.capitalize());ax.set_xlabel('Questions');ax.set_xticks([0,2,4,6,8]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean normalized Hamming error')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=(0,.16,1,1));fig.savefig(root/'hamming_curves.png',dpi=200);plt.close(fig)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);plot(parser.parse_args().out)
