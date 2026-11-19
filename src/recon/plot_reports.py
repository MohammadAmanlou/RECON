"""Regenerate publication-style SVG plots from stored JSON result snapshots.

Charts are descriptive visualization of actually executed experiments; they do
not generate data or imply neural transfer beyond the precise recorded scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def render(root: Path, only: str = 'all') -> list[Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    produced = []
    if only in ('all', 'robustness'):
        f = root / 'robustness' / 'stress_summary.json'
        if f.exists():
            summary = json.loads(f.read_text(encoding='utf8'))
            scenes = ['calibrated','low_replication','weak_signal',
                      'recovery_interference','unmodeled_four_way']
            labels = ['Calibrated','Low data','Weak signal*',
                      'Recovery interference','Omitted order-4']
            pols = [('random','Random'),('d_optimal','D-optimal'),
                    ('adaptive_margin','Adaptive heuristic')]
            x = np.arange(len(scenes)); width = .25
            fig, ax = plt.subplots(figsize=(11, 5))
            for j, (pol, name) in enumerate(pols):
                vals = [summary['aggregate'][sc][pol]['support_f1']['mean'] for sc in scenes]
                ax.bar(x+(j-1)*width, vals, width, label=name)
            ax.set_xticks(x, labels)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(f'Planted support F1 ({summary["seeds"]} worlds per regime)')
            ax.legend(fontsize=9)
            ax.set_title('RECON controlled synthetic stress experiment')
            fig.text(.5,.015,
                     '* Nonzero true effects lie below the fixed support threshold. '
                     'All results are synthetic.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            target = root/'robustness'/'stress_support_f1.svg'
            fig.savefig(target, format='svg')
            plt.close(fig)
            produced.append(target)
    if only in ('all', 'neural'):
        f = root / 'neural' / 'neural_module_summary.json'
        if f.exists():
            summary = json.loads(f.read_text(encoding='utf8'))
            names = ['Initial', 'Skill A', 'Skill B', 'Both', 'Mask',
                     'Mask lifted', 'Projection reset', 'After relearning']
            keys = ['in_domain_empty_score','in_domain_square_score',
                    'in_domain_sine_score','in_domain_both_score',
                    'reversible_mask_score','mask_restored_zero_update_score',
                    'destructive_output_projection_reset_score',
                    'retrained_after_reset_score']
            vals = [summary['metrics'][k]['mean'] for k in keys]
            fig, ax = plt.subplots(figsize=(11,4.5))
            ax.bar(np.arange(len(keys)), vals)
            ax.set_xticks(np.arange(len(keys)), names, rotation=20, ha='right')
            ax.set_ylabel('Independent probe score = 1/(1+MSE)')
            ax.set_ylim(0,1.07)
            ax.set_title('A functional mask can suppress performance without unlearning')
            fig.text(.5,.015, f'{summary["seeds"]} trained modular-MLP seeds; '
                     'composition hardcoded; not an LLM.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, 1))
            target = root/'neural'/'neural_interventions.svg'
            fig.savefig(target, format='svg')
            plt.close(fig)
            produced.append(target)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--only', choices=['all','robustness','neural'], default='all')
    args = parser.parse_args()
    for path in render(args.root, only=args.only):
        print('Wrote:', path)

if __name__ == '__main__':
    main()