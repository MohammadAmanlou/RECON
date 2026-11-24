"""Audit saved evaluation artifacts without running any test world."""
import argparse, collections, json
from pathlib import Path
import numpy as np
from . import assessment as a, evaluation as e

def audit(p):
    fit=json.loads((p/'fit_manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((p/'heldout_summary.json').read_text(encoding='utf-8'))
    rows=json.loads((p/'per_world.json').read_text(encoding='utf-8'))
    assert fit['hashes']==e.hashes() and summary['fit_sha256']==e.sha(p/'fit_manifest.json')
    expected={(reg,s,m) for reg,seeds in a.PROTOCOL['worlds']['held_out'].items() for s in seeds for m in e.METHODS}
    assert len(rows)==len(expected) and {(x['regime'],x['seed'],x['method']) for x in rows}==expected
    counts=collections.Counter(); seen=set(); bank={q['id']:q['item_family'] for q in a.BANK}
    for line in (p/'histories.jsonl').open(encoding='utf-8'):
        v=json.loads(line); h=v['history']; key=(v['regime'],v['seed'],v['method']); learner=(*key,v['learner'])
        assert key in expected and 0<=v['learner']<e.N and learner not in seen; seen.add(learner)
        assert len(h)==e.BUDGET and len({x['family'] for x in h})==e.BUDGET
        assert all(bank[x['id']]==x['family'] and 0<=x['predicted']<=1 for x in h)
        counts[key]+=1
    assert set(counts)==expected and set(counts.values())=={e.N}
    for regime,v in summary['summary'].items():
        for method,k in v['means'].items():
            group=[x for x in rows if x['regime']==regime and x['method']==method]
            assert len(group)==20
            for metric in k: assert np.allclose(k[metric],np.mean([x[metric] for x in group],axis=0))
        for method,c in v['paired_contrasts'].items():
            diff=np.array([next(r['hamming'][-1] for r in rows if r['regime']==regime and r['seed']==s and r['method']=='pairwise_ig')-next(r['hamming'][-1] for r in rows if r['regime']==regime and r['seed']==s and r['method']==method) for s in a.PROTOCOL['worlds']['held_out'][regime]])
            boot=np.random.default_rng(990001).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
            assert np.allclose(c['world_differences'],diff) and np.isclose(c['mean'],diff.mean()) and np.allclose(c['ci95'],np.percentile(boot,[2.5,97.5]))
    print(f'PASS: frozen hashes, {len(rows)} world-method rows, {len(seen)} unique learner histories, {len(seen)*e.BUDGET} item events; saved means and paired intervals verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);audit(p.parse_args().out)
