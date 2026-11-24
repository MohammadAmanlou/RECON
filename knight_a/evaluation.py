"""Frozen evaluation: tune on dev, persist manifest, then evaluate once on held-out."""
import argparse, hashlib, itertools, json, os
from pathlib import Path
from time import perf_counter
import numpy as np
from . import assessment as a

METHODS = (*a.METHODS, 'mixture_ig', 'irt_cat')
ROOT = Path(__file__).resolve().parent
BUDGET = a.PROTOCOL['question_budget']
N = a.PROTOCOL['worlds']['learners_per_world']
THETA = (np.arange(7)-3)/1.5

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes():
    paths = [ROOT/'assessment.py', ROOT/'evaluation.py', ROOT/'PROTOCOL_AMENDMENT.md', *sorted(a.DATA.glob('*.json')), a.DATA/'SHA256SUMS.txt']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def world(regime, seed, split):
    if seed not in a.PROTOCOL['worlds'][split][regime]: raise ValueError('Unregistered split/seed')
    # Keep generator byte-equivalent in behaviour to the phase-2 DevWorld.
    r,b,c,d = (a.rng_for(seed,ns) for ns in range(1,5))
    parents = dict(a.PARENTS)
    if regime == 'mismatch':
        child,removed = ((5,3),(2,1))[int(r.integers(2))]
        parents[child] = tuple(s for s in parents[child] if s != removed)
    u=b.normal(0,{'matched':0.,'noisy':.4,'mismatch':.6}[regime],N)
    z=np.zeros((N,len(a.SKILLS)),dtype=int)
    for j in range(len(a.SKILLS)):
        ps=parents.get(j,())
        if ps:
            x=z[:,ps]; f=x.mean(axis=1) if regime=='mismatch' else x.prod(axis=1)
            p=a.sigmoid(r.uniform(-1.8,-.6)+r.uniform(1.5,3.2)*f+u)
        else: p=r.uniform(.35,.75)
        z[:,j]=b.random(N)<p
    if regime=='mismatch':
        p=a.sigmoid(np.log(.25/.75)+2.5*(z@a.Q.T/a.Q.sum(axis=1))+.8*u[:,None]-c.uniform(-.6,.6,len(a.BANK)))
    else:
        sr,gr=((.06,.18),(.18,.30)) if regime=='matched' else ((.15,.30),(.20,.35))
        p=np.where(z@a.Q.T==a.Q.sum(axis=1),1-c.uniform(*sr,len(a.BANK)),c.uniform(*gr,len(a.BANK)))
    return z,d.random(p.shape)<p

def random_histories(seed, answers):
    rng=a.rng_for(seed,5); ids=[]
    for _ in answers:
        eligible=np.ones(len(a.BANK),dtype=bool); row=[]
        for _ in range(BUDGET):
            j=int(rng.choice(np.flatnonzero(eligible))); row.append(j); eligible &= a.FAMILIES!=a.FAMILIES[j]
        ids.append(row)
    ids=np.array(ids); return ids,np.take_along_axis(answers,ids,axis=1)

def tune(out):
    out.mkdir(parents=True,exist_ok=True); path=out/'fit_manifest.json'
    if path.exists() or (out/'heldout_summary.json').exists(): raise FileExistsError('Use a fresh output directory; never overwrite a frozen fit')
    dev=[]
    for regime,seeds in a.PROTOCOL['worlds']['development'].items():
        for seed in seeds:
            z,y=world(regime,seed,'development'); ids,obs=random_histories(seed,y); dev.append((z,ids,obs))
    z=np.concatenate([x[0] for x in dev]); ids=np.concatenate([x[1] for x in dev]); obs=np.concatenate([x[2] for x in dev])
    grid=a.PROTOCOL['inference']['development_grid']; tuples=list(itertools.product(*(grid[k] for k in ('r','b','w','slip','guess'))))
    best={}; scores={}
    for family in ('pairwise','hyperedge'):
        rows=[]
        for params in tuples:
            like=a.likelihood(params); p=a.prior(family,params)
            l=like.T[ids]; logp=np.log(p)[None,:]+np.log(np.where(obs[:,:,None],l,1-l)).sum(axis=1)
            post=np.exp(logp-logp.max(axis=1,keepdims=True)); post/=post.sum(axis=1,keepdims=True)
            loss=float(np.mean((post@a.STATES-z)**2)); rows.append({'params':list(params),'brier':loss})
        rows.sort(key=lambda x:(x['brier'],x['params'])); best[family]=rows[0]['params']; scores[family]=rows
    k=z.sum(axis=1); counts=np.bincount(k,minlength=7); skill=np.array([(z[k==i].sum(axis=0)+1)/(counts[i]+2) for i in range(7)])
    ability=(counts+1)/(len(z)+7); irt=[]
    for j in range(len(a.BANK)):
        ii,tt=np.where(ids==j); t=THETA[k[ii]]; response=obs[ii,tt]; options=[]
        for slope,intercept in itertools.product([.5,1,1.5,2],np.arange(-3,3.01,.25)):
            p=a.sigmoid(slope*t+intercept); loss=-np.sum(np.where(response,np.log(p),np.log1p(-p)))
            options.append((float(loss),slope,float(intercept)))
        _,slope,intercept=min(options); irt.append([slope,intercept])
    manifest={'protocol':'v1.0 primary + pre-evaluation amendment v1.1','hashes':hashes(),'dev_seeds':a.PROTOCOL['worlds']['development'],'fit':best,'grid_scores':scores,'irt':{'params':irt,'prior':ability.tolist(),'skill_given_ability':skill.tolist()},'development_learners':len(z),'numpy_version':np.__version__}
    save(path,manifest); print('FIT FROZEN',sha(path),best,flush=True)
    return manifest

def infer(method, fit, respond, rng):
    if method=='irt_cat':
        model=fit['irt']; params=np.array(model['params']); slopes=params[:,0]; intercepts=params[:,1]
        p=np.array(model['prior']); features=np.array(model['skill_given_ability']); like=a.sigmoid(THETA[:,None]*slopes+intercepts)
    elif method=='mixture_ig':
        families=('pairwise','hyperedge'); p=np.concatenate([.5*a.prior(f,fit['fit'][f]) for f in families]); like=np.concatenate([a.likelihood(fit['fit'][f]) for f in families]); features=np.tile(a.STATES,(2,1))
    else:
        family='hyperedge' if method=='hyperedge_ig' else 'pairwise'; params=fit['fit'][family]; p=a.prior(family,params); like=a.likelihood(params); features=a.STATES
    eligible=np.ones(len(a.BANK),dtype=bool); marginals=[p@features]; history=[]
    for _ in range(BUDGET):
        choices=np.flatnonzero(eligible)
        if method=='irt_cat':
            prob=a.sigmoid(float(p@THETA)*slopes+intercepts); scores=slopes**2*prob*(1-prob); j=int(choices[np.argmax(scores[choices])])
        elif method=='mixture_ig':
            joint=p.reshape(2,64); state=joint.sum(axis=0); cond=(joint[:,:,None]*like.reshape(2,64,-1)).sum(axis=0)/state[:,None]
            scores=a.entropy(p@like)-state@a.entropy(cond); j=int(choices[np.argmax(scores[choices])])
        else: j=a.select(method,p,like,eligible,rng)
        pred=float(p@like[:,j]); y=bool(respond(j)); p=a.posterior_update(p,like[:,j],y)
        history.append({'id':a.BANK[j]['id'],'family':str(a.FAMILIES[j]),'correct':y,'predicted':pred}); eligible &= a.FAMILIES!=a.FAMILIES[j]; marginals.append(p@features)
    return np.array(marginals),history

def ece(p,y):
    p=p.ravel(); y=y.ravel(); bins=np.minimum((p*10).astype(int),9); return sum(abs(p[bins==i].sum()-y[bins==i].sum()) for i in range(10))/len(p)
def evaluate(out):
    path=out/'fit_manifest.json'; manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest['hashes']!=hashes() or manifest['dev_seeds']!=a.PROTOCOL['worlds']['development']: raise ValueError('Frozen fit/code/data mismatch')
    if (out/'heldout_summary.json').exists() or (out/'heldout_started.json').exists(): raise FileExistsError('Held-out already started; do not rerun/tune silently')
    save(out/'heldout_started.json',{'fit_sha256':sha(path)})
    rows=[]
    with (out/'histories.jsonl').open('w',encoding='utf-8') as f:
        for regime,seeds in a.PROTOCOL['worlds']['held_out'].items():
            for seed in seeds:
                z,y=world(regime,seed,'held_out')
                for method in METHODS:
                    rng=a.rng_for(seed,5); start=perf_counter(); trajectories=[]; histories=[]
                    for i in range(N):
                        p,h=infer(method,manifest,lambda j:y[i,j],rng); trajectories.append(p); histories.append(h)
                    seconds=perf_counter()-start; m=np.array(trajectories); err=(m>=.5)!=z[:,None,:]; exact=~err.any(axis=2); first=np.where(exact.any(axis=1),exact.argmax(axis=1),BUDGET+1)
                    rows.append(dict(regime=regime,seed=seed,method=method,hamming=err.mean(axis=(0,2)).tolist(),exact=exact.mean(axis=0).tolist(),first_exact_censored=float(first.mean()),ever_exact=float((first<=BUDGET).mean()),final_brier=float(np.mean((m[:,-1]-z)**2)),ece=float(ece(m[:,-1],z)),response_brier=float(np.mean([(v['predicted']-v['correct'])**2 for h in histories for v in h])),seconds=seconds))
                    for i,h in enumerate(histories): f.write(json.dumps(dict(regime=regime,seed=seed,method=method,learner=i,history=h))+'\n')
                print('held-out',regime,seed,flush=True)
    summary={}
    for regime in a.PROTOCOL['worlds']['held_out']:
        subset=[r for r in rows if r['regime']==regime]; means={}
        for method in METHODS:
            group=[r for r in subset if r['method']==method]; means[method]={k:np.mean([r[k] for r in group],axis=0).tolist() for k in ('hamming','exact','first_exact_censored','ever_exact','final_brier','ece','response_brier','seconds')}
        contrasts={}
        for method in ('hyperedge_ig','mixture_ig'):
            diff=np.array([next(r['hamming'][-1] for r in subset if r['seed']==s and r['method']=='pairwise_ig')-next(r['hamming'][-1] for r in subset if r['seed']==s and r['method']==method) for s in a.PROTOCOL['worlds']['held_out'][regime]])
            boot=np.random.default_rng(990001).choice(diff,(10000,len(diff)),replace=True).mean(axis=1); contrasts[method]={'mean':float(diff.mean()),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'world_differences':diff.tolist()}
        summary[regime]={'means':means,'paired_contrasts':contrasts}
    save(out/'per_world.json',rows); save(out/'heldout_summary.json',{'fit_sha256':sha(path),'summary':summary,'methods':METHODS,'worlds':len(rows)//len(METHODS),'learners_per_world':N}); print('COMPLETE',out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['tune','heldout']); p.add_argument('--out',type=Path,required=True); args=p.parse_args()
    if args.stage=='tune': tune(args.out)
    else: evaluate(args.out)
if __name__=='__main__': main()
