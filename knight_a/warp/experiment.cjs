/* Exploratory fixed-budget KNIGHT experiment. No tuning on evaluation worlds. */
const {compileItem,planSpectral,planDense}=require('./spectral.cjs');
const {mixture,robustPlan}=require('./robust.cjs');
function rng(seed){let s=seed>>>0;return ()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};}

function sigmoid(x){return 1/(1+Math.exp(-x));}

function priorFromGraph(k,edges,mode,root,bias,strength){
  const p=Array(2**k).fill(0);
  for(let x=0;x<p.length;x++){
    let w=1;
    for(let j=0;j<k;j++){
      const parents=edges[j]||[],v=parents.map(i=>(x>>i)&1);
      const u=v.length?sigmoid(bias+strength*(mode==="and"?v.reduce((a,b)=>a*b,1):v.reduce((a,b)=>a+b,0)/v.length)):root;
      w*=((x>>j)&1)?u:1-u;
    }
    p[x]=w;
  }
  return p;
}

function makeItems(bank,k){
  return bank.slice().sort((a,b)=>a.id.localeCompare(b.id)).map(q=>{
    const mask=q.skills.reduce((s,v)=>s|(1<<(Number(v.slice(1))-1)),0);
    return {...compileItem(q.id,q.item_family,Array.from({length:2**k},(_,x)=>(x&mask)===mask?.85:.2)),mask};
  });
}

function entropy(p){return p<=0||p>=1?0:-p*Math.log2(p)-(1-p)*Math.log2(1-p);}

function update(p,item,y){
  const v=p.map((w,x)=>w*(y?item.likelihood[x]:1-item.likelihood[x])),s=v.reduce((a,b)=>a+b,0);
  return v.map(x=>x/s);
}

function marginals(p,k){return Array.from({length:k},(_,j)=>p.reduce((s,v,x)=>s+((x>>j)&1)*v,0));}

function risk(p,k){return marginals(p,k).reduce((s,x)=>s+Math.min(x,1-x),0)/k;}

function greedy(p,items,a,k,mode,random){
  if(mode==="random")return a[Math.floor(random()*a.length)];
  let best=a[0],score=-Infinity;
  for(const i of a){
    const q=items[i],py=p.reduce((s,v,x)=>s+v*q.likelihood[x],0);
    const v=mode==="mer"?-(py*risk(update(p,q,1),k)+(1-py)*risk(update(p,q,0),k)):entropy(py)-p.reduce((s,v,x)=>s+v*entropy(q.likelihood[x]),0);
    if(v>score+1e-12){best=i;score=v;}
  }
  return best;
}

function draw(p,r){let z=r();for(let i=0;i<p.length;i++){z-=p[i];if(z<=0)return i;}return p.length-1;}

function makeWorld(seed,regime,k,edges){
  const r=rng(seed),root=.35+.3*r(),b=-1.8+.9*r(),w=1.6+1.4*r(),g=.12+.16*r(),slip=.1+.16*r();
  let graph=edges.map(a=>a.slice());
  if(regime==="shift"){graph[4]=[1];graph[5]=[0,3];}
  const p=priorFromGraph(k,graph,regime==="additive"?"mean":"and",root,b,w);
  return {p,response(item,x){const n=item.mask.toString(2).replace(/0/g,"").length,m=(item.mask&x).toString(2).replace(/0/g,"").length;
    return regime==="shift"?g+(1-g-slip)*(m/n):g+(1-g-slip)*(m===n?1:0);},seed};
}

function simulate({prior,mode,tree},items,k,budget,x,ys,r){
  let p=prior,a=items.map((_,i)=>i),node=tree,first=budget+1,final=null;
  for(let t=0;t<budget;t++){
    const i=node?node.item:greedy(p,items,a,k,mode,r);
    if(!a.includes(i))throw Error("invalid action");
    const y=ys[i];p=update(p,items[i],y);
    if(node)node=node.children[y];
    a=a.filter(j=>items[j].family!==items[i].family);
    const probs=node?node.probabilities:marginals(p,k);
    const dec=node?node.decision:probs.map(v=>v>.5?1:0);
    const errors=dec.reduce((s,v,j)=>s+(v!==((x>>j)&1)?1:0),0);
    if(!errors&&first===budget+1)first=t+1;
    final={hamming:errors/k,exact:Number(!errors),brier:probs.reduce((s,v,j)=>s+(v-((x>>j)&1))**2,0)/k,first,probs};
  }
  return final;
}

function pairedCI(rows,a,b,regime){
  const rs=rows.filter(r=>r.regime===regime),d=rs.map(r=>r.metrics[a].hamming-r.metrics[b].hamming);
  const r=rng(97123),v=Array.from({length:2000},()=>d.reduce(s=>s+d[Math.floor(r()*d.length)],0)/d.length).sort((a,b)=>a-b);
  return {mean:d.reduce((a,b)=>a+b,0)/d.length,lo:v[50],hi:v[1949]};
}

function experiment(bank,graph){
  const k=graph.skills.length,edges=Array.from({length:k},()=>[]);
  for(const e of graph.candidate_hyperedges)edges[Number(e.child.slice(1))-1]=e.parents.map(s=>Number(s.slice(1))-1);
  const items=makeItems(bank,k),budget=2,iterations=16;
  const pair=priorFromGraph(k,edges,"mean",.5,-1.3,2.2),hyper=priorFromGraph(k,edges,"and",.5,-1.3,2.2);
  const altered=edges.map(a=>a.slice());altered[4]=[1];altered[5]=[0,3];
  const models=[pair,hyper,priorFromGraph(k,altered,"and",.5,-1.3,2.2),Array(2**k).fill(1/(2**k))];
  const avg=mixture(models,models.map(()=>1/models.length));
  const planned=planSpectral(avg,items,k,budget),dense=planDense(avg,items,k,budget);
  if(Math.abs(planned.tree.risk-dense.tree.risk)>1e-10)throw Error("exact planner check failed");
  const robust=robustPlan(models,items,k,budget,iterations,4);
  const names=["random","pairwise_IG","hyperedge_IG","mixture_IG","mixture_MER","mixture_plan","WARP_regret"];
  const rows=[];
  for(const [g,regime] of ["conjunctive","additive","shift"].entries())for(let w=0;w<12;w++){
    const seed=270001+1000*g+w,world=makeWorld(seed,regime,k,edges),r=rng(seed+800000);
    const metrics=Object.fromEntries(names.map(n=>[n,{hamming:0,exact:0,brier:0,first:0,bins:Array.from({length:10},()=>[0,0,0])}]));
    for(let l=0;l<64;l++){
      const x=draw(world.p,r),ys=items.map(q=>Number(r()<world.response(q,x)));
      const lr=rng(seed*101+l+700000);
      const policy=[{prior:avg,mode:"random"},{prior:pair,mode:"ig"},{prior:hyper,mode:"ig"},{prior:avg,mode:"ig"},{prior:avg,mode:"mer"},{prior:avg,tree:planned.tree},{prior:avg,tree:robust.policies[Math.floor(lr()*robust.policies.length)]}];
      for(let m=0;m<names.length;m++){
        const z=simulate(policy[m],items,k,budget,x,ys,rng(seed*31+l)),s=metrics[names[m]];
        for(const key of ["hamming","exact","brier","first"])s[key]+=z[key]/64;
        z.probs.forEach((p,j)=>{const b=s.bins[Math.min(9,Math.floor(p*10))];b[0]++;b[1]+=p;b[2]+=(x>>j)&1;});
      }
    }
    for(const s of Object.values(metrics)){s.ece=s.bins.reduce((v,b)=>v+Math.abs(b[1]-b[2]),0)/(64*k);delete s.bins;}
    rows.push({seed,regime,metrics});
  }
  const summary=[];
  for(const regime of ["conjunctive","additive","shift"])for(const method of names){
    const a=rows.filter(r=>r.regime===regime);
    summary.push({regime,method,...Object.fromEntries(["hamming","exact","brier","first","ece"].map(key=>[key,a.reduce((s,r)=>s+r.metrics[method][key],0)/a.length]))});
  }
  return {protocol:{budget,iterations,eta:4,worlds:rows.length,learnersPerWorld:64,calibration:"mean world 10-bin ECE; not a coverage guarantee",tuning:"none; fixed illustrative parameters, exploratory independent generators, not confirmatory"},bank:{items:items.length,families:new Set(items.map(i=>i.family)).size,skills:k},planner:{risk:planned.tree.risk,denseRisk:dense.tree.risk,denseNodes:dense.nodes,...planned.stats},game:{oracle:robust.oracle,modelRisks:robust.modelRisks,lower:robust.lower,upper:robust.upper,gap:robust.gap,policies:robust.policies.length},summary,contrasts:["conjunctive","additive","shift"].map(regime=>({regime,warpVsMixtureMER:pairedCI(rows,"mixture_MER","WARP_regret",regime),warpVsMixturePlan:pairedCI(rows,"mixture_plan","WARP_regret",regime),pairwiseVsHyperedge:pairedCI(rows,"pairwise_IG","hyperedge_IG",regime)})),rows};
}
module.exports={experiment,makeItems,priorFromGraph,simulate};
if(require.main===module){const fs=require('node:fs');const bank=require('../data/question_bank.json'),graph=require('../data/candidate_graph.json');const result=experiment(bank,graph);const out=process.argv[2];if(out)fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');else console.log(JSON.stringify(result,null,2));}
