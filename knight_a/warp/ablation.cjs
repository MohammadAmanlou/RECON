const {planSpectral,planDense}=require('./spectral.cjs');
const {makeItems,priorFromGraph}=require('./experiment.cjs');
function densePruned(prior,items,k,depth){
 let nodes=0,pruned=0;
 function risk(p){const mass=p.reduce((a,b)=>a+b,0);let r=0;for(let j=0;j<k;j++){let v=0;for(let x=0;x<p.length;x++)if(x&(1<<j))v+=p[x];r+=Math.min(v,mass-v)/k;}return r;}
 function beta(a,d){const map=new Map();for(const i of a)map.set(items[i].family,Math.min(map.get(items[i].family)??1,items[i].beta));return [...map.values()].sort((a,b)=>a-b).slice(0,d).reduce((a,b)=>a*b,1);}
 function visit(p,a,d){nodes++;const end=risk(p);if(!d||p.reduce((a,b)=>a+b,0)<1e-15)return end;
 let best=Infinity;for(const i of a){const rest=a.filter(j=>items[j].family!==items[i].family),q=items[i];const children=[0,1].map(y=>p.map((v,x)=>v*(y?q.likelihood[x]:1-q.likelihood[x])));
 const lb=(risk(children[0])+risk(children[1]))*beta(rest,d-1);if(lb>best+1e-12){pruned++;continue;}
 best=Math.min(best,visit(children[0],rest,d-1)+visit(children[1],rest,d-1));}return best;}
 return {risk:visit(prior,items.map((_,i)=>i),depth),nodes,pruned};
}
function ablation(bank,graph){
 const k=graph.skills.length,edges=Array.from({length:k},()=>[]);for(const e of graph.candidate_hyperedges)edges[Number(e.child.slice(1))-1]=e.parents.map(s=>Number(s.slice(1))-1);
 const p=priorFromGraph(k,edges,"and",.5,-1.3,2.2),items=makeItems(bank,k);
 return [1,2].map(d=>{const dense=planDense(p,items,k,d),dp=densePruned(p,items,k,d),sp=planSpectral(p,items,k,d),su=planSpectral(p,items,k,d,{prune:false});
 const risks=[dense.tree.risk,dp.risk,sp.tree.risk,su.tree.risk];if(Math.max(...risks)-Math.min(...risks)>1e-10)throw Error("ablation risk mismatch");
 return {depth:d,risk:risks[0],denseNodes:dense.nodes,densePrunedNodes:dp.nodes,spectralPrunedNodes:sp.stats.nodes,spectralUnprunedNodes:su.stats.nodes,rootMoments:sp.rootMasks.length,fullStates:p.length,spectralUnprunedMomentReads:su.stats.momentReads,denseStateEntries:dense.nodes*p.length};});
}
module.exports={ablation};
if(require.main===module)console.log(JSON.stringify(ablation(require('../data/question_bank.json'),require('../data/candidate_graph.json')),null,2));
