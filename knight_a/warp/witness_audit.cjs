const {walsh}=require('./spectral.cjs');
const {makeItems}=require('./experiment.cjs');
function audit(items,targets,k,maxDepth=3,delta=.4){
  if(delta<=0||delta>=1)throw Error("delta must be in (0,1)");
  const N=2**k,output=[];
  for(let d=1;d<=maxDepth;d++){
    const best=targets.map(s=>({mask:s,tv:-1,questions:[]}));
    let bundles=0;
    function evaluate(ids){
      bundles++;
      const scores=targets.map(()=>0);
      for(let y=0;y<2**ids.length;y++){
        const l=Array(N).fill(1);
        for(let j=0;j<ids.length;j++)for(let x=0;x<N;x++)l[x]*=((y>>j)&1)?items[ids[j]].likelihood[x]:1-items[ids[j]].likelihood[x];
        const c=walsh(l);
        for(let i=0;i<targets.length;i++)scores[i]+=delta*Math.abs(c[targets[i]])/N;
      }
      scores.forEach((tv,i)=>{if(tv>best[i].tv+1e-12)best[i]={mask:targets[i],tv,questions:ids.map(j=>items[j].id)};});
    }
    function enumerate(start,ids,families){
      if(ids.length===d){evaluate(ids);return;}
      for(let i=start;i<items.length;i++)if(!families.includes(items[i].family))enumerate(i+1,[...ids,i],[...families,items[i].family]);
    }
    enumerate(0,[],[]);output.push({depth:d,bundles,best});
  }
  return {delta,interpretation:"maximum nonadaptive bundle total variation between uniform +/- parity perturbations; common fixed item response model; not a measure of causal prerequisite validity",output};
}
module.exports={audit};
if(require.main===module){const graph=require('../data/candidate_graph.json'),bank=require('../data/question_bank.json'),k=graph.skills.length,targets=graph.candidate_hyperedges.map(e=>[...e.parents,e.child].reduce((s,v)=>s|(1<<(Number(v.slice(1))-1)),0));console.log(JSON.stringify(audit(makeItems(bank,k),targets,k),null,2));}
