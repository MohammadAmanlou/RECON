/* Finite fixed-model minimax regret via exact Bayes best responses and Hedge. */
const {planSpectral,treeRisk}=require('./spectral.cjs');
function mixture(priors,weights){
  return priors[0].map((_,x)=>priors.reduce((s,p,m)=>s+weights[m]*p[x],0));
}

function robustPlan(priors,items,k,horizon,iterations=24,eta=2){
  if(!priors.length||!Number.isInteger(iterations)||iterations<1||eta<=0)throw Error("invalid game");
  // Nature chooses one fixed model before assessment. The learner is never resampled.
  const M=priors.length,oracle=priors.map(p=>planSpectral(p,items,k,horizon).tree.risk);
  const logits=Array(M).fill(0),cumulative=Array(M).fill(0),trees=[],trace=[];
  let lower=-Infinity,upper=Infinity,bestCount=0;
  for(let t=0;t<iterations;t++){
    const shift=Math.max(...logits),e=logits.map(v=>Math.exp(v-shift)),z=e.reduce((a,b)=>a+b,0),w=e.map(v=>v/z);
    const tree=planSpectral(mixture(priors,w),items,k,horizon).tree;
    const regrets=priors.map((p,m)=>treeRisk(tree,p,items,k)-oracle[m]);
    trees.push(tree);
    const dual=regrets.reduce((s,r,m)=>s+w[m]*r,0);
    lower=Math.max(lower,dual);
    for(let m=0;m<M;m++){cumulative[m]+=regrets[m];logits[m]+=eta*regrets[m];}
    const primal=Math.max(...cumulative.map(v=>v/(t+1)));
    if(primal<upper){upper=primal;bestCount=t+1;}
    trace.push({iteration:t+1,lower,upper,weights:w,regrets});
  }
  // Uniform random choice among the best prefix is part of the policy, not per-question randomization.
  const policies=trees.slice(0,bestCount);
  const modelRisks=priors.map(p=>policies.reduce((s,tr)=>s+treeRisk(tr,p,items,k),0)/policies.length);
  return {policies,oracle,modelRisks,lower,upper,gap:Math.max(0,upper-lower),trace};
}
module.exports={mixture,robustPlan};
