const {walsh,compileItem,planSpectral,planDense,treeRisk}=require('./spectral.cjs');
const {robustPlan}=require('./robust.cjs');
function tests(){
  let count=0;function check(ok,msg){count++;if(!ok)throw Error(msg);}
  const close=(a,b)=>Math.abs(a-b)<1e-10;
  let s=1729;const r=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};
  for(let trial=0;trial<12;trial++){
    const k=3,N=2**k,v=Array.from({length:N},()=>.1+r()),sum=v.reduce((a,b)=>a+b,0),p=v.map(x=>x/sum);
    const items=Array.from({length:4},(_,i)=>{const mask=[1,2,4,3][i];return compileItem("q"+i,i===3?"f0":"f"+i,Array.from({length:N},(_,x)=>(x&mask)===mask?.7+.2*r():.1+.2*r()));});
    const fw=walsh(walsh(v));check(fw.every((x,i)=>close(x,N*v[i])),"Walsh involution");
    for(let d=0;d<=2;d++){
      const full=planDense(p,items,k,d),a=planSpectral(p,items,k,d),b=planSpectral(p,items,k,d,{prune:false});
      check(close(full.tree.risk,a.tree.risk),"spectral vs independent dense");
      check(close(a.tree.risk,b.tree.risk),"pruning changed optimum");
      check(close(treeRisk(a.tree,p,items,k),a.tree.risk),"tree replay");
      function walk(node,seen){if(node.item===null)return;const f=items[node.item].family;check(!seen.includes(f),"duplicate family");for(const ch of node.children)walk(ch,[...seen,f]);}
      walk(a.tree,[]);
    }
  }
  const r0=robustPlan([[.9,.1],[.1,.9]],[],1,0,20,4);
  check(close(r0.upper,.4)&&close(r0.lower,.4),"known minimax solution");
  const k=3,N=8,ps=[-1,1].map(sign=>Array.from({length:N},(_,x)=>(1+sign*.4*((x&1)?-1:1)*((x&2)?-1:1)*((x&4)?-1:1))/N));
  for(let x=0;x<4;x++)check(close(ps[0][x]+ps[0][x+4],ps[1][x]+ps[1][x+4]),"blind pairwise marginal");
  check(close(ps[0][0]/(ps[0][0]+ps[0][4]),.3),"opposite diagnostic posterior");
  check(close(ps[1][0]/(ps[1][0]+ps[1][4]),.7),"opposite diagnostic posterior");
  // Identical prior and available bank imply same plan, irrespective of hidden test world.
  const items=[compileItem("a","a",[.2,.85,.2,.85,.2,.85,.2,.85]),compileItem("b","b",[.2,.2,.85,.85,.2,.2,.85,.85])];
  const makePolicy=()=>planSpectral(ps[0],items,k,2).tree;
  const differentHiddenWorlds=[ps[0],ps[1]];
  const traces=differentHiddenWorlds.map(()=>JSON.stringify(makePolicy()));
  check(traces[0]===traces[1],"planner must not accept hidden world");
  const rr=robustPlan(ps,items,k,2,16,4);check(rr.lower<=rr.upper+1e-10,"game bound order");check(rr.modelRisks.every((x,m)=>x>=rr.oracle[m]-1e-10),"oracle lower bound");
  return count;
}
console.log('Passed '+tests()+' assertions');
