/* Exact complementary-query counterexample, not a human/KNIGHT benchmark. */
const {compileItem,planDense}=require('./spectral.cjs');
const {robustPlan}=require('./robust.cjs');
function complementaryFixture(){
 const k=7,n=2**k;
 function prior(delta){return Array.from({length:n},(_,x)=>{const bit=j=>(x>>j)&1, parity=bit(0)===bit(1)?1:-1;let p=.25;for(let j=2;j<5;j++){const v=.5+.5*delta*parity;p*=bit(j)?v:1-v;}for(let j=5;j<7;j++)p*=bit(j)?.35:.65;return p;});}
 const items=[0,1,5,6].map(j=>compileItem("skill_"+j,"family_"+j,Array.from({length:n},(_,x)=>j<2?((x>>j)&1)?.8:.2:((x>>j)&1))));
 const priors=[prior(.8),prior(.9)];
 function posterior(w,q,y){return w.map((v,x)=>v*(y?q.likelihood[x]:1-q.likelihood[x]));}
 function risk(w){const m=w.reduce((s,v)=>s+v,0);return Array.from({length:k},(_,j)=>{const p=w.reduce((s,v,x)=>s+((x>>j)&1)*v,0);return Math.min(p,m-p)/k;}).reduce((s,v)=>s+v,0);}
 function h(p){return p<=0||p>=1?0:-p*Math.log2(p)-(1-p)*Math.log2(1-p);}
 function greedy(w,a,d,mode){
   if(!d)return risk(w);const m=w.reduce((s,v)=>s+v,0);if(m<1e-15)return 0;
   let best=a[0],score=-Infinity;
   for(const i of a){const q=items[i],c=[0,1].map(y=>posterior(w,q,y));const v=mode==="MER"?-(risk(c[0])+risk(c[1])):m*h(c[1].reduce((s,v)=>s+v,0)/m)-w.reduce((s,v,x)=>s+v*h(q.likelihood[x]),0);if(v>score+1e-12){score=v;best=i;}}
   return [0,1].reduce((s,y)=>s+greedy(posterior(w,items[best],y),a.filter(i=>i!==best),d-1,mode),0);
 }
 const robust=robustPlan(priors,items,k,2,12,4);
 const result=priors.map((p,i)=>({delta:[.8,.9][i],initial:risk(p),greedyMER:greedy(p,[0,1,2,3],2,"MER"),greedyIG:greedy(p,[0,1,2,3],2,"IG"),exact:planDense(p,items,k,2).tree.risk,warp:robust.modelRisks[i]}));
 for(const r of result){const expected=(.4+3*(1-r.delta*.36)/2+.7)/7;if(Math.abs(r.exact-expected)>1e-10||Math.abs(r.greedyMER-2.5/7)>1e-10||Math.abs(r.greedyIG-2.5/7)>1e-10||Math.abs(r.warp-r.exact)>1e-10)throw Error("complementarity fixture failed");}
 return {description:"Constructed nonhuman fixture: two noisy source skills, three parity-dependent target skills without direct items, two perfect distractors. Known interaction sign, uncertain strength. Demonstrates a greedy failure, not superiority to generic exact multi-step planning.",budget:2,result,regret:{lower:robust.lower,upper:robust.upper,gap:robust.gap}};
}
module.exports={complementaryFixture};
if(require.main===module)console.log(JSON.stringify(complementaryFixture(),null,2));
