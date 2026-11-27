/* Exact finite-horizon diagnostic planning. Node 18+, no dependencies. */
function walsh(v) {
  const a=Array.from(v); if (!a.length || (a.length&(a.length-1))) throw Error("power of two");
  for(let d=1;d<a.length;d*=2) for(let i=0;i<a.length;i+=2*d) for(let j=0;j<d;j++){const x=a[i+j],y=a[i+j+d];a[i+j]=x+y;a[i+j+d]=x-y;}
  return a;
}

function compileItem(id,family,likelihood) {
  if(likelihood.some(p=>!(p>=0&&p<=1)))throw Error("invalid likelihood");
  const coeff=walsh(likelihood).map(x=>x/likelihood.length);
  const support=coeff.map((x,i)=>x!==0?i:-1).filter(i=>i>=0);
  // Do not threshold nonzero coefficients: exact closure up to floating arithmetic.
  if(!support.includes(0))support.push(0);
  return {id,family,likelihood:Array.from(likelihood),coeff,support,beta:1-Math.max(...likelihood)+Math.min(...likelihood)};
}

function terminal(m,k) {
  const mass=m[0]; const probabilities=[],decision=[];
  let risk=0;
  for(let j=0;j<k;j++){const s=m[1<<j];risk+=(mass-Math.abs(s))/(2*k);probabilities.push(mass>0?(1-s/mass)/2:0.5);decision.push(s<0?1:0);}
  return {risk:Math.max(0,risk),probabilities,decision};
}

function closureFactory(items,k){
  const base=[0,...Array.from({length:k},(_,j)=>1<<j)],memo=new Map();
  function need(available,depth){
    const key=depth+"|"+available.join(",");
    if(memo.has(key))return memo.get(key);
    const masks=new Set(base);
    if(depth>0)for(const i of available){
      const next=available.filter(j=>items[j].family!==items[i].family);
      for(const t of need(next,depth-1))for(const s of items[i].support)masks.add(t^s);
    }
    const result=[...masks].sort((a,b)=>a-b);memo.set(key,result);return result;
  }
  return need;
}

function stepMoments(m,item,y,masks){
  const next=Object.create(null);
  for(const t of masks){
    let value=0;
    for(const s of item.support){
      const c=y?item.coeff[s]:(s===0?1-item.coeff[s]:-item.coeff[s]);
      if(c===0)continue;
      if(m[t^s]===undefined)throw Error("closure missing "+(t^s));
      value+=c*m[t^s];
    }
    next[t]=value;
  }
  return next;
}

function contraction(items,available,depth){
  const best=new Map();
  for(const i of available){const q=items[i];best.set(q.family,Math.min(best.get(q.family)??1,q.beta));}
  return [...best.values()].sort((a,b)=>a-b).slice(0,depth).reduce((a,b)=>a*b,1);
}

function planSpectral(prior,items,k,depth,options={}){
  if(prior.length!==2**k||prior.some(x=>x<0)||Math.abs(prior.reduce((a,b)=>a+b,0)-1)>1e-8)throw Error("invalid prior");
  if(!Number.isInteger(depth)||depth<0)throw Error("invalid horizon");
  const available=options.available??items.map((_,i)=>i);
  if(new Set(available.map(i=>items[i].family)).size<depth)throw Error("budget exceeds distinct families");
  const need=closureFactory(items,k);
  const all=walsh(prior),rootMasks=need(available,depth),root=Object.fromEntries(rootMasks.map(s=>[s,all[s]]));
  const stats={nodes:0,pruned:0,maxMoments:rootMasks.length,momentReads:0};
  function visit(m,a,d){
    stats.nodes++;stats.momentReads+=Object.keys(m).length;
    const end=terminal(m,k);
    if(!d||m[0]<1e-15)return {...end,item:null};
    let best=null;
    for(const i of a){
      const rest=a.filter(j=>items[j].family!==items[i].family),masks=need(rest,d-1);
      const no=stepMoments(m,items[i],0,masks),yes=stepMoments(m,items[i],1,masks);
      const lb=(terminal(no,k).risk+terminal(yes,k).risk)*contraction(items,rest,d-1);
      if(options.prune!==false&&best&&lb>best.risk+1e-12){stats.pruned++;continue;}
      const left=visit(no,rest,d-1),right=visit(yes,rest,d-1),risk=left.risk+right.risk;
      if(!best||risk<best.risk-1e-12)best={...end,risk,item:i,children:[left,right]};
    }
    return best;
  }
  return {tree:visit(root,available,depth),stats,rootMasks};
}

function planDense(prior,items,k,depth,available=items.map((_,i)=>i)){
  let nodes=0;
  function visit(p,a,d){
    nodes++;
    const mass=p.reduce((s,x)=>s+x,0),probabilities=Array.from({length:k},(_,j)=>mass?p.reduce((s,x,i)=>s+((i&(1<<j))?x:0),0)/mass:0.5);
    const end={risk:mass*probabilities.reduce((s,x)=>s+Math.min(x,1-x),0)/k,probabilities,decision:probabilities.map(p=>p>0.5?1:0),item:null};
    if(!d||mass<1e-15)return end;
    let best=null;
    for(const i of a){
      const rest=a.filter(j=>items[j].family!==items[i].family);
      const children=[0,1].map(y=>visit(p.map((v,x)=>v*(y?items[i].likelihood[x]:1-items[i].likelihood[x])),rest,d-1));
      const risk=children[0].risk+children[1].risk;
      if(!best||risk<best.risk-1e-12)best={...end,item:i,children,risk};
    }
    return best;
  }
  return {tree:visit(prior,available,depth),nodes};
}

function treeRisk(tree,prior,items,k){
  if(tree.item===null)return prior.reduce((sum,p,x)=>sum+p*tree.decision.reduce((r,v,j)=>r+(v!==((x>>j)&1)?1:0),0)/k,0);
  return tree.children.reduce((r,child,y)=>r+treeRisk(child,prior.map((p,x)=>p*(y?items[tree.item].likelihood[x]:1-items[tree.item].likelihood[x])),items,k),0);
}
module.exports={walsh,compileItem,terminal,closureFactory,stepMoments,contraction,planSpectral,planDense,treeRisk};
