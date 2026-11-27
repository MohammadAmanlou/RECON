// Run with Node.js >= 18: node knight_a/test_demo.cjs
const fs=require("node:fs"),assert=require("node:assert/strict"),path=require("node:path");
const source=fs.readFileSync(path.join(__dirname,"demo.html"),"utf8").split("<script>")[1].split("</script>")[0];
const elements=new Map();
class El{constructor(tag=""){this.tagName=tag;this.children=[];this.value="";this.style={};this.textContent="";this.onclick=null;}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=[...x]}click(){this.onclick?.()}}
const document={getElementById(id){if(!elements.has(id))elements.set(id,new El());return elements.get(id)},createElement(t){return new El(t)}};
document.getElementById("policy").value="hyperedge_ig";
document.getElementById("mode").value="simulation";
let exported;
const Blob=function(parts){exported=parts.join("")};
const URL={createObjectURL:()=>"blob:mock",revokeObjectURL:()=>{}};
const check=`
let checks=0;
for(const method of ["random","pairwise_static_bkt","pairwise_ig","hyperedge_ig","mixture_ig"]){
 const s=makeSession(method),t=makeSession(method),answers=simulatedAnswers([1,1,0,0,1,0],71);
 for(let i=0;i<8;i++){
   const q=s.next(),q2=t.next();assert.equal(q.j,q2.j);assert.ok(q.ig>=-1e-10);
   s.observe(answers[q.j]);t.observe(answers[q2.j]);
   assert.ok(Math.abs(s.normalized()-1)<1e-12);assert.ok(s.marg().every(x=>x>=0&&x<=1+1e-12));checks++;
 }
 assert.equal(s.next(),null);assert.equal(new Set(s.history.map(x=>x.family)).size,8);assert.deepEqual(s.history,t.history);
 const a=makeSession(method),b=makeSession(method);assert.equal(a.next().j,b.next().j);a.observe(true);b.observe(true);assert.equal(a.next().j,b.next().j);
}
return checks;`;
const checks=new Function("document","Blob","URL","assert",source+"\n"+check)(document,Blob,URL,assert);
assert.equal(checks,40);
for(let i=0;i<8;i++)elements.get("answers").children.find(x=>x.tagName==="button").click();
assert.equal(elements.get("step").textContent,"SESSION COMPLETE");
elements.get("policy").value="random";elements.get("export").click();
assert.equal(JSON.parse(exported).policy,"hyperedge_ig");assert.equal(JSON.parse(exported).history.length,8);
elements.get("mode").value="manual";elements.get("start").click();
for(let i=0;i<8;i++)elements.get("answers").children[0].click();
assert.equal(elements.get("step").textContent,"SESSION COMPLETE");
console.log("PASS: 40 engine steps; normalization, family budget, replay, prefix isolation; simulated/manual DOM-mock flows and correct export metadata.");
