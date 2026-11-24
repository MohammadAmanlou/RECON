"""Validate the immutable design pack; no simulator, no test-world access."""
import collections,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
D=ROOT/'data'
load=lambda f:json.loads((D/f).read_text(encoding='utf-8'))
bank=load('question_bank.json');graph=load('candidate_graph.json');p=load('protocol.json');audit=load('source_audit.json')
ids={s['id'] for s in graph['skills']}; order=graph['topological_order']; assert set(order)==ids and len(ids)==6
assert 20<=len(bank)<=50 and len({q['id'] for q in bank})==len(bank)==len({q['question'].strip().casefold() for q in bank})
assert all(q['skills'] and set(q['skills'])<=ids and q['correct_answer_key'] in q['options'] for q in bank)
assert {q['skills'][0] for q in bank if len(q['skills'])==1}==ids
assert len({q['item_family'] for q in bank})>=p['question_budget']
projection={(a,e['child']) for e in graph['candidate_hyperedges'] for a in e['parents']};assert projection==set(map(tuple,graph['pairwise_projection']))
assert all(order.index(a)<order.index(b) for a,b in projection)
seeds=[s for split in ('development','held_out') for a in p['worlds'][split].values() for s in a];assert len(seeds)==len(set(seeds)) and p['primary']['unit']=='independent world'
for row in (D/'SHA256SUMS.txt').read_text().splitlines():
    h,f=row.split('  ',1);assert hashlib.sha256((D/f).read_bytes()).hexdigest()==h,f
if len(sys.argv)>1:
    src=pathlib.Path(sys.argv[1]);
    for f,h in audit['source_sha256'].items():assert hashlib.sha256((src/f).read_bytes()).hexdigest()==h,f
    arrays={f:json.loads((src/f).read_text(encoding='utf-8-sig')) for f in audit['question_counts']}
    for q in bank:
        orig=arrays[q['source_file']][q['source_row_zero_based']]
        for key in ('id','question','options','correct_answer_key','complexity','source_type'): assert q[key]==orig[key],(q['id'],key)
print(f'PASS: {len(bank)} original question IDs, {len(ids)} skills, {len(set(q["item_family"] for q in bank))} families, {len(seeds)} disjoint world seeds; SHA-256 verified.')
print('Tag counts:',dict(collections.Counter(s for q in bank for s in q['skills'])))
print('No learner simulation or held-out evaluation executed.')
