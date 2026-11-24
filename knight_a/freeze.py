"""Rebuild the exact selected KNIGHT bank from local source exports. No network."""
import argparse, collections, hashlib, json, pathlib, re, unicodedata
ROOT=pathlib.Path(__file__).resolve().parent
D=ROOT/'data'
SOURCES=['generated_qa_pairs_level_1.json','generated_qa_pairs.json','records.json','records(1).json']
# Source, zero-based row, conservative pedagogical Q-tag, semantic family, annotation rationale.
SELECT=[
(0,70,['S1'],'vector_space','Definition of vector space; recognition of vector operations.'),
(0,76,['S1'],'vector_space','Recognition of vector spaces as the setting for linear transformations.'),
(0,64,['S3'],'linear_map_axioms','Recognition of the defining property of linear maps.'),
(0,113,['S3'],'linear_map_axioms','Equivalent definition; same family as row 64.'),
(0,118,['S3'],'linear_map_axioms','Identify preserved vector-space operations.'),
(0,114,['S2','S3'],'matrix_representation','Relate a matrix representation to a linear transformation.'),
(0,115,['S2'],'matrix_representation','Recognize matrix theory in the context of transformations.'),
(0,120,['S2'],'matrix_applications','Identify matrices by their applications.'),
(1,370,['S2'],'matrix_definition','Recognize the row-and-column structure; source node flag No, elementary answer manually checked.'),
(0,124,['S4','S6'],'cramer','Recognize determinants as used in Cramer’s rule.'),
(0,125,['S4','S6'],'cramer','Recognize Cramer’s rule from its determinant-based solution method.'),
(0,182,['S4','S6'],'cramer','Equivalent determinant-based theorem; same family.'),
(0,126,['S4'],'det_invertibility','Recognize determinant–invertibility relation.'),
(0,127,['S4'],'det_invertibility','Recognize determinants from their invertibility role.'),
(0,128,['S5'],'eigenvalue','Recognize eigenvalues in a transformation context; weak diagnostic specificity.'),
(1,626,['S5'],'eigenvalue','Recognize eigenvalues as scalar factors.'),
(0,131,['S5'],'eigenvector','Recognize definition of an eigenvector.'),
(1,172,['S5'],'eigenvector','Recognize the eigenvector equation.'),
(0,197,['S2','S6'],'matrix_systems','Recognize role of matrix properties in solving linear systems; coarse conceptual item.'),
(0,225,['S6'],'matrix_methods','Identify matrix methods for solving linear systems in an application.'),
]
def write(name,obj):
    (D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def graph_load(p):
    raw=p.read_text(encoding='utf-8-sig'); repairs=0
    def fix(m):
        nonlocal repairs
        if m.group()==chr(92):
            repairs+=1;return chr(92)*2
        return m.group()
    raw=re.sub(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\',fix,raw)
    return json.loads(raw),repairs

def build(source):
    D.mkdir(exist_ok=True)
    arrays=[json.loads((source/f).read_text(encoding='utf-8-sig')) for f in SOURCES[:2]]
    assert len({q['id'] for a in arrays for q in a})==sum(map(len,arrays))
    items=[]
    for si,i,tags,family,why in SELECT:
        q=arrays[si][i]; sd=q['source_details']; opts=q['options']; ans=q['correct_answer_key']
        assert set(opts)==set('ABCD') and ans in opts
        assert len(set(v.strip().casefold() for v in opts.values()))==4
        items.append(dict(id=q['id'],question=q['question'],options=opts,correct_answer_key=ans,source_file=SOURCES[si],source_row_zero_based=i,source_type=q['source_type'],complexity=q['complexity'],source_nodes=[{'name':n['name'],'fact_checked':n.get('fact_checked')} for n in sd['nodes']],source_relationships=sd['relationships'],skills=tags,item_family=family,annotation_rationale=why))
    write('question_bank.json',items)
    graphstats=[]; relevant=[]
    targets={'linear algebra','matrix theory','matrices','determinants','vector spaces','linear transformations','linear mappings','eigenvalues','eigenvectors',"cramer's rule",'properties of matrices','solving systems of linear equations','matrix methods','systems of equations','solve systems of linear equations'}
    for f in SOURCES[2:]:
        records,repairs=graph_load(source/f);nodes={};edges=set()
        for i,r in enumerate(records):
            n,m=r['n'],r['m'];rel=r['r'];nodes[n['identity']]=n;nodes[m['identity']]=m;edges.add(rel['identity']); a,b=n['properties']['name'],m['properties']['name']
            if a.casefold() in targets or b.casefold() in targets:
                relevant.append({'source_file':f,'source_row_zero_based':i,'source':a,'relation':rel['type'],'target':b,'source_node_identity':n['identity'],'target_node_identity':m['identity'],'relationship_identity':rel['identity']})
        graphstats.append({'file':f,'records':len(records),'unique_nodes':len(nodes),'unique_relationships':len(edges),'invalid_escape_repairs':repairs})
    write('kg_evidence.json',relevant)
    write('source_audit.json',{'source_sha256':{f:sha(source/f) for f in SOURCES},'question_counts':dict(zip(SOURCES[:2],map(len,arrays))),'cross_file_id_overlap':0,'graph_exports':graphstats,'selected_items':len(items),'source_node_flags_note':'Preserved exactly; not independent validation. Selected row 370 has a No node flag but simple definition is checked by curator. Graph files parsed after minimal invalid backslash repair, raw files unchanged. Native Neo4j dumps and SVGs not needed for this reproducible extraction.'})
    assert len(items)==len({q['id'] for q in items})==len({unicodedata.normalize('NFKC',q['question']).strip().casefold() for q in items})
    assert 20<=len(items)<=50
    print('Built',len(items),'original items; families:',len({q['item_family'] for q in items}),'tags:',dict(collections.Counter(s for q in items for s in q['skills'])))
    print('Graph evidence:',len(relevant),'triples; graph audit:',graphstats)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',type=pathlib.Path,required=True);args=ap.parse_args();build(args.source_dir)
