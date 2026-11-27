# KNIGHT Direction A — design freeze v1.0 (24 November 2026)

A self-contained design/data module for the existing RECON project. It addresses Andrey's learner-profile-aware adaptive assessment request, not RECON-Ω mechanism identification. **No simulation result yet.** No real student responses or human experiment. The bank consists of 20 genuine supplied KNIGHT-generated MCQs, not newly generated/demo questions. The topic changed from provisional probability to linear algebra because source coverage supports this topic. Questions are conceptual recognition items, not numerical proof/solution competence tests. Tags and soft prerequisites are curator hypotheses, not validated learning dependencies.

## Files and validation

- `data/question_bank.json`: 20 original question IDs, exact question/choices/key, source file and zero-based index, preserved semantic path, manual Q-tags, semantic family and annotation rationale.
- `data/kg_evidence.json`: 47 relevant semantic triples extracted from both Neo4j JSON exports; original graph node and relationship IDs, source filename and row retained. The export contained nonstandard backslash escapes; `freeze.py` repairs invalid escapes for parsing only. No need to restore native Neo4j dumps for this step.
- `data/candidate_graph.json`: six skills, three conjunctive candidate hyperedges and one unary prerequisite, seven projected pairwise edges. Not a learned causal graph.
- `data/protocol.json`: prespecified estimand, generator regimes, models, dev-only tuning, disjoint seeds, matched budget, metrics, tests and falsification criterion.
- `data/source_audit.json` and `data/SHA256SUMS.txt`: original source fingerprints and freeze fingerprints. `freeze.py` rebuilds selected questions and evidence from original exports. `validate_freeze.py` checks frozen artifacts and optionally exact source fidelity. `freeze_validation.txt` contains actual local execution output.

From the RECON root, switch to this branch and validate using Python 3 (standard library only):

```bash
git fetch origin
git switch knight-direction-a
python knight_a/validate_freeze.py
```

For a fresh checkout:

```bash
git clone --branch knight-direction-a https://github.com/MohammadAmanlou/RECON.git
cd RECON
python knight_a/validate_freeze.py
```

Optionally verify against the original files locally (do not publish source data without permission):

```bash
python knight_a/validate_freeze.py /path/to/original/KNIGHT/exports
```

To reconstruct the generated bank/evidence/audit from originals: `python knight_a/freeze.py --source-dir /path/to/original/KNIGHT/exports`; then run validation. It does not rewrite the protocol, graph or checksum manifest. Any change that alters a checksum is a new version, not a silent freeze update. The four JSON source files suffice; preserve SVG/PNG and Neo4j dumps outside repository for archival use.

## Next stage (25 Nov)

Implement `knight_a/simulator.py` (private oracle environment), `models.py`, `policies.py`, `run.py`, `tests/`. Keep oracle state/true graph/potential responses outside policy API; a policy receives only candidate graph, question tags/family, observed history, its inferred probabilities and private policy randomness. Train/tune the specified grid only on development worlds. Save parameters and code/data hashes, then run held-out once. All methods get eight distinct question families. At most 64 states, no LLM/GPU necessary. No fabricated results. The static BKT adaptation is explicitly restricted to assessment, not a full sequential learning benchmark. Primary contrast is mismatch pairwise IG minus hyperedge IG Hamming error; matched regime is only a positive control. A general pairwise graph can express conjunction via CPDs, so restrict the claim to additive-vs-conjunctive model forms.

## Repository hygiene

RECON's existing `src/recon` and historical experiments remain intact. This dedicated `knight-direction-a` branch contains only the selected frozen question subset, semantic evidence subset, protocol and supporting scripts; the full source exports, raw Neo4j dumps and graph visualizations are not published here. GitHub reports the repository as public. Do not add raw data, private credentials or other unpublished materials indiscriminately. Do not email the professor until the two-page plan and runnable demo with actual results are ready together.
