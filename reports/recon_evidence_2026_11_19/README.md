# RECON-Ω preliminary experimental evidence report (12 pages)

This is a polished evidence audit with **14 vector PDF charts**, reproducible from the project's already committed `results/` data. It does not contain new pretrained-LLM experiments or fabricated improvements.

- PDF: `report.pdf`
- Editable LaTeX: `main.tex` (uses local `figures/*.pdf`)
- Reproducible charts: `make_figures.py`
- Data integrity and exact SHA-256 digests: `DATA_PROVENANCE.md`

From repository root:

```bash
python -m pip install -e ".[plot]"
python reports/recon_evidence_2026_11_19/make_figures.py --repo-root . --out reports/recon_evidence_2026_11_19/figures
cd reports/recon_evidence_2026_11_19
latexmk -pdf main.tex
```

**Evidence boundary:** The promising F1 results are calibrated planted synthetic results. The neural pilot is a trained 2-module PyTorch MLP (not an LLM) whose chain-rule composition is hardcoded. The adaptive policy is exploratory and currently not better than fixed D-optimal in matched synthetic comparisons. No universal causal structure, erasure of LLM knowledge or top-tier novelty claim is established.