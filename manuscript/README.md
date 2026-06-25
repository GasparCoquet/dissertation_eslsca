# Manuscript sources

The full reconstructed dissertation is built from the editable markdown sections in
`sections/` by `../code/build_dissertation.py`, which emits **both**:

- `../Dissertation_Coquet_FINAL.docx` — editable Word file (with an auto-updating Table of
  Contents: open it, right-click the TOC, choose *Update Field*).
- `../Dissertation_Coquet_FINAL.pdf` — rendered PDF.

## Why a reconstruction
No editable source of the original dissertation existed (only PDFs). Chapters 1–2 and the
References were therefore **extracted from `Dissertation_Gaspar_Coquet_RL_IS.pdf`** and cleaned of
PDF-extraction artefacts and of the fabricated data/results/interview claims, preserving the
author's wording. Chapter 3 (methodology), Chapter 4 (results) and the Abstract were **rewritten
honestly** around the real, reproducible simulation study in `../code/`. Appendix B reports the
actual hyperparameters.

## Rebuild
```bash
pip install python-docx fpdf2 matplotlib
python ../code/build_dissertation.py        # reads sections/, writes the .docx and .pdf
```

## Editing
Edit the markdown in `sections/` (plain text: `#`/`##`/`###` headings, `**bold**`, `- ` bullets,
`> ` quotes, pipe tables, `![caption](path)` figures) and rebuild. The Chapter-4 figures are
pulled from `../code/figures/`.

## Still your responsibility before submission
- Proofread Chapters 1–2 for any residual PDF-extraction glitches (equations in Ch.2 were images
  in the original and are rendered here as plain-text notation — re-insert formal equations if your
  school requires them).
- Confirm with your supervisor that a **calibrated-simulation** study (rather than a real-data
  backtest) is acceptable for your jury, and that the honest reframing is signed off.
- Re-insert any institutional front-matter your school mandates (declaration, acknowledgements,
  list of figures/tables) — only Appendix B is included here.
