# AutoVibe Gym Paper V1

This directory is the TMLR-style working-manuscript package selected in the
frozen protocol. It is complete enough for internal scientific review, but it
has not been submitted or publicly released.

## Files

- `manuscript.template.md` is the editable prose source. Its results token is
  replaced by the deterministic builder.
- `manuscript.md` is the generated review copy with all numeric results inserted
  from the content-hashed primary analysis.
- `references.bib` contains the checked primary-paper and dataset citations.
- `REPRODUCIBILITY.md` defines evidence layers, exact hashes, verification, and
  submission-only gates.
- `artifact-manifest.json` binds the tracked package inputs and generated
  manuscript by SHA-256.

Build or verify from the repository root:

```powershell
python -m research.build_manuscript
python -m research.build_manuscript --check
```

The generated figures remain in `research/publication/paper_v1/` and are linked
from the manuscript. Edit the template, not the generated result block.
