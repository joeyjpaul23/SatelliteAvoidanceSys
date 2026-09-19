# Whitepaper source

> **Under revision (2026-09-19).** A review found that the certificate the
> paper's claims rest on can pass a plan that violates a dropped constraint,
> and that unmeasured induced counts were recorded as zero. The headline
> numbers in `results-premium.tex` and `results-accel.tex` have to be
> re-measured after the fixes. See section 12 of `../docs/LIMITATIONS.md`.

`aegis-certified-fleet-cam.tex` plus two generated results includes.

```bash
pdflatex aegis-certified-fleet-cam.tex   # twice, for references
```

No TeX toolchain was available in the environment that produced this, so the
source is written against a standard `article` class with only widely-available
packages (`amsmath`, `amssymb`, `amsthm`, `booktabs`, `hyperref`, `siunitx`,
`graphicx`) and has **not been compiled here**. Compile before circulating.

Every number in `results-premium.tex` and `results-accel.tex` is reproducible:

```bash
cd ../aegis
export AEGIS_ALLOW_SYNTHETIC=1
PYTHONPATH=src python3 -m aegis.experiments benchmark --acknowledge-synthetic \
    --family induced-cascade --seeds 80 --budget-mm-s 2000 --output /tmp/headline
PYTHONPATH=src python3 -m aegis.experiments frontier --acknowledge-synthetic \
    --family induced-cascade --seed 8
```

See `../docs/session-2026-09-12-build-report.md` for the full measurement
record, including the nine things that were wrong on the first attempt.
