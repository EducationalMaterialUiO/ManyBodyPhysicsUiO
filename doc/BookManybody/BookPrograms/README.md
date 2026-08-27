# BookPrograms

Every program of the book, one directory per chapter.  Each file runs as a
script and prints the tables that appear in the corresponding chapter, and each
is also imported by the chapter's notebook in `../../LectureNotes/`, so the
numbers on the page and the numbers on the Jupyter-book site come from the same
code.

All of them run on `numpy` alone unless the table below says otherwise.  Some
programs import a program from another chapter (`mbpt.py` uses `fci.py`,
`hartreefock.py` and `rpa.py`; `dmc.py` uses `vmc.py` and `vmcoptimise.py`;
`manybodymc.py` uses `vmc.py`, `slater_update.py` and `vmcoptimise.py`;
`vqe.py` uses `qpe.py` and `jordanwigner.py`), so put every chapter directory
on the path rather than a single one -- and the project directory of the
appendix (`appendixA`, whose `fam.py` uses `rpa.py`) as well:

```python
import sys, os, glob
for pattern in ('chapter*', 'appendix*'):
    for d in sorted(glob.glob(os.path.join('BookPrograms', pattern))):
        sys.path.insert(0, d)
```

That is exactly what the notebooks do.

| Chapter | Directory | Programs |
|---|---|---|
| 1 | `chapter01` | `spectral.py`, `direct_solvers.py`, `iterative_solvers.py`, `householder.py`, `lanczos.py`, `svd.py`, `schrodinger_diagonalization.py` |
| 2 | `chapter02` | `slaterdeterminant.py`, `slater_update.py` |
| 3 | `chapter03` | `wick.py`, `jordanwigner.py` |
| 4 | `chapter04` | `models.py` |
| 5 | `chapter05` | `fci.py` |
| 6 | `chapter06` | `hartreefock.py` |
| 7 | `chapter07` | `rpa.py` |
| 8 | `chapter08` | `dft.py` |
| 9 | `chapter09` | `mbpt.py` |
| 10 | `chapter10` | `coupledcluster.py`, `ucc.py`, `CCD_PairingModel.py`, `NeutronMatterCCD_Ladders.py` |
| 11 | `chapter11` | `quantumdot.py`, `ho1dim.py`, `ho2dim.py` |
| 12 | `chapter12` | `montecarlo.py` |
| 13 | `chapter13` | `vmc.py` |
| 14 | `chapter14` | `vmcoptimise.py` |
| 15 | `chapter15` | `dmc.py` |
| 16 | `chapter16` | `manybodymc.py` |
| 17 | `chapter17` | `rbm.py` |
| 18 | `chapter18` | `pinn.py` |
| 19 | `chapter19` | `qpe.py` |
| 20 | `chapter20` | `vqe.py` |
| 21 | `chapter21` | -- (no programs; see the notebook `conclusions.ipynb`) |
| A | `appendixA` | `fam.py` (Project 1: the finite amplitude method, built on `rpa.py`) |

Figures are collected the same way, in `../BookFigures/chapterNN/` (and
`../BookFigures/appendixA/` for the projects), and are produced by re-executing
the notebooks; see `../../LectureNotes/README.md`.
