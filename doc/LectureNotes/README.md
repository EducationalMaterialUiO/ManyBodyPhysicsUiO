# Lecture notes: the executable companion to the book

This directory is a [Jupyter Book](https://jupyterbook.org).  Each notebook is
the runnable counterpart of one chapter of *Quantum Mechanics for
Many-Particle Systems*, and each imports the corresponding program from
`../BookManybody/BookPrograms/chapterNN`, so the numbers on these pages and the
numbers in the book come from the same code.

The notebooks are also the source of the book's figures.  Re-executing them
with the harness described in `../BookManybody/BookPrograms/README.md` writes
every matplotlib figure as a vector PDF (and a 300 dpi PNG) into
`../BookManybody/BookFigures/chapterNN/`, which is where the LaTeX chapters
pick them up.  A figure that changes here therefore changes in the book, and
nowhere else does a figure have to be edited by hand.

## Building

```bash
pip install -r requirements.txt
jupyter-book build .
```

Open `_build/html/index.html`.  A full build from cold takes about four
minutes, almost all of it in `quantumdots.ipynb`.

The configuration uses `execute_notebooks: cache`, so the outputs are stored
in `_build/.jupyter_cache` and later builds only re-run the notebooks that
have changed.  To force everything to run again:

```bash
jupyter-book build --all .
```

To start completely clean:

```bash
rm -rf _build && jupyter-book build .
```

## Contents

| Notebook | Chapter | Companion program |
|---|---|---|
| `linearalgebra.ipynb` | 1, the spectral decomposition and its physics, linear algebra and eigenvalue problems | `spectral.py`, `direct_solvers.py`, `iterative_solvers.py`, `householder.py`, `lanczos.py`, `svd.py` |
| `manybodybasics.ipynb` | 2, Slater determinants and the energy functional | `slaterdeterminant.py`, `slater_update.py` |
| `wicktheorem.ipynb` | 3, Wick's theorem and its generalisation | `wick.py` |
| `jordanwigner.ipynb` | 3, from fermions to qubits | `jordanwigner.py` |
| `models.ipynb` | 4, Lipkin, pairing, Hubbard, Heisenberg, Calogero | `models.py` |
| `fullci.ipynb` | 5, full configuration interaction | `fci.py` |
| `hartreefock.ipynb` | 6, Hartree-Fock and Thouless' theorem | `hartreefock.py` |
| `tdarpa.ipynb` | 7, TDA, RPA, BCS and QRPA | `rpa.py` |
| `dft.ipynb` | 8, density functional theory | `dft.py` |
| `mbptheory.ipynb` | 9, many-body perturbation theory | `mbpt.py` |
| `coupledcluster.ipynb` | 10, coupled cluster and unitary CC | `coupledcluster.py` |
| `quantumdots.ipynb` | 11, the two-dimensional quantum dot | `quantumdot.py` |
| `dmrg.ipynb` | the DMRG chapter (after 11): matrix product states and the density matrix renormalisation group | `chapterdmrg/dmrg.py` |
| `statistics.ipynb` | 12, statistics, random walks and Metropolis | `montecarlo.py` |
| `fam.ipynb` | Appendix A, Project 1: the finite amplitude method | `appendixA/fam.py` (with `rpa.py`) |

## Regenerating the book's figures

`../BookManybody/BookFigures/chapterNN/` is written by a harness that executes
each notebook with `nbclient` and, after every code cell, saves any open
matplotlib figure as `<tag>_figNN.pdf` and `<tag>_figNN.png` together with a
`_manifest_<tag>.json` recording the axis labels, legends and producing cell.
The harness

* puts every `../BookManybody/BookPrograms/chapter*` and `appendix*` directory on `sys.path`;
* sets print-quality `rcParams` (serif, 10 pt, `savefig.dpi = 300`,
  `bbox_inches = "tight"`);
* strips the in-figure title of a single-panel figure, because in the book the
  caption carries it -- panel titles of multi-panel figures are kept, since
  those are panel labels.

Keep setting `ax.set_title(...)` in the notebooks: it is what the Jupyter-book
page shows, and the harness records it in the manifest for the book caption
before removing it from the image.

If you change a notebook, re-run the harness for that notebook only and the
chapter's figures are replaced in place; the LaTeX needs no editing unless the
number of figures changes.

## Notes for anyone editing these

* **The notebooks are executed from this directory.**  They put every
  `../BookManybody/BookPrograms/chapter*` directory on `sys.path`, which only
  resolves if the working directory is `doc/LectureNotes`.  All of them are
  added, not just the chapter's own, because several programs import a program
  belonging to another chapter.  `run_in_temp` is therefore
  set to `false` in `_config.yml`; do not change it.

* **`allow_errors` is `false`.**  A cell that raises stops the build.  That is
  deliberate: these notebooks are the regression test for the programs, and a
  silent failure would leave a wrong number on a page.

* **Watch out for notebooks written without trailing newlines.**  A cell whose
  `source` list has lines that do not end in `\n` renders as one glued-together
  line and fails with a `SyntaxError`.  To check the whole directory:

  ```python
  import json, glob
  for f in sorted(glob.glob("*.ipynb")):
      nb = json.load(open(f))
      bad = sum(1 for c in nb["cells"] for l in c["source"][:-1]
                if not l.endswith("\n"))
      if bad:
          print(f, bad, "glued lines")
  ```

* **`only_build_toc_files` is `true`**, so notebooks that are not listed in
  `_toc.yml` are ignored.  `notation.ipynb` and `gradientmethods.ipynb` predate the book and are
  currently left out; add them to `_toc.yml` if you want them on the site.
  `Hubbard.ipynb` is likewise not listed, but two of its figures (the lattice
  graph and the Hubbard/t-J comparison) are used by chapter 4; note that it
  exhausts memory if run to the end, so only its first sixteen cells are
  executed when the figures are regenerated.

* **The two expensive notebooks** are `quantumdots.ipynb` (about ninety
  seconds, dominated by the six-electron coupled-cluster runs at forty-two
  orbitals) and `statistics.ipynb` (about thirty-five seconds, dominated by
  the variational Monte Carlo).  If a build times out, raise
  `execute.timeout` in `_config.yml` rather than shrinking the calculations.

## If the build fails with `no such table: nbcache`

The execution cache in `_build/.jupyter_cache/` has gone stale (this happens
if a build is interrupted, or if the tree is copied between machines).  The
fix is to remove the build directory and start again:

```
rm -rf _build && jupyter-book build .
```

## Notebook cell ids

`nbformat` 5.1.4 and later require every cell to carry an `id`, and a missing
one is a warning now and an error later.  Notebooks generated by scripts
often lack them.  To fix every notebook in place:

```
python -c "import nbformat, glob
for p in glob.glob('*.ipynb'):
    nb = nbformat.read(p, as_version=4)
    _, nb = nbformat.validator.normalize(nb)
    nbformat.write(nb, p)"
```

`gradientmethods.ipynb` is an old nbformat 4.4 file that `normalize` cannot
repair; it is not listed in `_toc.yml`, and `only_build_toc_files: true`
keeps it out of the build.
