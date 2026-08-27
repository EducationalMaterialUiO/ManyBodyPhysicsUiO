#!/usr/bin/env python3
"""Regenerate the book's figures.

Execute the LectureNotes notebooks and save every matplotlib figure they
produce as a vector PDF (and a 300 dpi PNG) under BookFigures/chapterNN/.

    python3 BookPrograms/make_figures.py                # all notebooks
    python3 BookPrograms/make_figures.py dft.ipynb      # just one

A manifest (JSON) records, for each figure, the notebook, the cell index, the
source of the producing cell, and whatever axis titles/labels matplotlib knows
about, so that captions can be written from the actual content."""

import json, os, sys, shutil, glob, argparse, traceback
import nbformat
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NB = os.path.join(ROOT, "LectureNotes")
FIG = os.path.join(ROOT, "BookManybody", "BookFigures")
PROG = os.path.join(ROOT, "BookManybody", "BookPrograms")

# notebook -> (chapter number, short tag used in file names)
NOTEBOOKS = {
    "linearalgebra.ipynb":    (1,  "linalg"),
    "manybodybasics.ipynb":   (2,  "mbbasics"),
    "wicktheorem.ipynb":      (3,  "wick"),
    "jordanwigner.ipynb":     (3,  "jw"),
    "models.ipynb":           (4,  "models"),
    "Hubbard.ipynb":          (4,  "hubbard"),
    "fullci.ipynb":           (5,  "fci"),
    "hartreefock.ipynb":      (6,  "hf"),
    "tdarpa.ipynb":           (7,  "rpa"),
    "dft.ipynb":              (8,  "dft"),
    "mbptheory.ipynb":        (9,  "mbpt"),
    "coupledcluster.ipynb":   (10, "cc"),
    "quantumdots.ipynb":      (11, "qdot"),
    "statistics.ipynb":       (12, "stat"),
    "variationalmc.ipynb":    (13, "vmc"),
    "optimisation.ipynb":     (14, "opt"),
    "diffusionmc.ipynb":      (15, "dmc"),
    "manybodymc.ipynb":       (16, "mbmc"),
    "boltzmannmachines.ipynb": (17, "rbm"),
    "pinns.ipynb":            (18, "pinn"),
    "quantumfourier.ipynb":   (19, "qft"),
    "vqe.ipynb":              (20, "vqe"),
    "conclusions.ipynb":      (21, "concl"),
    # the project notebooks of the appendix: the first entry is the figure
    # directory name rather than a chapter number
    "fam.ipynb":              ("appendixA", "fam"),
}

SETUP = r'''
import os, sys, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _pattern in ("chapter*", "appendix*"):
    for _d in sorted(glob.glob(os.path.join(r"{prog}", _pattern))):
        if _d not in sys.path:
            sys.path.insert(0, _d)

plt.rcParams.update({{
    "figure.figsize":   (6.0, 3.9),
    "figure.dpi":       110,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.03,
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif"],
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.grid":        False,
    "lines.linewidth":  1.4,
    "lines.markersize": 4.5,
    "mathtext.fontset": "dejavuserif",
}})

_FIGDIR  = r"{figdir}"
_TAG     = "{tag}"
_MANIFEST = []
_COUNT   = [0]
os.makedirs(_FIGDIR, exist_ok=True)

def _describe(fig):
    info = {{"suptitle": None, "axes": []}}
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        info["suptitle"] = st.get_text()
    for ax in fig.get_axes():
        a = {{
            "title":  ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
            "nlines": len(ax.get_lines()),
            "labels": [l.get_label() for l in ax.get_lines()
                       if not l.get_label().startswith("_")],
        }}
        leg = ax.get_legend()
        if leg is not None:
            a["legend"] = [t.get_text() for t in leg.get_texts()]
        info["axes"].append(a)
    return info

def _flush(cell_index):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if not fig.get_axes():
            plt.close(fig)
            continue
        _COUNT[0] += 1
        base = "%s_fig%02d" % (_TAG, _COUNT[0])
        pdf = os.path.join(_FIGDIR, base + ".pdf")
        png = os.path.join(_FIGDIR, base + ".png")
        _info = _describe(fig)
        # In a book the caption carries the title, so strip in-figure titles.
        # Panel titles of multi-panel figures are kept: they are panel labels.
        if getattr(fig, "_suptitle", None) is not None:
            fig.suptitle("")
        if len(fig.get_axes()) == 1:
            fig.get_axes()[0].set_title("")
        try:
            fig.savefig(pdf)
            fig.savefig(png, dpi=300)
            rec = {{"base": base, "cell": cell_index,
                    "size": list(fig.get_size_inches()),
                    "npanels": len(fig.get_axes())}}
            rec.update(_info)
            _MANIFEST.append(rec)
        except Exception as exc:
            _MANIFEST.append({{"base": base, "cell": cell_index,
                               "error": repr(exc)}})
        plt.close(fig)

def _dump_manifest():
    with open(os.path.join(_FIGDIR, "_manifest_%s.json" % _TAG), "w") as fh:
        json.dump(_MANIFEST, fh, indent=1)
    print("SAVED_FIGURES", _COUNT[0])
'''


def figure_dir(chapter):
    """BookFigures/chapterNN for a chapter number, BookFigures/<name> otherwise."""
    if isinstance(chapter, int):
        return os.path.join(FIG, "chapter%02d" % chapter)
    return os.path.join(FIG, chapter)


def build_exec_copy(path, chapter, tag):
    nb = nbformat.read(path, as_version=4)
    figdir = figure_dir(chapter)
    cells = [nbformat.v4.new_code_cell(
        SETUP.format(prog=PROG, figdir=figdir, tag=tag))]
    for i, c in enumerate(nb.cells):
        cells.append(c)
        if c.cell_type == "code":
            cells.append(nbformat.v4.new_code_cell("_flush(%d)" % i))
    cells.append(nbformat.v4.new_code_cell("_dump_manifest()"))
    nb.cells = cells
    return nb


def source_map(path):
    nb = nbformat.read(path, as_version=4)
    out = {}
    for i, c in enumerate(nb.cells):
        out[i] = {"type": c.cell_type, "source": c.source}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    todo = args.notebooks or list(NOTEBOOKS)
    results = {}
    for name in todo:
        chapter, tag = NOTEBOOKS[name]
        path = os.path.join(NB, name)
        if not os.path.exists(path):
            print("MISSING", name); continue
        print("=" * 70)
        print("running", name, "->", os.path.basename(figure_dir(chapter)),
              flush=True)
        nb = build_exec_copy(path, chapter, tag)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                allow_errors=True, resources={"metadata": {"path": NB}})
        try:
            client.execute()
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "failed"
        # collect errors
        errs = []
        for i, c in enumerate(nb.cells):
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    errs.append((i, o.get("ename"), (o.get("evalue") or "")[:200]))
        figdir = figure_dir(chapter)
        mf = os.path.join(figdir, "_manifest_%s.json" % tag)
        nfig = len(json.load(open(mf))) if os.path.exists(mf) else 0
        results[name] = {"status": status, "figures": nfig, "errors": errs[:8],
                         "nerrors": len(errs)}
        print(f"--> {name}: {status}, {nfig} figures, {len(errs)} cell errors",
              flush=True)
        for e in errs[:5]:
            print("    cell %d: %s: %s" % e, flush=True)
        # save executed notebook for reference

    with open(os.path.join(FIG, "_run_summary.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps({k: {"status": v["status"], "figures": v["figures"],
                          "nerrors": v["nerrors"]} for k, v in results.items()},
                     indent=1))


if __name__ == "__main__":
    main()
