"""ccsd_diagrams.py -- the diagrammatic CCSD equations of chapter 10.

Three things are done here, all with numpy only:

1. The three energy diagrams, the fourteen singles diagrams and the
   thirty-one doubles diagrams of Section "Diagrammatic derivation of the
   coupled-cluster equations" are evaluated term by term for a random
   Hamiltonian (non-diagonal Fock matrix) and random antisymmetric amplitudes.
2. They are compared with the factored Stanton-Gauss-Watts-Bartlett form of
   the same equations, and with the brute-force projections of
   exp(-T) H exp(T) computed in the full space of determinants (bit strings),
   which is independent of any coupled-cluster algebra.
3. The TikZ code of the figures (the elements, the energy, T1 and T2
   diagrams) is regenerated with  python3 ccsd_diagrams.py --tikz DIR .
   The styles pline/hline/vint/vtx/fvtx/amp/dlab/fcidiagram are those of
   book.tex.

Run as a script: prints the three comparisons (all agree to ~1e-15).
"""
import sys, os
import numpy as np
import itertools

HOLES = set("ijklmn"); PARTS = set("abcdef")
EXT = set("ijab")

def term_pictures(H, amps, externals=EXT, scale=0.6, label=None, dangle=False, labels_ext=True):
    """H = ('f', out, in) or ('v', (p,r), (q,s)); amps = list of bars, each a list of (hole, particle) vertices.
    Lines are drawn bottom->top: pline/hline styles put the arrows right."""
    L = []
    # ---- T bars
    slots = {}          # letter -> (x, kind)
    bars = []
    x = 0.0
    for bar in amps:
        nv = len(bar)
        xs = [x + 0.9 * k for k in range(nv)]
        bars.append((xs[0] - 0.4, xs[-1] + 0.4))
        for xv, (hl, pl) in zip(xs, bar):
            slots[hl] = (xv - 0.22, 'h'); slots[pl] = (xv + 0.22, 'p')
        x = xs[-1] + 0.4 + 0.5 + 0.4
    width = (bars[-1][1] - bars[0][0]) if bars else 1.2
    xmid = (bars[0][0] + bars[-1][1]) / 2 if bars else 0.0
    # ---- H vertices: list of (letters) each with (creator, annihilator)
    if H[0] == 'f':
        verts = [(H[1], H[2])]
    else:
        verts = [H[1], H[2]]
    yH = 1.35; ytop = 2.15
    def classify(v):
        """returns list of (letter, kind, direction): direction 'down' (internal, from below) or 'up' (external)"""
        c, a = v
        out = []
        out.append((c, 'p', 'up') if c in PARTS else (c, 'h', 'down'))
        out.append((a, 'p', 'down') if a in PARTS else (a, 'h', 'up'))
        return out
    desired = []
    for v in verts:
        xs_ = [slots[l][0] for l, k, d in classify(v) if d == 'down' and l in slots]
        desired.append(sum(xs_) / len(xs_) if xs_ else None)
    if H[0] == 'f':
        xv = [desired[0] if desired[0] is not None else xmid]
    else:
        d0, d1 = desired
        if d0 is None and d1 is None:
            xv = [xmid - 0.6, xmid + 0.6]
        elif d0 is None or d1 is None:
            dk = d1 if d0 is None else d0
            # externals from T slots that will run straight up
            internal = {l for v_ in verts for (l, k_, d_) in classify(v_) if d_ == 'down' and l in slots}
            text = [xs_ for l, (xs_, k_) in slots.items() if l not in internal]
            def score(xf):
                cand = [xf - 0.33, xf + 0.33]
                return min([abs(c - t) for c in cand for t in text] + [0.45]) - 0.05 * abs(xf - dk)
            cands = [dk + sgn * sep for sep in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8) for sgn in (1, -1)]
            xf = max(cands, key=lambda c: (round(score(c), 3), c))
            xv = [xf, dk] if d0 is None else [dk, xf]
        else:
            if d0 > d1: verts = verts[::-1]; d0, d1 = d1, d0
            c = (d0 + d1) / 2; sep = max(0.9, d1 - d0)
            xv = [c - sep / 2, c + sep / 2]
    # draw bars
    for (x0, x1) in bars:
        L.append(f"\\draw[amp] ({x0:.2f},0) -- ({x1:.2f},0);")
    # H vertices
    if H[0] == 'f':
        L.append(f"\\node[fvtx] (h0) at ({xv[0]:.2f},{yH}) {{}};")
    else:
        L.append(f"\\node[vtx] (h0) at ({xv[0]:.2f},{yH}) {{}}; \\node[vtx] (h1) at ({xv[1]:.2f},{yH}) {{}}; \\draw[vint] (h0) -- (h1);")
    used = set()
    for k, v in enumerate(verts):
        cls = classify(v)
        ups = [c for c in cls if c[2] == 'up']
        for (l, kind, d) in cls:
            sty = 'pline' if kind == 'p' else 'hline'
            if d == 'down':
                if l in slots:
                    xs_, _ = slots[l]; used.add(l)
                    L.append(f"\\draw[{sty}] ({xs_:.2f},0) .. controls ({xs_:.2f},0.7) and ({xv[k]:.2f},0.7) .. (h{k}.center);")
                elif dangle:
                    dx = -0.3 if kind == 'h' else 0.3
                    L.append(f"\\draw[{sty}] ({xv[k]+dx:.2f},{yH-1.0:.2f}) node[dlab,below] {{${l}$}} -- (h{k}.center);")
            else:
                dx = 0.0 if len(ups) == 1 else (-0.33 if kind == 'h' else 0.33)
                lab = f" node[dlab,above] {{${l}$}}" if ((labels_ext or dangle) and l in externals) else ""
                L.append(f"\\draw[{sty}] (h{k}.center) -- ({xv[k]+dx:.2f},{ytop}){lab};")
    # external lines from T slots
    for l, (xs_, kind) in slots.items():
        if l in used: continue
        sty = 'pline' if kind == 'p' else 'hline'
        lab = f" node[dlab,above] {{${l}$}}" if labels_ext else ""
        L.append(f"\\draw[{sty}] ({xs_:.2f},0) -- ({xs_:.2f},{ytop}){lab};")
    if label:
        ylab = -0.25 if not dangle else yH - 1.75
        xlab = xmid if not dangle else sum(xv) / len(xv)
        L.append(f"\\node[dlab,below] at ({xlab:.2f},{ylab:.2f}) {{{label}}};")
    body = "\n    ".join(L)
    return f"\\begin{{tikzpicture}}[fcidiagram,scale={scale}]\n    {body}\n\\end{{tikzpicture}}"

E_terms = [
 (('f','i','a'), [[('i','a')]]),
 (('v',('i','a'),('j','b')), [[('i','a'),('j','b')]]),
 (('v',('i','a'),('j','b')), [[('i','a')],[('j','b')]]),
]
T1_terms = [
 (('f','a','i'), []),
 (('f','a','e'), [[('i','e')]]),
 (('f','m','i'), [[('m','a')]]),
 (('v',('m','e'),('a','i')), [[('m','e')]]),
 (('f','m','e'), [[('i','a'),('m','e')]]),
 (('v',('a','e'),('m','f')), [[('i','e'),('m','f')]]),
 (('v',('m','e'),('n','i')), [[('m','e'),('n','a')]]),
 (('f','m','e'), [[('i','e')],[('m','a')]]),
 (('v',('a','e'),('m','f')), [[('i','e')],[('m','f')]]),
 (('v',('m','e'),('n','i')), [[('m','e')],[('n','a')]]),
 (('v',('m','e'),('n','f')), [[('m','e')],[('n','f'),('i','a')]]),
 (('v',('m','e'),('n','f')), [[('i','e')],[('m','a'),('n','f')]]),
 (('v',('m','e'),('n','f')), [[('n','a')],[('m','e'),('i','f')]]),
 (('v',('m','e'),('n','f')), [[('i','e')],[('m','a')],[('n','f')]]),
]
T2_terms = [
 (('v',('a','i'),('b','j')), []),
 (('v',('a','e'),('b','j')), [[('i','e')]]),
 (('v',('a','i'),('m','j')), [[('m','b')]]),
 (('f','b','e'), [[('i','a'),('j','e')]]),
 (('f','m','i'), [[('m','a'),('j','b')]]),
 (('v',('a','e'),('b','f')), [[('i','e'),('j','f')]]),
 (('v',('m','i'),('n','j')), [[('m','a'),('n','b')]]),
 (('v',('m','e'),('b','j')), [[('i','a'),('m','e')]]),
 (('v',('a','e'),('b','f')), [[('i','e')],[('j','f')]]),
 (('v',('m','i'),('n','j')), [[('m','a')],[('n','b')]]),
 (('v',('m','e'),('b','j')), [[('i','e')],[('m','a')]]),
 (('v',('m','e'),('n','f')), [[('i','e'),('j','f')],[('m','a'),('n','b')]]),
 (('v',('m','e'),('n','f')), [[('i','a'),('m','e')],[('n','f'),('j','b')]]),
 (('v',('m','e'),('n','f')), [[('i','a'),('j','e')],[('m','b'),('n','f')]]),
 (('v',('m','e'),('n','f')), [[('m','e'),('i','f')],[('n','a'),('j','b')]]),
 (('f','m','e'), [[('i','e')],[('m','a'),('j','b')]]),
 (('f','m','e'), [[('i','a'),('j','e')],[('m','b')]]),
 (('v',('a','e'),('m','f')), [[('i','e')],[('m','f'),('j','b')]]),
 (('v',('a','e'),('m','f')), [[('i','e'),('j','f')],[('m','b')]]),
 (('v',('b','e'),('m','f')), [[('i','a'),('j','e')],[('m','f')]]),
 (('v',('m','e'),('n','j')), [[('i','a'),('m','e')],[('n','b')]]),
 (('v',('m','e'),('n','j')), [[('i','e')],[('m','a'),('n','b')]]),
 (('v',('m','e'),('n','i')), [[('m','e')],[('n','a'),('j','b')]]),
 (('v',('a','e'),('m','f')), [[('i','e')],[('j','f')],[('m','b')]]),
 (('v',('m','e'),('n','j')), [[('i','e')],[('m','a')],[('n','b')]]),
 (('v',('m','e'),('n','f')), [[('i','e')],[('m','a'),('n','b')],[('j','f')]]),
 (('v',('m','e'),('n','f')), [[('i','e')],[('m','a')],[('n','f'),('j','b')]]),
 (('v',('m','e'),('n','f')), [[('m','a')],[('i','e'),('j','f')],[('n','b')]]),
 (('v',('m','e'),('n','f')), [[('m','e')],[('i','f')],[('n','a'),('j','b')]]),
 (('v',('m','e'),('n','f')), [[('i','a'),('j','e')],[('m','b')],[('n','f')]]),
 (('v',('m','e'),('n','f')), [[('i','e')],[('m','a')],[('j','f')],[('n','b')]]),
]
# H_N elements (dangling lower lines), with labels
F_elems = [(('f','a','i'), r"$f_{ai}$, $+1$"), (('f','a','b'), r"$f_{ab}$, $0$"),
           (('f','i','j'), r"$f_{ij}$, $0$"), (('f','i','a'), r"$f_{ia}$, $-1$")]
V_elems = [(('v',('a','i'),('b','j')), r"$\overline{v}_{abij}$, $+2$"),
           (('v',('a','c'),('b','j')), r"$\overline{v}_{abcj}$, $+1$"),
           (('v',('k','i'),('a','j')), r"$\overline{v}_{kaij}$, $+1$"),
           (('v',('a','c'),('b','d')), r"$\overline{v}_{abcd}$, $0$"),
           (('v',('k','i'),('l','j')), r"$\overline{v}_{klij}$, $0$"),
           (('v',('k','c'),('a','j')), r"$\overline{v}_{kacj}$, $0$"),
           (('v',('k','c'),('a','d')), r"$\overline{v}_{kacd}$, $-1$"),
           (('v',('k','c'),('l','j')), r"$\overline{v}_{klcj}$, $-1$"),
           (('v',('k','c'),('l','d')), r"$\overline{v}_{klcd}$, $-2$")]

def grid(pics, ncol, colsep="1.2em"):
    rows = [pics[k:k+ncol] for k in range(0, len(pics), ncol)]
    out = ["\\begin{tabular}{@{}" + ("c@{\\hspace{%s}}" % colsep) * (ncol - 1) + "c@{}}"]
    for r in rows:
        out.append(" &\n".join(r) + " \\\\[0.9em]")
    out.append("\\end{tabular}")
    return "\n".join(out)



def verify(seed=7):
    """Evaluate the diagram equations, the factored equations and the determinant-space
    similarity transformation for one random case and print the agreement."""
    rng = np.random.default_rng(seed)
    n, N = 8, 4
    o = list(range(N)); v = list(range(N, n)); no, nv = len(o), len(v)
    eps = np.sort(rng.uniform(0, 4, n))
    u = rng.normal(size=(n, n)) * 0.3; u = (u + u.T) / 2
    h = np.diag(eps) + u
    vb = np.zeros((n, n, n, n))
    for p, q, r, s in itertools.product(range(n), repeat=4):
        if p < q and r < s:
            x = rng.normal() * 0.3
            vb[p, q, r, s] = x; vb[q, p, r, s] = -x; vb[p, q, s, r] = -x; vb[q, p, s, r] = x
    vb = (vb + vb.transpose(2, 3, 0, 1)) / 2        # hermitian (real)
    f = h + sum(vb[:, j, :, j] for j in o)
    # ---------------- determinant space
    def popcount(x): return bin(x).count("1")
    def ann(state, p):
        if not (state >> p) & 1: return 0, 0
        return (-1 if popcount(state & ((1 << p) - 1)) & 1 else 1), state ^ (1 << p)
    def cre(state, p):
        if (state >> p) & 1: return 0, 0
        return (-1 if popcount(state & ((1 << p) - 1)) & 1 else 1), state | (1 << p)
    dets = [sum(1 << p for p in c) for c in itertools.combinations(range(n), N)]
    idx = {d: k for k, d in enumerate(dets)}; dim = len(dets)
    def apply_string(ops, state):
        """ops: list of ('c'|'a', p) applied right to left"""
        sign = 1
        for kind, p in reversed(ops):
            s, state = (cre if kind == 'c' else ann)(state, p)
            if s == 0: return 0, 0
            sign *= s
        return sign, state
    H = np.zeros((dim, dim)); T = np.zeros((dim, dim))
    t1 = rng.normal(size=(no, nv)) * 0.1
    t2 = rng.normal(size=(no, no, nv, nv)) * 0.1
    t2 = t2 - t2.transpose(1, 0, 2, 3); t2 = t2 - t2.transpose(0, 1, 3, 2)
    for d in dets:
        k = idx[d]
        for p in range(n):
            for q in range(n):
                if h[p, q] != 0:
                    s, e = apply_string([('c', p), ('a', q)], d)
                    if s: H[idx[e], k] += s * h[p, q]
        for p, q, r, s_ in itertools.product(range(n), repeat=4):
            if vb[p, q, r, s_] != 0:
                s, e = apply_string([('c', p), ('c', q), ('a', s_), ('a', r)], d)
                if s: H[idx[e], k] += 0.25 * s * vb[p, q, r, s_]
        for ii, i in enumerate(o):
            for aa, a in enumerate(v):
                s, e = apply_string([('c', a), ('a', i)], d)
                if s: T[idx[e], k] += s * t1[ii, aa]
        for ii, i in enumerate(o):
            for jj, j in enumerate(o):
                for aa, a in enumerate(v):
                    for bb, b in enumerate(v):
                        s, e = apply_string([('c', a), ('c', b), ('a', j), ('a', i)], d)
                        if s: T[idx[e], k] += 0.25 * s * t2[ii, jj, aa, bb]
    # exp(T) by nilpotent series
    eT = np.eye(dim); term = np.eye(dim)
    for k in range(1, 12):
        term = term @ T / k; eT += term
        if abs(term).max() < 1e-16: break
    emT = np.eye(dim); term = np.eye(dim)
    for k in range(1, 12):
        term = -term @ T / k; emT += term
    Hbar = emT @ H @ eT
    ref = idx[sum(1 << p for p in o)]
    E_ref = H[ref, ref]
    E_det = Hbar[ref, ref] - E_ref
    R1_det = np.zeros((no, nv)); R2_det = np.zeros((no, no, nv, nv))
    for ii, i in enumerate(o):
        for aa, a in enumerate(v):
            s, e = apply_string([('c', a), ('a', i)], dets[ref])
            R1_det[ii, aa] = s * Hbar[idx[e], ref]
    for ii, i in enumerate(o):
        for jj, j in enumerate(o):
            for aa, a in enumerate(v):
                for bb, b in enumerate(v):
                    s, e = apply_string([('c', a), ('c', b), ('a', j), ('a', i)], dets[ref])
                    if s: R2_det[ii, jj, aa, bb] = s * Hbar[idx[e], ref]
    # ---------------- the diagram equations (ccdiag.tex), f_q^p -> f[p,q]; <pq||rs> -> vb[p,q,r,s]
    O, V = np.ix_(o), np.ix_(v)
    foo = f[np.ix_(o, o)]; fvv = f[np.ix_(v, v)]; fov = f[np.ix_(o, v)]; fvo = f[np.ix_(v, o)]
    def W(*blk): return vb[np.ix_(*[o if c == 'o' else v for c in blk])]
    def Pij(x): return x - x.transpose(1, 0, 2, 3)
    def Pab(x): return x - x.transpose(0, 1, 3, 2)
    es = np.einsum
    # energy
    E_diag = es("ia,ia->", fov, t1) + 0.25 * es("ijab,ijab->", W('o','o','v','v'), t2) + 0.5 * es("ijab,ia,jb->", W('o','o','v','v'), t1, t1)
    # T1: 14 terms in the order of the slides
    T1terms = [
     fvo.T,                                                                   # 1 f_i^a = f[a,i]
     es("ae,ie->ia", fvv, t1),                                                # 2 f_e^a t_i^e
     -es("mi,ma->ia", foo, t1),                                               # 3 -f_i^m t_m^a
     es("maei,me->ia", W('o','v','v','o'), t1),                               # 4 <ma||ei> t_m^e
     es("me,imae->ia", fov, t2),                                              # 5 f_e^m t_im^ae
     0.5 * es("amef,imef->ia", W('v','o','v','v'), t2),                       # 6 1/2 <am||ef> t_im^ef
     -0.5 * es("mnei,mnea->ia", W('o','o','v','o'), t2),                      # 7 -1/2 <mn||ei> t_mn^ea
     -es("me,ie,ma->ia", fov, t1, t1),                                        # 8 -f_e^m t_i^e t_m^a
     es("amef,ie,mf->ia", W('v','o','v','v'), t1, t1),                        # 9 <am||ef> t_i^e t_m^f
     -es("mnei,me,na->ia", W('o','o','v','o'), t1, t1),                       # 10 -<mn||ei> t_m^e t_n^a
     es("mnef,me,nifa->ia", W('o','o','v','v'), t1, t2),                      # 11 <mn||ef> t_m^e t_ni^fa
     -0.5 * es("mnef,ie,mnaf->ia", W('o','o','v','v'), t1, t2),               # 12 -1/2 <mn||ef> t_i^e t_mn^af
     -0.5 * es("mnef,na,mief->ia", W('o','o','v','v'), t1, t2),               # 13 -1/2 <mn||ef> t_n^a t_mi^ef
     -es("mnef,ie,ma,nf->ia", W('o','o','v','v'), t1, t1, t1),                # 14 -<mn||ef> t_i^e t_m^a t_n^f
    ]
    R1_diag = sum(T1terms)
    oovv = W('o','o','v','v')
    T2terms = [
     W('v','v','o','o').transpose(2, 3, 0, 1),                                             # 1 <ab||ij>
     Pij(es("abej,ie->ijab", W('v','v','v','o'), t1)),                                    # 2
     -Pab(es("amij,mb->ijab", W('v','o','o','o'), t1)),                                   # 3
     Pab(es("be,ijae->ijab", fvv, t2)),                                                   # 4
     -Pij(es("mi,mjab->ijab", foo, t2)),                                                  # 5
     0.5 * es("abef,ijef->ijab", W('v','v','v','v'), t2),                                 # 6
     0.5 * es("mnij,mnab->ijab", W('o','o','o','o'), t2),                                 # 7
     Pij(Pab(es("mbej,imae->ijab", W('o','v','v','o'), t2))),                             # 8
     0.5 * Pij(es("abef,ie,jf->ijab", W('v','v','v','v'), t1, t1)),                       # 9
     0.5 * Pab(es("mnij,ma,nb->ijab", W('o','o','o','o'), t1, t1)),                       # 10
     -Pij(Pab(es("mbej,ie,ma->ijab", W('o','v','v','o'), t1, t1))),                       # 11
     0.25 * es("mnef,ijef,mnab->ijab", oovv, t2, t2),                                     # 12
     0.5 * Pij(Pab(es("mnef,imae,njfb->ijab", oovv, t2, t2))),                            # 13
     -0.5 * Pab(es("mnef,ijae,mnbf->ijab", oovv, t2, t2)),                                # 14
     -0.5 * Pij(es("mnef,mief,njab->ijab", oovv, t2, t2)),                                # 15
     -Pij(es("me,ie,mjab->ijab", fov, t1, t2)),                                           # 16
     -Pab(es("me,ijae,mb->ijab", fov, t2, t1)),                                           # 17
     Pij(Pab(es("amef,ie,mjfb->ijab", W('v','o','v','v'), t1, t2))),                      # 18
     -0.5 * Pab(es("amef,ijef,mb->ijab", W('v','o','v','v'), t2, t1)),                    # 19
     Pab(es("bmef,ijae,mf->ijab", W('v','o','v','v'), t2, t1)),                           # 20
     -Pij(Pab(es("mnej,imae,nb->ijab", W('o','o','v','o'), t2, t1))),                     # 21
     0.5 * Pij(es("mnej,ie,mnab->ijab", W('o','o','v','o'), t1, t2)),                     # 22
     -Pij(es("mnei,me,njab->ijab", W('o','o','v','o'), t1, t2)),                          # 23
     -0.5 * Pij(Pab(es("amef,ie,jf,mb->ijab", W('v','o','v','v'), t1, t1, t1))),          # 24
     0.5 * Pij(Pab(es("mnej,ie,ma,nb->ijab", W('o','o','v','o'), t1, t1, t1))),           # 25
     0.25 * Pij(es("mnef,ie,mnab,jf->ijab", oovv, t1, t2, t1)),                           # 26
     -Pij(Pab(es("mnef,ie,ma,njfb->ijab", oovv, t1, t1, t2))),                            # 27
     0.25 * Pab(es("mnef,ma,ijef,nb->ijab", oovv, t1, t2, t1)),                           # 28
     -Pij(es("mnef,me,if,njab->ijab", oovv, t1, t1, t2)),                                 # 29
     -Pab(es("mnef,ijae,mb,nf->ijab", oovv, t2, t1, t1)),                                 # 30
     0.25 * Pij(Pab(es("mnef,ie,ma,jf,nb->ijab", oovv, t1, t1, t1, t1))),                 # 31
    ]
    R2_diag = sum(T2terms)
    print("energy   det %.12f  diagrams %.12f  diff %.1e" % (E_det, E_diag, abs(E_det - E_diag)))
    print("T1 eq    max|R1| %.3e   max diff %.1e" % (abs(R1_det).max(), abs(R1_det - R1_diag).max()))
    print("T2 eq    max|R2| %.3e   max diff %.1e" % (abs(R2_det).max(), abs(R2_det - R2_diag).max()))
    # ---------------- the factored (Stanton-Gauss-Watts-Bartlett) equations of the chapter
    tau = t2 + es("ia,jb->ijab", t1, t1) - es("ib,ja->ijab", t1, t1)
    taut = t2 + 0.5 * (es("ia,jb->ijab", t1, t1) - es("ib,ja->ijab", t1, t1))
    fp_vv = fvv - np.diag(np.diag(fvv)); fp_oo = foo - np.diag(np.diag(foo))
    Fae = fp_vv - 0.5 * es("me,ma->ae", fov, t1) + es("mf,mafe->ae", t1, W('o','v','v','v')) - 0.5 * es("mnaf,mnef->ae", taut, oovv)
    Fmi = fp_oo + 0.5 * es("ie,me->mi", t1, fov) + es("ne,mnie->mi", t1, W('o','o','o','v')) + 0.5 * es("inef,mnef->mi", taut, oovv)
    Fme = fov + es("nf,mnef->me", t1, oovv)
    # P(ij) acts on the last two indices of W_mnij, P(ab) on the first two of W_abef
    Wmnij = W('o','o','o','o') + Pab(es("je,mnie->mnij", t1, W('o','o','o','v'))) + 0.25 * es("ijef,mnef->mnij", tau, oovv)
    Wabef = W('v','v','v','v') - Pij(es("mb,amef->abef", t1, W('v','o','v','v'))) + 0.25 * es("mnab,mnef->abef", tau, oovv)
    Wmbej = (W('o','v','v','o') + es("jf,mbef->mbej", t1, W('o','v','v','v')) - es("nb,mnej->mbej", t1, W('o','o','v','o'))
             - es("jnfb,mnef->mbej", 0.5 * t2 + es("jf,nb->jnfb", t1, t1), oovv))
    Do = np.diag(foo); Dv = np.diag(fvv)
    R1_fac = (fov + es("ie,ae->ia", t1, Fae) - es("ma,mi->ia", t1, Fmi) + es("imae,me->ia", t2, Fme)
              - es("nf,naif->ia", t1, W('o','v','o','v')) - 0.5 * es("imef,maef->ia", t2, W('o','v','v','v'))
              - 0.5 * es("mnae,nmei->ia", t2, W('o','o','v','o'))) - (Do[:, None] - Dv[None, :]) * t1
    D2 = Do[:, None, None, None] + Do[None, :, None, None] - Dv[None, None, :, None] - Dv[None, None, None, :]
    R2_fac = (oovv + Pab(es("ijae,be->ijab", t2, Fae - 0.5 * es("mb,me->be", t1, Fme)))
              - Pij(es("imab,mj->ijab", t2, Fmi + 0.5 * es("je,me->mj", t1, Fme)))
              + 0.5 * es("mnab,mnij->ijab", tau, Wmnij) + 0.5 * es("ijef,abef->ijab", tau, Wabef)
              + Pij(Pab(es("imae,mbej->ijab", t2, Wmbej) - es("ie,ma,mbej->ijab", t1, t1, W('o','v','v','o'))))
              + Pij(es("ie,abej->ijab", t1, W('v','v','v','o'))) - Pab(es("ma,mbij->ijab", t1, W('o','v','o','o')))) - D2 * t2
    print("factored  max|R1 - R1_fac| %.1e   max|R2 - R2_fac| %.1e" % (abs(R1_diag - R1_fac).max(), abs(R2_diag - R2_fac).max()))
    return dict(E=(E_det, E_diag), R1=(R1_det, R1_diag, R1_fac), R2=(R2_det, R2_diag, R2_fac))


def write_tikz(outdir):
    os.makedirs(outdir, exist_ok=True)
    parts = {}
    parts['E'] = grid([term_pictures(H, A, externals=set(), label=f"({k})") for k, (H, A) in enumerate(E_terms, 1)], 3, "3em")
    parts['T1'] = grid([term_pictures(H, A, label=f"({k})") for k, (H, A) in enumerate(T1_terms, 1)], 4)
    parts['T2a'] = grid([term_pictures(H, A, label=f"({k})") for k, (H, A) in enumerate(T2_terms[:16], 1)], 4)
    parts['T2b'] = grid([term_pictures(H, A, label=f"({k})") for k, (H, A) in enumerate(T2_terms[16:], 17)], 4)
    parts['F'] = grid([term_pictures(H, [], externals=set("abcdijkl"), dangle=True, label=lab, labels_ext=False, scale=0.8) for H, lab in F_elems], 4, "3em")
    parts['V'] = grid([term_pictures(H, [], externals=set("abcdijkl"), dangle=True, label=lab, labels_ext=False, scale=0.8) for H, lab in V_elems], 3, "3em")
    def barpic(nv, lab):
        xs = [0.9 * k for k in range(nv)]
        L = [f"\\draw[amp] ({xs[0]-0.4:.2f},0) -- ({xs[-1]+0.4:.2f},0);"]
        letters = [('i','a'),('j','b'),('k','c')]
        for xv, (hl, pl) in zip(xs, letters[:nv]):
            L.append(f"\\draw[hline] ({xv-0.22:.2f},0) -- ({xv-0.22:.2f},1.3) node[dlab,above] {{${hl}$}};")
            L.append(f"\\draw[pline] ({xv+0.22:.2f},0) -- ({xv+0.22:.2f},1.3) node[dlab,above] {{${pl}$}};")
        L.append(f"\\node[dlab,below] at ({(xs[0]+xs[-1])/2:.2f},-0.25) {{{lab}}};")
        return "\\begin{tikzpicture}[fcidiagram,scale=0.8]\n    " + "\n    ".join(L) + "\n\\end{tikzpicture}"
    parts['T'] = grid([barpic(1, r"$t_i^a$, $+1$"), barpic(2, r"$t_{ij}^{ab}$, $+2$"), barpic(3, r"$t_{ijk}^{abc}$, $+3$")], 3, "3em")
    for k, v in parts.items():
        open(os.path.join(outdir, f"tikz_{k}.tex"), "w").write(v + "\n")
    print("TikZ written to", outdir)


if __name__ == "__main__":
    verify()
    if "--tikz" in sys.argv:
        write_tikz(sys.argv[sys.argv.index("--tikz") + 1] if len(sys.argv) > sys.argv.index("--tikz") + 1 else "tikz_out")
