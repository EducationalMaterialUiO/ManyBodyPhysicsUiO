# Green's functions -- the chapter after coupled cluster theory

Companion notebook: `../../LectureNotes/greensfunctions.ipynb`

Figures generated from that notebook: `../../BookFigures/chaptergreenfunction/`

| Program | What it does | Runtime |
|---|---|---|
| `greensfunction.py` | the exact one-body propagator from full CI in the N and N±1 spaces (`LehmannPropagator`: Lehmann representation, spectral functions, occupation numbers, density matrix, the Migdal-Galitskii-Koltun sum rule); the [12] and [13] pair-space forms of a four-index vertex (`PairSpaces`); the free two-time propagators and the parquet equations at fixed starting energies (`parquet_vertex`); the self energy in pole form with a dressed vertex (`self_energy_poles`) and with the full energy dependence of the ladder and ring channels through the RPA phonons (`phonon_self_energy_poles`); the Dyson equation as an eigenvalue problem in the extended space (`dyson`), the sum rule (`koltun_energy`) and the driver with the static and quasiparticle self-consistencies (`green_function_energy`); the Lipkin model in a spin-orbital basis; checks of the two channels against the RPA and the pp-RPA; `compare_methods` for HF, MP2, MP3, GF2, parquet, RPA-dressed, CCD, CCSD and FCI side by side | about 20 seconds |

The program imports `coupledcluster.py` (chapter 10) for the pairing models,
Hartree-Fock, CCSD and the validating full CI, `mbpt.py` (chapter 9) for the
Møller-Plesset series and `models.py` (chapter 4) for the exact Lipkin
energies, so the whole of `BookPrograms/` must be on the path:

```bash
cd BookPrograms
PYTHONPATH=$(ls -d chapter* | tr '\n' ':') python3 chaptergreenfunction/greensfunction.py
```

It runs on `numpy` alone.
