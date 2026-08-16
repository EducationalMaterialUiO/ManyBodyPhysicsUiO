# Chapter 10 -- Coupled cluster theory

Companion notebook: `../../LectureNotes/coupledcluster.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter10/`

| Program | What it does | Runtime |
|---|---|---|
| `coupledcluster.py` | general spin-orbital CCSD with intermediates, CCD, a validating full CI in the same basis, the order-by-order expansion, and `UnitaryCC} (UCCD and UCCSD, exact and Trotterised) | a few seconds |
| `ucc.py` | an earlier standalone unitary coupled cluster implementation, kept for reference | a few seconds |
| `CCD_PairingModel.py` | the original CCD solver for the pairing model | a few seconds |
| `NeutronMatterCCD_Ladders.py` | coupled cluster doubles with ladder diagrams for infinite neutron matter | minutes |

Run any of them as a script:

```bash
cd BookPrograms/chapter10
python3 <program>.py
```
