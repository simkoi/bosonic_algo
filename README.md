# bosonic_algo

Reproducible **Shin–Metiu molecular dynamics on bosonic quantum hardware**, with
full-potential sparse Fourier fitting, classical reference propagation, and
cavity-free or cavity-coupled TGIFS propagation.

## Quick start

```bash
python -m pip install -e '.[dev]'
```

For quantum propagation, additionally install **compatible** versions of
`qiskit` and `bosonic_qiskit` in your environment. The classical workflow does
not import them. The quantum engine uses Bosonic-Qiskit-specific CV gate APIs;
validate the gate conventions with your installed version before production.

Start with the three small notebooks in order:

1. `notebooks/01_fourier_workflow.ipynb` — generate candidate Fourier fits,
   inspect fit metrics, **explicitly select** a few for classical validation.
   Coefficients and full fitting metadata are saved under `data/fits/fit_*/`.
2. `notebooks/02_dynamics.ipynb` — set the ID of a **saved** Fourier fit,
   then run or load exact classical, fitted classical, and TGIFS simulations.
   `configs/dynamics.yaml` contains the shared parameters; toggle the cavity
   there. No Fourier fitting occurs in this notebook.
3. `notebooks/03_results_analysis.ipynb` — search metadata and load recorded
   observables and snapshots **without propagating anything**.

Run notebooks from the repository root or the `notebooks/` directory. Each
notebook automatically finds `src/bosonic_algo` and adds the local checkout to
`sys.path` **before importing the package**, so an editable install is not
required for package discovery. Dependencies must still be installed in the
notebook's selected Python kernel.

### If a notebook cannot import the package

First verify where its kernel is running:

```python
import sys
from pathlib import Path
print(sys.executable)
print(Path.cwd())
```

From a terminal in the repository root, run `python -m pip install -e .`
using the Python environment selected as the Jupyter kernel. Alternatively,
run `%pip install -e ..` **inside the notebook** if its working directory is
`bosonic_algo/notebooks/`, or `%pip install -e .` if it is the repository
root. The `%pip` command installs into the active notebook kernel. Restart
the kernel after installation if the import error persists.

If you launch Jupyter from outside the checkout, change its working directory
to the repository root or `notebooks/`; otherwise the automatic checkout
discovery cannot find `configs/` or `data/`.

## Python API

```python
from bosonic_algo import generate_fits, run_or_load_classical, run_or_load_tgifs, ResultStore

rows = generate_fits()  # FIT only: no dynamics
fit_id = next(row['fit_id'] for row in rows if row['status'] == 'saved')
exact, _, exact_cache_hit = run_or_load_classical(potential='exact')
fitted, _, fit_cache_hit = run_or_load_classical(potential='fitted', fit_id=fit_id)
quantum, metadata, quantum_cache_hit = run_or_load_tgifs(fit_id)

store = ResultStore('results')
for run in store.find_results(fit_id=fit_id):
    print(run['run_id'],run['method'])
    arrays, metadata = store.load(run['method'],run['run_id'])
```

The default Fock basis and numerical grid are computationally expensive. Use
small basis/grid settings when first testing the installation. The Fourier
workflow can produce *zero* retained fits if `raw_v_rmse_max` is too strict;
inspect the rejected summaries before changing it.

## Reading Fourier screening results

`notebooks/01_fourier_workflow.ipynb` displays **two distinct tables**:
- `saved_df` (from `generate_fits()`): accepted, reduced, and saved Fourier fits only. Every row has complete reduced-fit metrics and a usable `fit_id`. The same rows are written to `data/fits/index.csv`.
- `raw_df` (from `data/fits/raw_fit_summary.csv`): every raw Fourier candidate, including `passed` and `rejected` entries, with raw potential and force errors.

Rejected fits are intentionally **not reduced or saved**. If `saved_df` is empty, inspect the raw table's `v_rmse` values relative to `raw_v_rmse_max` in `configs/fourier.yaml`. Adjust the threshold only if the raw approximations are scientifically acceptable, or improve the Fourier configuration. No classical dynamics are run until `selected_for_dynamics` explicitly names saved fit IDs.

## Results and reproducibility

A **fit** has a content-derived ID (`fit_<checksum>`), immutable coefficient
file (`coefficients.npz`), and `metadata.json` containing the exact fitting
parameters, model, grid, and diagnostics. The dynamics routines **only load**
fits and verify that the saved model matches the requested physical model.

Every dynamics run has a content-derived ID incorporating its physical and
numerical configuration, initial-state coefficient hash, fit checksum (when
applicable), and a fingerprint of computational source files. Results are in:

```text
results/classical/<run_id>/metadata.json
results/classical/<run_id>/observables.npz
results/classical/<run_id>/snapshots.npz
results/quantum/<run_id>/metadata.json
results/quantum/<run_id>/observables.npz
results/quantum/<run_id>/snapshots.npz
```

`run_or_load_*()` verifies complete metadata and numerical checksums before
reusing results. Use `force=True` to explicitly recompute. The snapshot policy
is part of the cache key, so changing it cannot silently return the wrong
wavefunctions. `ResultStore.find_results(...)` searches only small metadata;
`list_results()` scans directories and does not require a central database.

Simulation arrays are not committed to Git; **selected fit files may be
committed intentionally** for shared reproducible experiments. Record the
model and dependency versions with results. Never assume results from an older
quantum gate-library version have identical gate semantics.

## Scientific assumptions and checks

- Fourier modes use `(R, r)` wavevector order. `omega`, `Omega` are Fock
  coordinate scales, **not** added harmonic potential terms.
- Classical dynamics are propagated using the full Fock-projected Hamiltonian
  with Krylov exponentiation, for the exact or fitted potential, including
  photon, coupling, and dipole self-energy terms when the cavity is enabled.
- TGIFS potential gates use the imported Fourier coefficients; kinetic motion
  uses rotation–squeeze–rotation or the optional finite-Fock matrix version.
- The cavity code retains the SUM-gate ordering workaround from the prior
  notebook; numerical validation is required on each Bosonic-Qiskit release.
- Initial state is the second BO electronic state at `R=-4` and a Gaussian
  nuclear packet (configurable under `initial`), with cavity vacuum.
- Ancilla-|0> TGIFS observables are conditioned on successful postselection;
  the success probability `p0` is saved separately. Quantum cutoff effects,
  postselection probability, Strang step size, and Fourier error all require
  convergence checks.

## Validation

```bash
pytest -q
```

The tests cover fitting, round-trip persistence, metadata-based cache reuse
and invalidation, and small classical evolution. The original notebooks remain at the repository root as reference
implementations while the consolidated code is validated. Full TGIFS execution requires a compatible `bosonic_qiskit`
installation and is not exercised by the dependency-light CI tests.
