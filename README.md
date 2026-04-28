# Is Silence Information? Evidence from Missingness in Asset Pricing

<p align="center">
  <b>Codebase for an empirical asset pricing study of residual missingness, MAR falsification, and the pricing of silence.</b><br>
  <i>Current repository status: research implementation in progress</i>
</p>

---

## Overview

This repository contains the research code for the project **"Is Silence Information? Evidence from Missingness in Asset Pricing"**.

The project studies whether missing predictor values in asset pricing panels are:

- inconsistent with a Missing at Random (MAR) benchmark
- economically meaningful for return prediction
- concentrated in specific latent signal regions
- priced in the cross-section of expected returns

This repository is an active research codebase.

- A paper draft or public paper link is **not** available yet.
- The working title matches the current research proposal: **"Is Silence Information? Evidence from Missingness in Asset Pricing"**.
- For research motivation and identification logic, see [Research_Proposal_MNAR.pdf](</D:/source/repos/AP_MNAR/Research_Proposal_MNAR.pdf>).

## Current study design

The implementation follows a staged empirical design:

1. `Phase 0`: build a canonical monthly panel for the analyst pilot signals
2. `Phase 1`: classify residual missingness after eligibility filtering
3. `Phase 2`: test and reject the MAR benchmark
4. `Phase 3`: run counterfactual MAR-style imputations and signal-sorted sensitivity analysis
5. `Phase 4`: test whether residual missingness is priced in returns

The current pilot signal set is:

- `AnalystRevision`
- `REV6`
- `ForecastDispersion`
- `FEPS`

## Main hypotheses

- `H1`: MAR implications are violated in the analyst panel
- `H2`: MAR-style counterfactual replacement materially changes predictive and portfolio outcomes
- `H3`: those counterfactual gains are concentrated in particular regions of the latent signal distribution
- `H4`: residual missingness is priced in the cross-section of returns
- `H5`: missingness patterns align with broader information-asymmetry proxies

At the moment, the repository has implemented the full pipeline through `Phase 4` for the core pricing block. The external validation layer under `H5` is still a next-step extension.

## Repository structure

```text
AP_MNAR/
+-- configs/                       # Fixed research configs (X_obs, missingness rules)
+-- data/                          # Raw and metadata tables used by the project
+-- outputs/                       # Generated interim panels, tables, and figures
+-- scripts/                       # CLI entry points for each phase
+-- src/ap_mnar/
|   +-- data/                      # Loading, merging, metadata, target construction
|   +-- missingness/               # Eligibility and residual missingness logic
|   +-- models/                    # MAR benchmark and counterfactual helpers
|   +-- pricing/                   # Phase 4 pricing-panel and Fama-MacBeth code
|   +-- reporting/                 # Tables and figures
|   +-- stats/                     # J-test, diagnostics, prediction metrics
|   +-- experiments/               # Phase 0-4 orchestrators
+-- tests/                         # Synthetic end-to-end tests by phase
+-- IMPLEMENTATION_PLAN_MNAR.md    # Project-wide implementation plan
+-- PHASE0_IMPLEMENTATION.md
+-- PHASE1_IMPLEMENTATION.md
+-- PHASE2_IMPLEMENTATION.md
+-- PHASE3_IMPLEMENTATION.md
+-- PHASE4_IMPLEMENTATION.md
+-- DATA_STRUCTURE_MNAR.md         # Data context notes for the analyst pilot
```

## Data

This repository is built around monthly asset pricing panel data and signal metadata already stored in the project workspace.

Important notes:

- the current implementation uses internal/local research data files
- some underlying source data originate from datasets that typically require WRDS or related institutional access
- this repository is therefore best understood as a research implementation environment, not a fully public replication package

The most important working inputs for the current pilot are:

- `data/raw/firm_characs.csv`
- `data/raw/open_source_asset_pricing.csv`
- `data/info/SignalDoc.csv`
- `configs/x_obs.yaml`

## Environment

The codebase is written in Python and currently relies on the standard scientific stack used throughout the implemented phases.

Typical dependencies include:

- `numpy`
- `pandas`
- `statsmodels`
- `scipy`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `pyyaml`
- `tqdm`
- `pytest`

## Running the pipeline

### Phase 0: Canonical panel build

```powershell
python scripts/run_step0_audit.py
```

### Phase 1: Residual missingness classification

```powershell
python scripts/run_phase1_missingness.py
```

### Phase 2: MAR falsification

```powershell
python scripts/run_phase2_mar_test.py
```

### Phase 3: Counterfactual characterization

```powershell
python scripts/run_phase3_counterfactual.py
```

### Phase 4: Pricing of missingness

```powershell
python scripts/run_phase4_pricing.py
```

Recommended first pilot for Phase 4:

```powershell
python scripts/run_phase4_pricing.py --signals ForecastDispersion
```

## Output convention

Generated artifacts are written under `outputs/`:

- `outputs/interim/`
  - panel-level parquet files used across phases
- `outputs/tables/`
  - regression outputs, diagnostics, and summary tables
- `outputs/figures/`
  - signal-level and phase-level visual summaries

Examples:

- `outputs/interim/panel_with_missingness.parquet`
- `outputs/tables/mar_jtest_results.csv`
- `outputs/tables/counterfactual_signal_sorted_results.csv`
- `outputs/tables/missingness_fama_macbeth_results.csv`

## Testing

Each implemented phase has a synthetic end-to-end test.

Run the current full test suite with:

```powershell
pytest -q -p no:cacheprovider tests/test_phase0.py tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py
```

## Documentation

If you are trying to understand the current research state, the most useful files are:

- [IMPLEMENTATION_PLAN_MNAR.md](</D:/source/repos/AP_MNAR/IMPLEMENTATION_PLAN_MNAR.md>)
- [DATA_STRUCTURE_MNAR.md](</D:/source/repos/AP_MNAR/DATA_STRUCTURE_MNAR.md>)
- [PHASE0_IMPLEMENTATION.md](</D:/source/repos/AP_MNAR/PHASE0_IMPLEMENTATION.md>)
- [PHASE1_IMPLEMENTATION.md](</D:/source/repos/AP_MNAR/PHASE1_IMPLEMENTATION.md>)
- [PHASE2_IMPLEMENTATION.md](</D:/source/repos/AP_MNAR/PHASE2_IMPLEMENTATION.md>)
- [PHASE3_IMPLEMENTATION.md](</D:/source/repos/AP_MNAR/PHASE3_IMPLEMENTATION.md>)
- [PHASE4_IMPLEMENTATION.md](</D:/source/repos/AP_MNAR/PHASE4_IMPLEMENTATION.md>)

## Current status

What is implemented now:

- analyst-first pilot panel construction
- eligibility-aware residual missingness classification
- MAR falsification diagnostics
- deterministic and stochastic counterfactual analysis
- signal-sorted counterfactual result decomposition
- Phase 4 pricing of missingness with Fama-MacBeth and coverage decomposition

What remains for later expansion:

- state-dependent pricing robustness
- multi-signal joint pricing tests
- H5 external validation using market-based and event-based signals
- richer paper-ready robustness and final table polishing

## Citation

There is no public paper link for this repository yet.

If you need to reference the project internally, use the current working title:

> *Is Silence Information? Evidence from Missingness in Asset Pricing*
