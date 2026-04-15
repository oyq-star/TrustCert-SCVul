# TrustCert-SCVul

## Overview

TrustCert-SCVul is an interpretable selective triage pipeline for smart contract vulnerabilities. It combines structural Solidity features with real Slither analyzer signals through per-vulnerability Explainable Boosting Machines (EBMs), then applies split conformal prediction to abstain on uncertain inputs. Each accepted prediction is documented in a reproducible evidence record with Merkle-batched hashing.

The entire pipeline runs on a **single CPU** in under **19 minutes**.

## Key Results

| Task | Model | F1 (grouped 5-fold CV) | Accept Rate | Accepted Precision |
|------|-------|------------------------|-------------|-------------------|
| Reentrancy | TrustCert-EBM | 0.903 +/- 0.092 | 34.2% | 0.40 +/- 0.49 |
| Reentrancy | Random Forest | 0.924 +/- 0.053 | 63.3% | 1.00 +/- 0.00 |
| Reentrancy | L1-Logistic | 0.904 +/- 0.069 | 13.2% | 0.20 +/- 0.40 |
| DoS | TrustCert-EBM | 0.880 +/- 0.070 | 29.6% | 0.75 +/- 0.39 |
| DoS | Random Forest | 0.901 +/- 0.078 | 71.4% | 0.98 +/- 0.04 |
| DoS | L1-Logistic | 0.867 +/- 0.107 | 41.9% | 0.73 +/- 0.39 |

On a single held-out split, conformal prediction raises accepted-set precision to **1.000** for all four model families (L1-Logistic, Random Forest, LightGBM, EBM) at acceptance rates between 82% and 100%.

## Project Structure

```
TrustCert-SCVul/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── trustcert_scvul/
│       ├── __init__.py
│       ├── analyzers/           # Slither integration
│       │   └── slither_runner.py
│       ├── calibration/         # Split conformal prediction
│       │   └── conformal.py
│       ├── certificates/        # Evidence records & Merkle tree
│       │   └── evidence.py
│       ├── data/                # Dataset loading & preprocessing
│       │   ├── ingest.py        # SmartBugs-Curated loader
│       │   └── wild_loader.py   # SmartBugs-Wild loader
│       ├── evaluation/
│       ├── experiments/
│       │   ├── run_all.py       # Main experiment runner
│       │   ├── round2_fixes.py  # Grouped CV + abstention baselines
│       │   └── round3_fixes.py  # Per-fold metrics + case studies
│       ├── features/
│       │   ├── structural.py    # 40 structural Solidity features
│       │   └── analyzer.py      # Analyzer & consensus features
│       ├── models/
│       │   └── train.py         # EBM, RF, LR, LightGBM training
│       └── utils/
└── scripts/
    ├── make_figures.py          # Publication-quality figures (matplotlib)
    └── verify_splits.py         # Split leakage verification
```

## Installation

```bash
conda create -n trustcert python=3.10 -y
conda activate trustcert

pip install -r requirements.txt
```

### External Dependencies

- **Slither**: Install via `pip install slither-analyzer`. Requires `solc` (Solidity compiler) — install multiple versions via [solc-select](https://github.com/crytic/solc-select).
- **SmartBugs-Curated**: Clone from https://github.com/smartbugs/smartbugs-curated into `data/raw/smartbugs_curated/`.
- **SmartBugs-Wild**: Clone from https://github.com/smartbugs/smartbugs-wild into `data/raw/smartbugs_wild/`.

## Usage

### Run All Experiments

```bash
cd src
python -m trustcert_scvul.experiments.run_all
```

This runs:
1. Data ingestion and feature extraction (structural + Slither)
2. Model training (EBM, RF, LR, LightGBM, rule baselines)
3. Conformal calibration and selective prediction evaluation
4. Cross-dataset evaluation on SmartBugs-Wild
5. Evidence record and certificate benchmarking

Results are saved to `artifacts/`.

### Run Grouped CV + Case Studies

```bash
python -m trustcert_scvul.experiments.round3_fixes
```

Produces:
- `artifacts/grouped_cv_results.csv` — 5-fold CV metrics with per-fold acceptance rates
- `artifacts/abstention_baselines.csv` — Multi-alpha conformal comparison across 4 models
- `artifacts/case_studies.json` — EBM local explanations for TP and abstained contracts


### Key Components

1. **Feature Extraction**: 40 regex-based structural features (LOC, call patterns, loop density, etc.) + real Slither detector outputs + inter-analyzer consensus signals.

2. **Explainable Boosting Machine (EBM)**: Generalized additive model with learned univariate shape functions and up to 5 pairwise interactions. Provides exact local additive decomposition without post-hoc explainers.

3. **Split Conformal Prediction**: Distribution-free prediction sets with marginal coverage >= 1 - alpha. Singleton sets are accepted; non-singleton sets trigger abstention.

4. **Evidence Records**: JSON documents with contract hash, top-k feature contributions, supporting Slither findings, and Merkle-batched integrity hash.

## Datasets

| Dataset | Contracts | Source | Labels |
|---------|-----------|--------|--------|
| SmartBugs-Curated | 109 | Ground-truth | Expert-annotated |
| SmartBugs-Wild | 47,398 (200 sampled) | Ethereum mainnet | Slither silver labels |


