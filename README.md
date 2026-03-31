# COGITATE Constraint-Architecture Reanalysis

## Constraint-Dependent Perceptual Resolution
### A Structural-Architecture Reanalysis of the COGITATE Consciousness Dataset

**Author:** Jeremy C. Jones (ORCID 0009-0007-2515-3774) — HoldingLight LLC  
**Contact:** contact@universalcollapse.com  
**License:** CC BY 4.0 (text/data), MIT (code)

---

## Overview

This repository contains the analysis pipeline for testing whether a 
constraint-architecture model of perceptual resolution (Jones, 2026) 
predicts temporal dynamics in the COGITATE open iEEG dataset that 
neither Integrated Information Theory nor Global Neuronal Workspace 
Theory captured.

Three pre-specified predictions are tested:

- **Prediction A** — Constraint load delays resolution (onset latency)
- **Prediction B** — Accumulated constraint produces hysteresis across blocks
- **Prediction C** — Duration-tracking is constraint-modulated

## Data

This pipeline uses the COGITATE iEEG dataset:

> Seedat et al. (2025). Open multi-center intracranial electroencephalography 
> dataset with task probing conscious visual perception. *Scientific Data*, 12, 854.

**Access:** Register at https://www.arc-cogitate.com/data-release

Download the iEEG data bundle and extract to a local directory.
Update `DATA_ROOT` in `scripts/config.py` to point to your download.

## Requirements

```bash
pip install mne mne-bids scikit-learn statsmodels scipy numpy pandas matplotlib
```

Tested with:
- Python 3.10+
- MNE 1.11.0
- MNE-BIDS 0.18.0
- scikit-learn 1.8.0
- statsmodels 0.14.6

## Pipeline

Run the pipeline in order:

### Step 1: Preprocessing
```bash
cd scripts/
python preprocessing.py                    # all subjects
python preprocessing.py --subject CE101    # single subject
```

Loads BIDS data, excludes bad channels, applies common average reference,
extracts high-gamma (70-150 Hz) analytic amplitude, creates epochs.

### Step 2: Electrode Selection
```bash
python electrode_selection.py              # all subjects
python electrode_selection.py --subject CE101
```

Identifies onset-responsive electrodes (paired t-test, FDR-corrected),
then tests category selectivity via SVM decoding.

### Step 3: Run Predictions
```bash
python run_predictions.py                  # all predictions, all subjects
python run_predictions.py --prediction A   # single prediction
python run_predictions.py --group-only     # group analysis only
```

Tests all three predictions per subject, then aggregates to group level.
Generates summary figures.

## Directory Structure

```
cogitate_reanalysis/
├── scripts/
│   ├── config.py                # All parameters and paths
│   ├── utils.py                 # Shared utility functions
│   ├── preprocessing.py         # Step 1: Data preprocessing
│   ├── electrode_selection.py   # Step 2: Electrode selection
│   └── run_predictions.py       # Step 3: Prediction tests
├── output/                      # Preprocessed data and results
├── figures/                     # Generated figures
├── notebooks/                   # Jupyter notebooks (optional)
└── README.md                    # This file
```

## Output Files

Per subject:
- `sub-{ID}_hg_epochs-epo.fif` — Preprocessed high-gamma epochs
- `sub-{ID}_channel_info.json` — Channel metadata
- `sub-{ID}_electrode_selection.json` — Selected electrodes
- `sub-{ID}_prediction_{a,b,c}.csv` — Per-electrode results

Group level:
- `group_prediction_{a,b,c}.csv` — Aggregated results
- `figures/prediction_{a,c}_summary.pdf` — Summary figures

## Citation

If you use this pipeline, please cite:

- Jones, J. C. (2026). Constraint-Dependent Perceptual Resolution: 
  A Structural-Architecture Reanalysis of the COGITATE Consciousness Dataset.
  *PsyArXiv*. HoldingLight LLC.

- Jones, J. C. (2026). The Self the Ego Did Not Build: What Decides 
  Before You Decide. *PhilArchive*. HoldingLight LLC.

- Cogitate Consortium et al. (2025). Adversarial testing of global 
  neuronal workspace and integrated information theories of consciousness. 
  *Nature*, 642(8066), 133–142.

## AI Disclosure

Portions of this analysis pipeline were developed with assistance from 
Claude (Anthropic), used as a structural reasoning and production tool. 
All scientific claims, interpretations, and final decisions are the author's own.

---

*HoldingLight LLC — universalcollapse.com*
