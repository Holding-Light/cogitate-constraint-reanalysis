"""
Configuration for COGITATE Constraint-Architecture Reanalysis
=============================================================
Paper: "Constraint-Dependent Perceptual Resolution"
Author: Jeremy C. Jones — HoldingLight LLC

All parameters, paths, and constants in one place.
Edit DATA_ROOT to point to your local COGITATE download.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────
# PATHS — Edit DATA_ROOT to match your local setup
# ─────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/jeremy/Downloads/mnt/beegfs/workspace/2023-0385-Cogitatedatarelease/CURATE/COG_ECOG_EXP1_BIDS")
BIDS_ROOT = DATA_ROOT
OUTPUT_DIR = Path("~/cogitate_reanalysis/output").expanduser()
FIGURES_DIR = Path("~/cogitate_reanalysis/figures").expanduser()

# Create output dirs if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
# HIGH-GAMMA EXTRACTION
# ─────────────────────────────────────────────────────────
HG_FREQ_LOW = 70.0       # Hz — lower bound of high-gamma band
HG_FREQ_HIGH = 150.0     # Hz — upper bound of high-gamma band
HG_SMOOTH_SIGMA = 0.050  # seconds — Gaussian smoothing kernel σ
LOG_TRANSFORM = True      # log-transform high-gamma amplitude

# ─────────────────────────────────────────────────────────
# EPOCHING
# ─────────────────────────────────────────────────────────
EPOCH_TMIN = -0.5         # seconds before stimulus onset
EPOCH_TMAX = 2.0          # seconds after stimulus onset
BASELINE = (-0.3, 0.0)   # baseline correction window (seconds)

# ─────────────────────────────────────────────────────────
# ELECTRODE SELECTION — Onset Responsiveness
# ─────────────────────────────────────────────────────────
ONSET_PRE_WINDOW = (-0.3, 0.0)    # pre-stimulus window
ONSET_POST_WINDOW = (0.05, 0.35)  # post-stimulus window
ONSET_P_THRESHOLD = 0.01          # p-value threshold (FDR-corrected)

# ─────────────────────────────────────────────────────────
# ELECTRODE SELECTION — Category Selectivity (Decoding)
# ─────────────────────────────────────────────────────────
DECODING_WINDOW = (0.05, 0.5)     # time window for feature extraction
DECODING_N_FOLDS = 5              # cross-validation folds
DECODING_N_PERMUTATIONS = 1000    # permutation test iterations
DECODING_P_THRESHOLD = 0.05       # significance threshold

# ─────────────────────────────────────────────────────────
# PREDICTION A — Onset Latency
# ─────────────────────────────────────────────────────────
ONSET_LATENCY_WINDOW = (0.05, 0.5)    # search window for onset
ONSET_LATENCY_THRESHOLD = 0.5          # fraction of peak (50%)

# ─────────────────────────────────────────────────────────
# PREDICTION B — Hysteresis
# ─────────────────────────────────────────────────────────
HYSTERESIS_EARLY_TRIALS = 5       # number of early post-transition trials
HYSTERESIS_LATE_TRIALS_START = 10 # late trials start index
HYSTERESIS_N_PERMUTATIONS = 10000 # permutation test iterations
HYSTERESIS_AMPLITUDE_WINDOW = (0.05, 0.35)  # response window for amplitude

# ─────────────────────────────────────────────────────────
# PREDICTION C — Duration-Tracking
# ─────────────────────────────────────────────────────────
DURATION_TRACKING_WINDOW = (0.5, 1.5)  # sustained response window
DURATIONS_MS = [500, 1000, 1500]       # stimulus durations in ms

# ─────────────────────────────────────────────────────────
# TASK RELEVANCE CONDITIONS (matching COGITATE event codes)
# ─────────────────────────────────────────────────────────
TASK_RELEVANCE = {
    "target": "Relevant target",
    "nontarget": "Relevant non-target",
    "irrelevant": "Irrelevant",
}

# ─────────────────────────────────────────────────────────
# STIMULUS CATEGORIES
# ─────────────────────────────────────────────────────────
CATEGORIES = ["face", "object", "letter", "false"]

# Category groupings (for block-level target assignment)
PICTORIAL = ["face", "object"]
SYMBOLIC = ["letter", "false"]

# ─────────────────────────────────────────────────────────
# REGIONS OF INTEREST (Desikan atlas labels)
# ─────────────────────────────────────────────────────────
ROI_VENTRAL_TEMPORAL = [
    "fusiform",
    "inferiortemporal",
    "parahippocampal",
]

ROI_LATERAL_OCCIPITAL = [
    "lateraloccipital",
    "lingual",
    "cuneus",
    "pericalcarine",
]

ROI_PREFRONTAL = [
    "parsopercularis",
    "parstriangularis",
    "parsorbitalis",
    "rostralmiddlefrontal",
    "caudalmiddlefrontal",
    "lateralorbitofrontal",
]

ROIS = {
    "ventral_temporal": ROI_VENTRAL_TEMPORAL,
    "lateral_occipital": ROI_LATERAL_OCCIPITAL,
    "prefrontal": ROI_PREFRONTAL,
}

# ─────────────────────────────────────────────────────────
# STATISTICAL PARAMETERS
# ─────────────────────────────────────────────────────────
FDR_ALPHA = 0.05           # FDR correction threshold
RANDOM_SEED = 42           # reproducibility seed

# ─────────────────────────────────────────────────────────
# FIGURE STYLE
# ─────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"
COLOR_RELEVANT = "#2C5F8A"    # deep blue — relevant non-target
COLOR_IRRELEVANT = "#B8860B"  # dark goldenrod — irrelevant
COLOR_HYSTERESIS = "#8B2252"  # maroon — hysteresis decay
