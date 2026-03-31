"""
Prediction Tests for COGITATE Constraint-Architecture Reanalysis
================================================================
Implements all three pre-specified predictions:
  A — Constraint load delays resolution (onset latency)
  B — Accumulated constraint produces hysteresis
  C — Duration-tracking is constraint-modulated

Usage:
    python run_predictions.py                        # all predictions, all subjects
    python run_predictions.py --prediction A         # single prediction
    python run_predictions.py --subject CE101        # single subject
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR, FIGURES_DIR,
    ONSET_LATENCY_WINDOW, ONSET_LATENCY_THRESHOLD,
    HYSTERESIS_EARLY_TRIALS, HYSTERESIS_LATE_TRIALS_START,
    HYSTERESIS_N_PERMUTATIONS, HYSTERESIS_AMPLITUDE_WINDOW,
    DURATION_TRACKING_WINDOW, DURATIONS_MS,
    TASK_RELEVANCE, CATEGORIES, FDR_ALPHA, RANDOM_SEED,
    COLOR_RELEVANT, COLOR_IRRELEVANT, COLOR_HYSTERESIS,
    FIGURE_DPI, FIGURE_FORMAT,
)
from utils import (
    compute_onset_latency, compute_onset_latency_per_trial,
    permutation_test_early_vs_late, fit_exponential_decay,
    duration_tracking_index,
)


# ═════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ═════════════════════════════════════════════════════════

def load_subject_data(subject, output_dir=None):
    """Load preprocessed epochs and electrode selection for a subject."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    epochs_path = output_dir / f'sub-{subject}_hg_epochs-epo.fif'
    selection_path = output_dir / f'sub-{subject}_electrode_selection.json'

    if not epochs_path.exists():
        print(f"  Epochs not found: {epochs_path}")
        return None, None

    epochs = mne.read_epochs(epochs_path, verbose=False)

    selection = None
    if selection_path.exists():
        with open(selection_path) as f:
            selection = json.load(f)

    return epochs, selection


def get_condition_epochs(epochs, category, task_relevance):
    """
    Extract epochs matching a specific category and task relevance.

    Parameters
    ----------
    epochs : mne.Epochs
        Full epoched dataset.
    category : str
        Stimulus category (e.g., 'face').
    task_relevance : str
        Task relevance label (e.g., 'Relevant non-target', 'Irrelevant').

    Returns
    -------
    subset : mne.Epochs or None
        Matching epochs, or None if insufficient trials.
    """
    try:
        subset = epochs[f'{category}/{task_relevance}']
        if len(subset) < 5:
            return None
        return subset
    except (KeyError, RuntimeError):
        # Try alternative selection via metadata
        if epochs.metadata is not None:
            mask = (
                (epochs.metadata['category'] == category) &
                (epochs.metadata['task_relevance'] == task_relevance)
            )
            if mask.sum() < 5:
                return None
            return epochs[mask]
        return None


def get_task_relevance_labels(epochs):
    """Extract per-trial task relevance labels."""
    if epochs.metadata is not None and 'task_relevance' in epochs.metadata.columns:
        return epochs.metadata['task_relevance'].values

    inv_id = {v: k for k, v in epochs.event_id.items()}
    labels = []
    for ev in epochs.events[:, 2]:
        desc = inv_id.get(ev, '')
        if 'Relevant target' in desc:
            labels.append('Relevant target')
        elif 'Relevant non-target' in desc:
            labels.append('Relevant non-target')
        elif 'Irrelevant' in desc:
            labels.append('Irrelevant')
        else:
            labels.append('unknown')
    return np.array(labels)


def get_duration_labels(epochs):
    """Extract per-trial duration labels in ms."""
    if epochs.metadata is not None and 'duration_ms' in epochs.metadata.columns:
        return epochs.metadata['duration_ms'].values

    inv_id = {v: k for k, v in epochs.event_id.items()}
    labels = []
    for ev in epochs.events[:, 2]:
        desc = inv_id.get(ev, '')
        if '500ms' in desc:
            labels.append(500)
        elif '1000ms' in desc:
            labels.append(1000)
        elif '1500ms' in desc:
            labels.append(1500)
        else:
            labels.append(0)
    return np.array(labels)


def get_miniblock_labels(epochs):
    """Extract per-trial miniblock labels."""
    if epochs.metadata is not None and 'miniblock' in epochs.metadata.columns:
        return epochs.metadata['miniblock'].values

    inv_id = {v: k for k, v in epochs.event_id.items()}
    labels = []
    for ev in epochs.events[:, 2]:
        desc = inv_id.get(ev, '')
        mb = 0
        for part in desc.split('/'):
            if part.strip().startswith('miniblock_'):
                mb = int(part.strip().replace('miniblock_', ''))
        labels.append(mb)
    return np.array(labels)


# ═════════════════════════════════════════════════════════
# PREDICTION A — Constraint Load Delays Resolution
# ═════════════════════════════════════════════════════════

def run_prediction_a(subject, epochs, selection, output_dir=None):
    """
    Test Prediction A: High-gamma onset latency is later for
    Relevant Non-Targets than for Irrelevant stimuli of the same category.

    Returns per-electrode latency comparisons.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n  ── Prediction A: Constraint-Load Onset Latency ──")

    selective_chs = selection.get('selective_channels', [])
    if not selective_chs:
        print("  No selective channels. Skipping Prediction A.")
        return None

    times = epochs.times
    results = []

    for category in CATEGORIES:
        # Get epochs for this category under each task relevance
        ep_nontarget = get_condition_epochs(epochs, category, 'Relevant non-target')
        ep_irrelevant = get_condition_epochs(epochs, category, 'Irrelevant')

        if ep_nontarget is None or ep_irrelevant is None:
            continue

        for ch_name in selective_chs:
            if ch_name not in epochs.ch_names:
                continue
            ch_idx = epochs.ch_names.index(ch_name)

            # Mean response traces
            trace_nt = np.mean(ep_nontarget.get_data()[:, ch_idx, :], axis=0)
            trace_ir = np.mean(ep_irrelevant.get_data()[:, ch_idx, :], axis=0)

            # Compute onset latency
            lat_nt = compute_onset_latency(
                trace_nt, times,
                search_window=ONSET_LATENCY_WINDOW,
                threshold_frac=ONSET_LATENCY_THRESHOLD
            )
            lat_ir = compute_onset_latency(
                trace_ir, times,
                search_window=ONSET_LATENCY_WINDOW,
                threshold_frac=ONSET_LATENCY_THRESHOLD
            )

            results.append({
                'subject': subject,
                'channel': ch_name,
                'category': category,
                'latency_nontarget': lat_nt,
                'latency_irrelevant': lat_ir,
                'latency_diff': lat_nt - lat_ir if not (np.isnan(lat_nt) or np.isnan(lat_ir)) else np.nan,
            })

    if not results:
        print("  No valid comparisons for Prediction A.")
        return None

    df = pd.DataFrame(results)
    valid = df.dropna(subset=['latency_diff'])

    if len(valid) > 0:
        mean_diff = valid['latency_diff'].mean() * 1000  # convert to ms
        n_positive = (valid['latency_diff'] > 0).sum()
        n_total = len(valid)

        # Sign test: is the difference systematically positive?
        if len(valid) >= 5:
            try:
                stat, p_val = wilcoxon(valid['latency_diff'], alternative='greater')
            except ValueError:
                stat, p_val = np.nan, np.nan
        else:
            stat, p_val = np.nan, np.nan

        print(f"  Results: mean Δlatency = {mean_diff:.1f} ms "
              f"(NT later in {n_positive}/{n_total} comparisons)")
        print(f"  Wilcoxon signed-rank: W={stat:.1f}, p={p_val:.4f}")
        print(f"  Prediction A {'SUPPORTED' if p_val < FDR_ALPHA and mean_diff > 0 else 'NOT SUPPORTED'}")
    else:
        print("  Insufficient valid comparisons.")

    # Save
    save_path = output_dir / f'sub-{subject}_prediction_a.csv'
    df.to_csv(save_path, index=False)

    return df


# ═════════════════════════════════════════════════════════
# PREDICTION B — Hysteresis Across Blocks
# ═════════════════════════════════════════════════════════

def run_prediction_b(subject, epochs, selection, output_dir=None):
    """
    Test Prediction B: When a category transitions from Relevant to
    Irrelevant, early post-transition trials show elevated response
    that decays over trials.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n  ── Prediction B: Constraint Hysteresis ──")

    selective_chs = selection.get('selective_channels', [])
    if not selective_chs:
        print("  No selective channels. Skipping Prediction B.")
        return None

    times = epochs.times
    amp_mask = (times >= HYSTERESIS_AMPLITUDE_WINDOW[0]) & (times <= HYSTERESIS_AMPLITUDE_WINDOW[1])

    # Get miniblock labels and task relevance per trial
    miniblocks = get_miniblock_labels(epochs)
    task_rel = get_task_relevance_labels(epochs)

    unique_mbs = sorted(set(miniblocks[miniblocks > 0]))
    if len(unique_mbs) < 2:
        print("  Insufficient miniblocks for transition analysis.")
        return None

    results = []

    # For each category, find transitions from relevant → irrelevant
    for category in CATEGORIES:
        # Get category labels
        inv_id = {v: k for k, v in epochs.event_id.items()}
        cat_mask = np.array([
            category in inv_id.get(ev, '').lower()
            for ev in epochs.events[:, 2]
        ])

        if not np.any(cat_mask):
            continue

        # Find miniblocks where this category was relevant vs irrelevant
        for i in range(len(unique_mbs) - 1):
            mb_prev = unique_mbs[i]
            mb_curr = unique_mbs[i + 1]

            # Trials in each miniblock for this category
            prev_mask = cat_mask & (miniblocks == mb_prev)
            curr_mask = cat_mask & (miniblocks == mb_curr)

            prev_rel = task_rel[prev_mask]
            curr_rel = task_rel[curr_mask]

            if len(prev_rel) == 0 or len(curr_rel) == 0:
                continue

            # Check if this is a relevant → irrelevant transition
            was_relevant = np.any(np.isin(prev_rel, ['Relevant target', 'Relevant non-target']))
            now_irrelevant = np.all(curr_rel == 'Irrelevant')

            if not (was_relevant and now_irrelevant):
                continue

            # Get trial-by-trial amplitudes for the current miniblock
            curr_indices = np.where(curr_mask)[0]

            for ch_name in selective_chs:
                if ch_name not in epochs.ch_names:
                    continue
                ch_idx = epochs.ch_names.index(ch_name)

                data = epochs.get_data()[curr_indices, ch_idx, :]
                trial_amps = np.mean(data[:, amp_mask], axis=1)

                if len(trial_amps) < HYSTERESIS_LATE_TRIALS_START + 2:
                    continue

                # Permutation test: early vs late
                perm_result = permutation_test_early_vs_late(
                    trial_amps,
                    n_early=HYSTERESIS_EARLY_TRIALS,
                    n_late_start=HYSTERESIS_LATE_TRIALS_START,
                    n_permutations=HYSTERESIS_N_PERMUTATIONS,
                    seed=RANDOM_SEED,
                )

                # Fit exponential decay
                decay_result = fit_exponential_decay(trial_amps)

                results.append({
                    'subject': subject,
                    'channel': ch_name,
                    'category': category,
                    'transition': f'mb{mb_prev}→mb{mb_curr}',
                    'transition_type': 'relevant_to_irrelevant',
                    'n_trials': len(trial_amps),
                    'early_mean': np.mean(trial_amps[:HYSTERESIS_EARLY_TRIALS]),
                    'late_mean': np.mean(trial_amps[HYSTERESIS_LATE_TRIALS_START:]),
                    'observed_diff': perm_result['observed_diff'],
                    'p_value': perm_result['p_value'],
                    'decay_a': decay_result['a'],
                    'decay_tau': decay_result['tau'],
                    'decay_r2': decay_result['r_squared'],
                })

    if not results:
        print("  No valid transitions found for Prediction B.")
        return None

    df = pd.DataFrame(results)
    n_sig = (df['p_value'] < FDR_ALPHA).sum()
    n_positive = (df['observed_diff'] > 0).sum()

    print(f"  Found {len(df)} transition events across {df['channel'].nunique()} channels")
    print(f"  Early > Late: {n_positive}/{len(df)} comparisons")
    print(f"  Significant (p<{FDR_ALPHA}): {n_sig}/{len(df)}")
    print(f"  Mean decay τ: {df['decay_tau'].mean():.1f} trials")
    print(f"  Prediction B {'SUPPORTED' if n_sig > len(df)*0.2 else 'NOT SUPPORTED'}")

    save_path = output_dir / f'sub-{subject}_prediction_b.csv'
    df.to_csv(save_path, index=False)

    return df


# ═════════════════════════════════════════════════════════
# PREDICTION C — Duration-Tracking Is Constraint-Modulated
# ═════════════════════════════════════════════════════════

def run_prediction_c(subject, epochs, selection, output_dir=None):
    """
    Test Prediction C: Duration-tracking (neural discrimination of
    500/1000/1500 ms) is stronger for Relevant Non-Target than Irrelevant.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n  ── Prediction C: Duration-Tracking Modulation ──")

    selective_chs = selection.get('selective_channels', [])
    if not selective_chs:
        print("  No selective channels. Skipping Prediction C.")
        return None

    times = epochs.times
    duration_labels = get_duration_labels(epochs)

    results = []

    for category in CATEGORIES:
        ep_nt = get_condition_epochs(epochs, category, 'Relevant non-target')
        ep_ir = get_condition_epochs(epochs, category, 'Irrelevant')

        if ep_nt is None or ep_ir is None:
            continue

        dur_nt = get_duration_labels(ep_nt)
        dur_ir = get_duration_labels(ep_ir)

        for ch_name in selective_chs:
            if ch_name not in epochs.ch_names:
                continue
            ch_idx = epochs.ch_names.index(ch_name)

            # Duration tracking for Non-Target
            data_nt = ep_nt.get_data()[:, ch_idx, :]
            dti_nt = duration_tracking_index(
                data_nt, times, DURATIONS_MS, dur_nt,
                window=DURATION_TRACKING_WINDOW
            )

            # Duration tracking for Irrelevant
            data_ir = ep_ir.get_data()[:, ch_idx, :]
            dti_ir = duration_tracking_index(
                data_ir, times, DURATIONS_MS, dur_ir,
                window=DURATION_TRACKING_WINDOW
            )

            results.append({
                'subject': subject,
                'channel': ch_name,
                'category': category,
                'rho_nontarget': dti_nt['rho'],
                'rho_irrelevant': dti_ir['rho'],
                'rho_diff': (dti_nt['rho'] - dti_ir['rho'])
                            if not (np.isnan(dti_nt['rho']) or np.isnan(dti_ir['rho']))
                            else np.nan,
                'p_nontarget': dti_nt['p_value'],
                'p_irrelevant': dti_ir['p_value'],
            })

    if not results:
        print("  No valid comparisons for Prediction C.")
        return None

    df = pd.DataFrame(results)
    valid = df.dropna(subset=['rho_diff'])

    if len(valid) > 0:
        mean_diff = valid['rho_diff'].mean()
        n_positive = (valid['rho_diff'] > 0).sum()
        n_total = len(valid)

        if len(valid) >= 5:
            try:
                stat, p_val = wilcoxon(valid['rho_diff'], alternative='greater')
            except ValueError:
                stat, p_val = np.nan, np.nan
        else:
            stat, p_val = np.nan, np.nan

        print(f"  Mean Δρ (NT - Irr) = {mean_diff:.4f}")
        print(f"  NT > Irr in {n_positive}/{n_total} comparisons")
        print(f"  Wilcoxon: W={stat:.1f}, p={p_val:.4f}")
        print(f"  Prediction C {'SUPPORTED' if p_val < FDR_ALPHA and mean_diff > 0 else 'NOT SUPPORTED'}")

    save_path = output_dir / f'sub-{subject}_prediction_c.csv'
    df.to_csv(save_path, index=False)

    return df


# ═════════════════════════════════════════════════════════
# GROUP-LEVEL ANALYSIS
# ═════════════════════════════════════════════════════════

def run_group_analysis(output_dir=None):
    """
    Aggregate single-subject results into group-level statistics.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n{'='*60}")
    print("GROUP-LEVEL ANALYSIS")
    print(f"{'='*60}")

    for pred_label, pred_name in [('a', 'Onset Latency'),
                                    ('b', 'Hysteresis'),
                                    ('c', 'Duration-Tracking')]:
        files = sorted(output_dir.glob(f'sub-*_prediction_{pred_label}.csv'))
        if not files:
            print(f"\n  Prediction {pred_label.upper()}: No data files found.")
            continue

        df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        print(f"\n  ── Prediction {pred_label.upper()}: {pred_name} ──")
        print(f"  Subjects: {df_all['subject'].nunique()}")
        print(f"  Total comparisons: {len(df_all)}")

        if pred_label == 'a':
            valid = df_all.dropna(subset=['latency_diff'])
            if len(valid) > 0:
                mean_ms = valid['latency_diff'].mean() * 1000
                print(f"  Mean Δlatency: {mean_ms:.1f} ms")
                try:
                    stat, p = wilcoxon(valid['latency_diff'], alternative='greater')
                    print(f"  Group Wilcoxon: W={stat:.1f}, p={p:.6f}")
                except ValueError:
                    print("  Could not compute group statistic")

        elif pred_label == 'b':
            n_sig = (df_all['p_value'] < FDR_ALPHA).sum()
            print(f"  Significant transitions: {n_sig}/{len(df_all)}")
            print(f"  Mean decay τ: {df_all['decay_tau'].mean():.1f} trials")

        elif pred_label == 'c':
            valid = df_all.dropna(subset=['rho_diff'])
            if len(valid) > 0:
                print(f"  Mean Δρ: {valid['rho_diff'].mean():.4f}")
                try:
                    stat, p = wilcoxon(valid['rho_diff'], alternative='greater')
                    print(f"  Group Wilcoxon: W={stat:.1f}, p={p:.6f}")
                except ValueError:
                    print("  Could not compute group statistic")

        # Save group results
        group_path = output_dir / f'group_prediction_{pred_label}.csv'
        df_all.to_csv(group_path, index=False)
        print(f"  Saved: {group_path}")


# ═════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═════════════════════════════════════════════════════════

def plot_prediction_a_summary(output_dir=None, figures_dir=None):
    """Generate summary figure for Prediction A."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR

    group_file = output_dir / 'group_prediction_a.csv'
    if not group_file.exists():
        return

    df = pd.read_csv(group_file).dropna(subset=['latency_diff'])
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Distribution of latency differences
    ax = axes[0]
    diffs_ms = df['latency_diff'] * 1000
    ax.hist(diffs_ms, bins=30, color=COLOR_RELEVANT, alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(diffs_ms.mean(), color='red', linestyle='-', linewidth=2)
    ax.set_xlabel('Onset Latency Difference (ms)\n(Non-Target − Irrelevant)')
    ax.set_ylabel('Count')
    ax.set_title('A. Constraint-Load Effect on Onset Latency')

    # Panel B: By category
    ax = axes[1]
    cat_means = df.groupby('category')['latency_diff'].mean() * 1000
    cat_sems = df.groupby('category')['latency_diff'].sem() * 1000
    cats = cat_means.index.tolist()
    ax.bar(cats, cat_means.values, yerr=cat_sems.values,
           color=COLOR_RELEVANT, alpha=0.7, capsize=5, edgecolor='white')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Mean Latency Difference (ms)')
    ax.set_title('B. Effect by Category')

    plt.tight_layout()
    fig.savefig(figures_dir / f'prediction_a_summary.{FIGURE_FORMAT}',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved Prediction A figure")


def plot_prediction_c_summary(output_dir=None, figures_dir=None):
    """Generate summary figure for Prediction C."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR

    group_file = output_dir / 'group_prediction_c.csv'
    if not group_file.exists():
        return

    df = pd.read_csv(group_file).dropna(subset=['rho_diff'])
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Scatter of rho values
    ax = axes[0]
    ax.scatter(df['rho_irrelevant'], df['rho_nontarget'],
               alpha=0.5, s=20, color=COLOR_RELEVANT)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Duration-Tracking ρ (Irrelevant)')
    ax.set_ylabel('Duration-Tracking ρ (Non-Target)')
    ax.set_title('A. Duration-Tracking: Constraint Modulation')

    # Panel B: Difference distribution
    ax = axes[1]
    ax.hist(df['rho_diff'], bins=30, color=COLOR_RELEVANT, alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(df['rho_diff'].mean(), color='red', linestyle='-', linewidth=2)
    ax.set_xlabel('Δρ (Non-Target − Irrelevant)')
    ax.set_ylabel('Count')
    ax.set_title('B. Distribution of Constraint Effect')

    plt.tight_layout()
    fig.savefig(figures_dir / f'prediction_c_summary.{FIGURE_FORMAT}',
                dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved Prediction C figure")


# ═════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Run prediction tests for COGITATE constraint-architecture reanalysis'
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID. Default: all preprocessed subjects.')
    parser.add_argument('--prediction', type=str, default=None,
                        choices=['A', 'B', 'C', 'a', 'b', 'c'],
                        help='Run single prediction. Default: all.')
    parser.add_argument('--group-only', action='store_true',
                        help='Only run group analysis (skip per-subject).')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    pred = args.prediction.upper() if args.prediction else None

    if not args.group_only:
        if args.subject:
            subjects = [args.subject]
        else:
            subjects = sorted([
                f.stem.split('_')[0].replace('sub-', '')
                for f in output_dir.glob('sub-*_electrode_selection.json')
            ])

        print(f"COGITATE Constraint-Architecture Reanalysis — Predictions")
        print(f"Subjects: {len(subjects)}")
        print(f"Predictions: {pred or 'A, B, C'}")

        for subject in subjects:
            print(f"\n{'='*60}")
            print(f"Subject: {subject}")
            print(f"{'='*60}")

            epochs, selection = load_subject_data(subject, output_dir)
            if epochs is None or selection is None:
                print(f"  Skipping {subject} — missing data")
                continue

            # Exclude target trials (motor response confound)
            task_rel = get_task_relevance_labels(epochs)
            non_target_mask = task_rel != 'Relevant target'
            if np.any(~non_target_mask):
                epochs = epochs[non_target_mask]
                print(f"  Excluded {(~non_target_mask).sum()} target trials, "
                      f"{len(epochs)} remaining")

            if pred is None or pred == 'A':
                run_prediction_a(subject, epochs, selection, output_dir)
            if pred is None or pred == 'B':
                run_prediction_b(subject, epochs, selection, output_dir)
            if pred is None or pred == 'C':
                run_prediction_c(subject, epochs, selection, output_dir)

    # Group analysis
    run_group_analysis(output_dir)

    # Figures
    print(f"\nGenerating figures...")
    plot_prediction_a_summary(output_dir, FIGURES_DIR)
    plot_prediction_c_summary(output_dir, FIGURES_DIR)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {output_dir}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
