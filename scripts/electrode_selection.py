"""
Electrode Selection for COGITATE Constraint-Architecture Reanalysis
===================================================================
Identifies onset-responsive and category-selective electrodes.

Usage:
    python electrode_selection.py
    python electrode_selection.py --subject CE101
"""

import argparse
import json
from pathlib import Path
import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import (
    OUTPUT_DIR, FIGURES_DIR,
    ONSET_PRE_WINDOW, ONSET_POST_WINDOW, ONSET_P_THRESHOLD,
    DECODING_WINDOW, DECODING_N_FOLDS, DECODING_N_PERMUTATIONS,
    DECODING_P_THRESHOLD, FDR_ALPHA, ROIS, CATEGORIES, RANDOM_SEED,
)
from utils import onset_responsiveness_test, fdr_correct, get_electrode_roi


def identify_onset_responsive(epochs, p_threshold=None):
    """
    Identify electrodes with significant onset responses.

    Parameters
    ----------
    epochs : mne.Epochs
        High-gamma epoched data.
    p_threshold : float
        FDR-corrected p-value threshold.

    Returns
    -------
    responsive_channels : list
        Names of onset-responsive channels.
    results : dict
        Full results per channel.
    """
    if p_threshold is None:
        p_threshold = ONSET_P_THRESHOLD

    print("  Testing onset responsiveness...")
    results = onset_responsiveness_test(
        epochs,
        pre_window=ONSET_PRE_WINDOW,
        post_window=ONSET_POST_WINDOW,
    )

    results = fdr_correct(results, alpha=p_threshold)

    responsive = [ch for ch, r in results.items() if r['responsive']]
    print(f"  Found {len(responsive)} / {len(results)} onset-responsive channels")

    return responsive, results


def decode_category(epochs, channels, window=None, n_folds=None,
                    n_permutations=None):
    """
    Test category selectivity via SVM decoding.

    For each electrode, train an SVM to decode stimulus category
    from high-gamma features in the specified window.

    Parameters
    ----------
    epochs : mne.Epochs
        High-gamma epochs.
    channels : list
        Channel names to test.
    window : tuple
        Time window for feature extraction.
    n_folds : int
        CV folds.
    n_permutations : int
        Permutation test iterations.

    Returns
    -------
    results : dict
        Per-channel decoding accuracy, p-value, and selectivity.
    """
    if window is None:
        window = DECODING_WINDOW
    if n_folds is None:
        n_folds = DECODING_N_FOLDS
    if n_permutations is None:
        n_permutations = DECODING_N_PERMUTATIONS

    times = epochs.times
    t_mask = (times >= window[0]) & (times <= window[1])

    results = {}

    for ch_name in channels:
        ch_idx = epochs.ch_names.index(ch_name)

        # Extract features: mean HG amplitude in window per trial
        data = epochs.get_data()[:, ch_idx, :]  # (n_trials, n_times)
        X = data[:, t_mask]  # (n_trials, n_window_samples)

        # Get category labels from metadata or events
        labels = _get_category_labels(epochs)
        if labels is None:
            results[ch_name] = {
                'accuracy': np.nan, 'p_value': np.nan,
                'selective': False, 'best_contrast': None
            }
            continue

        # Test all pairwise category contrasts
        best_acc = 0
        best_p = 1.0
        best_contrast = None

        for i, cat_a in enumerate(CATEGORIES):
            for cat_b in CATEGORIES[i+1:]:
                mask_a = labels == cat_a
                mask_b = labels == cat_b

                if np.sum(mask_a) < 10 or np.sum(mask_b) < 10:
                    continue

                X_pair = np.vstack([X[mask_a], X[mask_b]])
                y_pair = np.array([0] * np.sum(mask_a) + [1] * np.sum(mask_b))

                clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
                cv = StratifiedKFold(n_splits=min(n_folds, min(np.sum(mask_a), np.sum(mask_b))),
                                     shuffle=True, random_state=RANDOM_SEED)

                try:
                    score, perm_scores, p_val = permutation_test_score(
                        clf, X_pair, y_pair, cv=cv,
                        n_permutations=min(n_permutations, 200),  # reduced for speed
                        scoring='accuracy',
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    )

                    if score > best_acc:
                        best_acc = score
                        best_p = p_val
                        best_contrast = f"{cat_a}_vs_{cat_b}"
                except Exception:
                    continue

        results[ch_name] = {
            'accuracy': best_acc,
            'p_value': best_p,
            'selective': best_p < DECODING_P_THRESHOLD,
            'best_contrast': best_contrast,
        }

        status = "✓" if results[ch_name]['selective'] else "✗"
        print(f"    {ch_name}: acc={best_acc:.3f}, p={best_p:.3f} [{best_contrast}] {status}")

    return results


def _get_category_labels(epochs):
    """Extract category labels from epoch metadata or event descriptions."""
    # Try metadata first
    if epochs.metadata is not None and 'category' in epochs.metadata.columns:
        return epochs.metadata['category'].values

    # Try event descriptions
    labels = []
    for event_desc in epochs.event_id.keys():
        found = False
        for cat in CATEGORIES:
            if cat in event_desc.lower():
                labels.append(cat)
                found = True
                break
        if not found:
            labels.append('unknown')

    # Map event IDs to trial-level labels
    events = epochs.events[:, 2]
    inv_event_id = {v: k for k, v in epochs.event_id.items()}
    trial_labels = []
    for ev in events:
        desc = inv_event_id.get(ev, '')
        found = False
        for cat in CATEGORIES:
            if cat in desc.lower():
                trial_labels.append(cat)
                found = True
                break
        if not found:
            trial_labels.append('unknown')

    trial_labels = np.array(trial_labels)
    if np.all(trial_labels == 'unknown'):
        return None
    return trial_labels


def run_electrode_selection(subject, output_dir=None):
    """
    Run full electrode selection pipeline for one subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    output_dir : Path
        Directory containing preprocessed epochs.

    Returns
    -------
    selection : dict
        Selected channels with responsiveness and selectivity info.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Load preprocessed epochs
    epochs_path = output_dir / f'sub-{subject}_hg_epochs-epo.fif'
    if not epochs_path.exists():
        print(f"  Preprocessed epochs not found: {epochs_path}")
        return None

    epochs = mne.read_epochs(epochs_path, verbose=False)
    print(f"  Loaded {len(epochs)} epochs, {len(epochs.ch_names)} channels")

    # Step 1: Onset responsiveness
    responsive_chs, onset_results = identify_onset_responsive(epochs)

    if not responsive_chs:
        print("  No onset-responsive channels found. Skipping.")
        return None

    # Step 2: Category selectivity (on responsive channels only)
    print(f"  Testing category selectivity on {len(responsive_chs)} responsive channels...")
    decoding_results = decode_category(epochs, responsive_chs)

    selective_chs = [ch for ch, r in decoding_results.items() if r['selective']]
    print(f"  Found {len(selective_chs)} category-selective channels")

    # Step 3: Map to ROIs
    roi_mapping = {}
    for ch in selective_chs:
        # ROI assignment will be refined once we have montage labels
        # For now, store placeholder
        roi_mapping[ch] = 'unassigned'

    # Compile selection results
    selection = {
        'subject': subject,
        'n_channels_total': len(epochs.ch_names),
        'n_responsive': len(responsive_chs),
        'n_selective': len(selective_chs),
        'responsive_channels': responsive_chs,
        'selective_channels': selective_chs,
        'onset_results': {ch: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                               for k, v in r.items()}
                          for ch, r in onset_results.items()},
        'decoding_results': {ch: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                  for k, v in r.items()}
                             for ch, r in decoding_results.items()},
        'roi_mapping': roi_mapping,
    }

    # Save
    save_path = output_dir / f'sub-{subject}_electrode_selection.json'
    with open(save_path, 'w') as f:
        json.dump(selection, f, indent=2, default=str)
    print(f"  Saved to {save_path}")

    return selection


def main():
    parser = argparse.ArgumentParser(
        description='Electrode selection for COGITATE reanalysis'
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID. Default: all preprocessed subjects.')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted([
            f.stem.split('_')[0].replace('sub-', '')
            for f in output_dir.glob('sub-*_hg_epochs-epo.fif')
        ])

    print(f"COGITATE Constraint-Architecture Reanalysis — Electrode Selection")
    print(f"Subjects: {len(subjects)}")

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        run_electrode_selection(subject, output_dir)


if __name__ == '__main__':
    main()
