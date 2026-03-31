"""
Preprocessing Pipeline for COGITATE iEEG Reanalysis
====================================================
Loads BIDS-formatted iEEG data, extracts high-gamma analytic
amplitude, creates epochs, and saves preprocessed data.

Usage:
    python preprocessing.py                    # process all subjects
    python preprocessing.py --subject CE101    # process single subject
"""

import argparse
import json
from pathlib import Path
import numpy as np
import mne
import mne_bids

from config import (
    BIDS_ROOT, OUTPUT_DIR,
    HG_FREQ_LOW, HG_FREQ_HIGH, HG_SMOOTH_SIGMA, LOG_TRANSFORM,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE,
    TASK_RELEVANCE, CATEGORIES, RANDOM_SEED,
)
from utils import extract_high_gamma


def get_subject_list(bids_root):
    """Get list of available subjects from BIDS directory."""
    bids_path = Path(bids_root)
    subjects = sorted([
        d.name.replace('sub-', '')
        for d in bids_path.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])
    return subjects


def load_raw_ieeg(subject, bids_root, run='01'):
    """
    Load raw iEEG data for a subject from BIDS.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'CE101').
    bids_root : str or Path
        Path to BIDS root directory.
    run : str
        Run identifier.

    Returns
    -------
    raw : mne.io.Raw
        Raw iEEG data.
    """
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session='1',
        task='Dur',
        run=run,
        datatype='ieeg',
        root=bids_root,
    )

    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    return raw


def load_all_runs(subject, bids_root):
    """
    Load and concatenate all available runs for a subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    bids_root : str or Path
        BIDS root path.

    Returns
    -------
    raw : mne.io.Raw
        Concatenated raw data across runs.
    """
    bids_path = Path(bids_root)
    sub_dir = bids_path / f'sub-{subject}'

    # Find all run files
    runs = []
    for sess_dir in sorted(sub_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        ieeg_dir = sess_dir / 'ieeg'
        if not ieeg_dir.exists():
            continue
        for f in sorted(ieeg_dir.iterdir()):
            if 'task-Dur' in f.name and f.suffix in ['.vhdr', '.edf']:
                # Extract run number
                parts = f.stem.split('_')
                for p in parts:
                    if p.startswith('run-'):
                        runs.append(p.replace('run-', ''))

    if not runs:
        raise FileNotFoundError(f"No Dur task runs found for sub-{subject}")

    raws = []
    for run in sorted(set(runs)):
        try:
            raw = load_raw_ieeg(subject, bids_root, run=run)
            raws.append(raw)
        except Exception as e:
            print(f"  Warning: Could not load run {run} for {subject}: {e}")
            continue

    if not raws:
        raise RuntimeError(f"No valid runs loaded for sub-{subject}")

    if len(raws) > 1:
        raw = mne.concatenate_raws(raws)
    else:
        raw = raws[0]

    return raw


def exclude_bad_channels(raw):
    """
    Exclude channels marked as bad (epileptic, noisy, flat).
    These are already annotated in the BIDS dataset.
    """
    if raw.info['bads']:
        print(f"  Excluding {len(raw.info['bads'])} bad channels: "
              f"{raw.info['bads'][:5]}{'...' if len(raw.info['bads']) > 5 else ''}")
    raw.drop_channels(raw.info['bads'])
    return raw


def apply_common_average_reference(raw):
    """Apply common average reference."""
    raw.set_eeg_reference('average', verbose=False)
    return raw


def create_epochs(raw_hg, event_id=None, tmin=None, tmax=None,
                  baseline=None):
    """
    Create epochs from high-gamma data time-locked to stimulus onset.

    Parameters
    ----------
    raw_hg : mne.io.Raw
        High-gamma amplitude data with events in annotations.
    event_id : dict or None
        Event ID mapping. If None, uses all 'stimulus onset' events.
    tmin, tmax : float
        Epoch boundaries in seconds.
    baseline : tuple
        Baseline correction window.

    Returns
    -------
    epochs : mne.Epochs
        Epoched data with metadata.
    """
    if tmin is None:
        tmin = EPOCH_TMIN
    if tmax is None:
        tmax = EPOCH_TMAX
    if baseline is None:
        baseline = BASELINE

    # Extract events from annotations
    events, event_id_all = mne.events_from_annotations(raw_hg, verbose=False)

    # Filter to stimulus onset events
    onset_ids = {k: v for k, v in event_id_all.items()
                 if 'stimulus onset' in k.lower() or 'stimulus_onset' in k.lower()}

    if not onset_ids:
        # Try alternate event naming
        onset_ids = {k: v for k, v in event_id_all.items()
                     if any(cat in k.lower() for cat in ['face', 'object', 'letter', 'false'])}

    if not onset_ids:
        print(f"  Warning: No stimulus onset events found. Available: {list(event_id_all.keys())[:10]}")
        return None

    epochs = mne.Epochs(
        raw_hg,
        events=events,
        event_id=onset_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    return epochs


def extract_trial_metadata(epochs):
    """
    Extract trial-level metadata from epoch event descriptions.

    Returns DataFrame with columns:
        category, identity, orientation, duration_ms,
        task_relevance, response, block, miniblock
    """
    import pandas as pd

    metadata_rows = []
    for event_desc in epochs.event_id.keys():
        # COGITATE events use '/' separators
        parts = event_desc.split('/')
        row = {}
        for part in parts:
            part = part.strip()
            if part in CATEGORIES:
                row['category'] = part
            elif part in TASK_RELEVANCE.values():
                row['task_relevance'] = part
            elif part.endswith('ms'):
                row['duration_ms'] = int(part.replace('ms', ''))
            elif part in ['Center', 'Left', 'Right']:
                row['orientation'] = part
            elif part.startswith('block_'):
                row['block'] = int(part.replace('block_', ''))
            elif part.startswith('miniblock_'):
                row['miniblock'] = int(part.replace('miniblock_', ''))
            elif part in ['Hit', 'CorrRej', 'Miss', 'FA']:
                row['response'] = part
        metadata_rows.append(row)

    if metadata_rows:
        return pd.DataFrame(metadata_rows)
    return None


def preprocess_subject(subject, bids_root=None, output_dir=None):
    """
    Full preprocessing pipeline for a single subject.

    Steps:
    1. Load all runs (BIDS format)
    2. Exclude bad channels
    3. Common average reference
    4. Extract high-gamma analytic amplitude
    5. Create epochs
    6. Save preprocessed epochs

    Parameters
    ----------
    subject : str
        Subject ID.
    bids_root : Path
        BIDS root directory.
    output_dir : Path
        Where to save preprocessed data.

    Returns
    -------
    epochs : mne.Epochs or None
        Preprocessed epochs, or None if failed.
    """
    if bids_root is None:
        bids_root = BIDS_ROOT
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print(f"\n{'='*60}")
    print(f"Processing sub-{subject}")
    print(f"{'='*60}")

    # Step 1: Load data
    try:
        raw = load_all_runs(subject, bids_root)
        print(f"  Loaded: {raw.info['nchan']} channels, "
              f"{raw.times[-1]:.1f}s duration, "
              f"{raw.info['sfreq']:.0f} Hz")
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    # Step 2: Exclude bad channels
    raw = exclude_bad_channels(raw)
    print(f"  Channels after exclusion: {raw.info['nchan']}")

    # Step 3: Common average reference
    try:
        raw = apply_common_average_reference(raw)
        print("  Applied common average reference")
    except Exception as e:
        print(f"  Warning: CAR failed ({e}), continuing without")

    # Step 4: Extract high-gamma
    print(f"  Extracting high-gamma ({HG_FREQ_LOW}-{HG_FREQ_HIGH} Hz)...")
    raw_hg = extract_high_gamma(
        raw,
        l_freq=HG_FREQ_LOW,
        h_freq=HG_FREQ_HIGH,
        smooth_sigma=HG_SMOOTH_SIGMA,
        log_transform=LOG_TRANSFORM,
    )
    print("  High-gamma extraction complete")

    # Step 5: Create epochs
    print("  Creating epochs...")
    epochs = create_epochs(raw_hg)
    if epochs is None:
        print("  ERROR: No epochs created")
        return None
    print(f"  Created {len(epochs)} epochs")

    # Step 6: Save
    save_path = output_dir / f'sub-{subject}_hg_epochs-epo.fif'
    epochs.save(save_path, overwrite=True, verbose=False)
    print(f"  Saved to {save_path}")

    # Save channel info
    info_path = output_dir / f'sub-{subject}_channel_info.json'
    ch_info = {
        'subject': subject,
        'n_channels': len(epochs.ch_names),
        'n_epochs': len(epochs),
        'ch_names': epochs.ch_names,
        'sfreq': epochs.info['sfreq'],
        'tmin': epochs.tmin,
        'tmax': epochs.tmax,
    }
    with open(info_path, 'w') as f:
        json.dump(ch_info, f, indent=2)

    return epochs


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess COGITATE iEEG data for constraint-architecture reanalysis'
    )
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID (e.g., CE101). Default: all subjects.')
    parser.add_argument('--bids-root', type=str, default=None,
                        help=f'BIDS root path. Default: {BIDS_ROOT}')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Output directory. Default: {OUTPUT_DIR}')
    args = parser.parse_args()

    bids_root = Path(args.bids_root) if args.bids_root else BIDS_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = get_subject_list(bids_root)

    print(f"COGITATE Constraint-Architecture Reanalysis — Preprocessing")
    print(f"BIDS root: {bids_root}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {len(subjects)}")

    results = {}
    for subject in subjects:
        epochs = preprocess_subject(subject, bids_root, output_dir)
        results[subject] = 'OK' if epochs is not None else 'FAILED'

    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    for subj, status in results.items():
        print(f"  sub-{subj}: {status}")
    n_ok = sum(1 for s in results.values() if s == 'OK')
    print(f"\n  {n_ok}/{len(results)} subjects preprocessed successfully")


if __name__ == '__main__':
    main()
