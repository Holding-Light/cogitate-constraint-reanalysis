"""
Utility Functions for COGITATE Constraint-Architecture Reanalysis
=================================================================
Shared helpers for high-gamma extraction, onset detection,
statistical testing, and electrode labeling.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
import warnings


def extract_high_gamma(raw, l_freq=70.0, h_freq=150.0,
                       smooth_sigma=0.050, log_transform=True):
    """
    Extract high-gamma analytic amplitude from raw iEEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw iEEG data (already preprocessed, bad channels excluded).
    l_freq, h_freq : float
        High-gamma band edges in Hz.
    smooth_sigma : float
        Gaussian smoothing kernel width in seconds.
    log_transform : bool
        Whether to log-transform the amplitude.

    Returns
    -------
    raw_hg : mne.io.Raw
        Raw object with high-gamma analytic amplitude as data.
    """
    # Bandpass filter to high-gamma range
    raw_filt = raw.copy().filter(
        l_freq=l_freq, h_freq=h_freq,
        method='iir', iir_params=dict(order=4, ftype='butter'),
        verbose=False
    )

    # Hilbert transform → analytic amplitude
    data = raw_filt.get_data()
    analytic = np.abs(hilbert(data, axis=-1))

    # Gaussian smoothing
    sfreq = raw_filt.info['sfreq']
    sigma_samples = smooth_sigma * sfreq
    analytic = gaussian_filter1d(analytic, sigma=sigma_samples, axis=-1)

    # Log transform (stabilizes variance, approximates normality)
    if log_transform:
        analytic = np.log10(analytic + 1e-10)  # small offset to avoid log(0)

    # Create new Raw with HG amplitude
    raw_hg = raw_filt.copy()
    raw_hg._data = analytic

    return raw_hg


def compute_onset_latency(trace, times, search_window=(0.05, 0.5),
                          threshold_frac=0.5):
    """
    Compute onset latency of a neural response.

    Onset is defined as the first time point within the search window
    at which the response reaches threshold_frac of its peak amplitude.

    Parameters
    ----------
    trace : np.ndarray
        1D array of response amplitude (e.g., mean high-gamma across trials).
    times : np.ndarray
        1D array of time points in seconds.
    search_window : tuple
        (start, end) of search window in seconds.
    threshold_frac : float
        Fraction of peak to use as threshold (default 0.5 = 50% of peak).

    Returns
    -------
    latency : float or np.nan
        Onset latency in seconds. NaN if not found.
    """
    mask = (times >= search_window[0]) & (times <= search_window[1])
    windowed = trace[mask]
    windowed_times = times[mask]

    if len(windowed) == 0:
        return np.nan

    peak = np.max(windowed)
    baseline = np.mean(trace[(times >= -0.3) & (times <= 0.0)])
    threshold = baseline + threshold_frac * (peak - baseline)

    crossings = np.where(windowed >= threshold)[0]
    if len(crossings) == 0:
        return np.nan

    return windowed_times[crossings[0]]


def compute_onset_latency_per_trial(epochs_data, times, search_window, 
                                     threshold_frac=0.5):
    """
    Compute onset latency for each trial individually.

    Parameters
    ----------
    epochs_data : np.ndarray
        Shape (n_trials, n_times) — single electrode.
    times : np.ndarray
        Time vector.
    search_window : tuple
        (start, end) seconds.
    threshold_frac : float
        Fraction of peak.

    Returns
    -------
    latencies : np.ndarray
        Onset latency per trial (NaN where not detectable).
    """
    latencies = np.array([
        compute_onset_latency(trial, times, search_window, threshold_frac)
        for trial in epochs_data
    ])
    return latencies


def onset_responsiveness_test(epochs, pre_window=(-0.3, 0.0),
                               post_window=(0.05, 0.35)):
    """
    Test whether each electrode shows significant onset response.

    Compares mean high-gamma amplitude in pre-stimulus vs. post-stimulus
    windows using paired t-test across trials.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched high-gamma data.
    pre_window, post_window : tuple
        Time windows in seconds.

    Returns
    -------
    results : dict
        Keys: channel names. Values: dict with 't_stat', 'p_value'.
    """
    times = epochs.times
    pre_mask = (times >= pre_window[0]) & (times <= pre_window[1])
    post_mask = (times >= post_window[0]) & (times <= post_window[1])

    results = {}
    data = epochs.get_data()  # (n_trials, n_channels, n_times)

    for idx, ch_name in enumerate(epochs.ch_names):
        pre_amp = np.mean(data[:, idx, pre_mask], axis=1)
        post_amp = np.mean(data[:, idx, post_mask], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_stat, p_val = ttest_rel(post_amp, pre_amp)

        results[ch_name] = {
            'ch_idx': idx,
            't_stat': t_stat,
            'p_value': p_val,
            'mean_pre': np.mean(pre_amp),
            'mean_post': np.mean(post_amp),
            'responsive': False,  # set after FDR correction
        }

    return results


def fdr_correct(results_dict, alpha=0.05):
    """
    Apply FDR correction to onset responsiveness results.

    Parameters
    ----------
    results_dict : dict
        Output of onset_responsiveness_test.
    alpha : float
        FDR threshold.

    Returns
    -------
    results_dict : dict
        Updated with 'responsive' and 'p_fdr' fields.
    """
    ch_names = list(results_dict.keys())
    p_values = np.array([results_dict[ch]['p_value'] for ch in ch_names])

    # Only consider positive t-stats (post > pre)
    t_stats = np.array([results_dict[ch]['t_stat'] for ch in ch_names])

    reject, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    for i, ch in enumerate(ch_names):
        results_dict[ch]['p_fdr'] = p_fdr[i]
        results_dict[ch]['responsive'] = reject[i] and t_stats[i] > 0

    return results_dict


def get_electrode_roi(ch_name, montage_labels, rois):
    """
    Map an electrode to a region of interest based on atlas labels.

    Parameters
    ----------
    ch_name : str
        Channel name.
    montage_labels : dict
        Output of mne.get_montage_volume_labels().
    rois : dict
        ROI definitions from config (keys: roi_name, values: list of labels).

    Returns
    -------
    roi_name : str or None
        Name of the ROI, or None if not in any ROI.
    """
    if ch_name not in montage_labels:
        return None

    labels = montage_labels[ch_name]
    if not labels:
        return None

    # Check each ROI
    for roi_name, roi_labels in rois.items():
        for label in labels:
            # Desikan labels may have hemisphere prefix — strip it
            clean_label = label.replace('ctx-lh-', '').replace('ctx-rh-', '')
            clean_label = clean_label.replace('Left-', '').replace('Right-', '')
            if clean_label.lower() in [r.lower() for r in roi_labels]:
                return roi_name

    return None


def fit_exponential_decay(amplitudes, trial_indices=None):
    """
    Fit exponential decay to trial-by-trial amplitudes.

    Model: amplitude = a * exp(-trial / tau) + baseline

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitude values per trial.
    trial_indices : np.ndarray or None
        Trial position indices (default: 0, 1, 2, ...).

    Returns
    -------
    params : dict
        'a' (initial elevation), 'tau' (decay constant),
        'baseline' (asymptote), 'r_squared' (goodness of fit).
    """
    from scipy.optimize import curve_fit

    if trial_indices is None:
        trial_indices = np.arange(len(amplitudes))

    # Remove NaNs
    valid = ~np.isnan(amplitudes)
    x = trial_indices[valid].astype(float)
    y = amplitudes[valid]

    if len(y) < 4:
        return {'a': np.nan, 'tau': np.nan, 'baseline': np.nan,
                'r_squared': np.nan}

    def decay_model(t, a, tau, baseline):
        return a * np.exp(-t / tau) + baseline

    try:
        # Initial guesses
        a0 = y[0] - y[-1]
        tau0 = len(y) / 3.0
        baseline0 = y[-1]

        popt, pcov = curve_fit(
            decay_model, x, y,
            p0=[a0, max(tau0, 0.5), baseline0],
            bounds=([0, 0.1, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=5000
        )

        y_pred = decay_model(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'a': popt[0],
            'tau': popt[1],
            'baseline': popt[2],
            'r_squared': r_squared,
        }
    except (RuntimeError, ValueError):
        return {'a': np.nan, 'tau': np.nan, 'baseline': np.nan,
                'r_squared': np.nan}


def permutation_test_early_vs_late(amplitudes, n_early, n_late_start,
                                    n_permutations=10000, seed=42):
    """
    Permutation test for early-vs-late amplitude difference.

    Parameters
    ----------
    amplitudes : np.ndarray
        Trial-by-trial amplitudes.
    n_early : int
        Number of early trials to average.
    n_late_start : int
        Index from which to start 'late' trials.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    result : dict
        'observed_diff', 'p_value', 'null_distribution'.
    """
    rng = np.random.RandomState(seed)

    early = amplitudes[:n_early]
    late = amplitudes[n_late_start:]

    # Remove NaNs
    early = early[~np.isnan(early)]
    late = late[~np.isnan(late)]

    if len(early) < 2 or len(late) < 2:
        return {'observed_diff': np.nan, 'p_value': np.nan,
                'null_distribution': np.array([])}

    observed_diff = np.mean(early) - np.mean(late)

    # Pool and shuffle
    pooled = np.concatenate([early, late])
    n_e = len(early)
    null_diffs = np.zeros(n_permutations)

    for i in range(n_permutations):
        rng.shuffle(pooled)
        null_diffs[i] = np.mean(pooled[:n_e]) - np.mean(pooled[n_e:])

    # One-sided p-value (we predict early > late)
    p_value = np.mean(null_diffs >= observed_diff)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'null_distribution': null_diffs,
    }


def duration_tracking_index(epochs_data, times, durations_ms,
                            duration_labels, window=(0.5, 1.5)):
    """
    Compute duration-tracking index: Spearman correlation between
    stimulus duration and sustained high-gamma amplitude.

    Parameters
    ----------
    epochs_data : np.ndarray
        Shape (n_trials, n_times) for a single electrode.
    times : np.ndarray
        Time vector.
    durations_ms : list
        Duration values in ms (e.g., [500, 1000, 1500]).
    duration_labels : np.ndarray
        Per-trial duration label (matching durations_ms values).
    window : tuple
        (start, end) of sustained response window in seconds.

    Returns
    -------
    result : dict
        'rho' (Spearman correlation), 'p_value', 'mean_by_duration'.
    """
    mask = (times >= window[0]) & (times <= window[1])
    sustained_amp = np.mean(epochs_data[:, mask], axis=1)

    valid = ~np.isnan(sustained_amp)
    if np.sum(valid) < 10:
        return {'rho': np.nan, 'p_value': np.nan, 'mean_by_duration': {}}

    rho, p_val = spearmanr(duration_labels[valid], sustained_amp[valid])

    mean_by_dur = {}
    for d in durations_ms:
        d_mask = duration_labels == d
        if np.any(d_mask & valid):
            mean_by_dur[d] = np.mean(sustained_amp[d_mask & valid])

    return {
        'rho': rho,
        'p_value': p_val,
        'mean_by_duration': mean_by_dur,
    }
