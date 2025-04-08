import numpy as np
import matplotlib.pyplot as plt


def brier_skill_score(y_true, y_pred, sample_weight=None):
    """
    Calculates the weighted Brier Skill Score (BSS).

    The Brier Skill Score evaluates the accuracy of probabilistic predictions
    in comparison to a baseline probabilistic prediction. This implementation
    supports weighted inputs, allowing the impact of each observation to be
    scaled by a specified weight.

    Parameters
    ----------
    y_true : ndarray
        Array of true binary outcome labels (0 or 1). Each entry corresponds
        to the observed outcome for a specific instance.
    y_pred : ndarray
        Array of predicted probabilities, each representing the probability
        of the positive class (label=1) for a corresponding instance.
    sample_weight : ndarray, optional
        Array of weights applied to each instance to scale their contribution.
        If not provided, all weights are assumed to be equal (default is uniform
        weights of 1).

    Returns
    -------
    float
        The weighted Brier Skill Score (BSS) value. A value close to or above 0
        indicates predictions are better than the baseline, whereas values below
        0 suggest predictions are worse than the baseline. Returns NaN when the
        baseline score is undefined (e.g., when all true labels are 0 or 1).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.array(sample_weight, dtype=float)

    assert len(y_true) == len(y_pred) == len(sample_weight), \
        "Length y_true, y_pred and sample_weight are not equal."

    # 1. Weighted Brier Score
    #    BS_model = sum(w_i * (p_i - y_i)^2) / sum(w_i)
    bs_model = np.average((y_pred - y_true) ** 2, weights=sample_weight)

    # 2. climatological probability
    #    p = sum(w_i * y_i) / sum(w_i)
    p_avg = np.average(y_true, weights=sample_weight)

    # 3. Brier Score of baseline (p * (1 - p))
    bs_baseline = p_avg * (1.0 - p_avg)

    # 4. Brier Skill Score:
    #    BSS = 1 - BS_model / BS_baseline
    if bs_baseline > 0:
        bss = 1.0 - (bs_model / bs_baseline)
    else:
        bss = np.nan

    return bss


def weighted_calibration_curve(y_true, y_prob, sample_weight=None, n_bins=10, strategy='quantile'):
    """
    Computes the weighted calibration curve to evaluate the calibration of probability
    predictions against the ground truth. This function calculates the average true
    class probabilities and predicted probabilities in discrete bins, optionally
    weighted by sample importance.

    The function supports two binning strategies: uniformly spaced bins or quantile-based
    bins that allocate approximately equal amounts of objects per bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (true binary labels) where values are either 0 or 1.

    y_prob : array-like of shape (n_samples,)
        Predicted probabilities corresponding to the positive class.

    sample_weight : array-like of shape (n_samples,) or None, optional (default=None)
        Sample weights. If None, all samples are assigned equal weight.

    n_bins : int, optional (default=10)
        Number of bins to be used to discretize the data.

    strategy : {'uniform', 'quantile'}, optional (default='quantile')
        Strategy used to define the bin edges:

        - 'uniform': Bins have equal width in the [0, 1] range.
        - 'quantile': Bins are defined by quantile thresholds to achieve (approximately)
          equal distribution of samples across bins.

    Returns
    -------
    bin_true : ndarray of shape (n_bins,)
        Mean true probabilities within each bin.

    bin_pred : ndarray of shape (n_bins,)
        Mean predicted probabilities within each bin.
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.array(sample_weight, dtype=float)

    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]
    w_sorted = sample_weight[sort_idx]

    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_sorted, bin_edges) - 1
    elif strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
        bin_edges = np.interp(quantiles, cdf, y_prob_sorted)
        bin_indices = np.searchsorted(bin_edges, y_prob_sorted, side='right') - 1
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")

    bin_true_sum = np.zeros(n_bins)
    bin_weight_sum = np.zeros(n_bins)
    bin_prob_sum = np.zeros(n_bins)

    for i in range(len(y_true_sorted)):
        b = bin_indices[i]
        if b < 0:
            b = 0
        elif b >= n_bins:
            b = n_bins - 1

        bin_true_sum[b] += y_true_sorted[i] * w_sorted[i]
        bin_weight_sum[b] += w_sorted[i]
        bin_prob_sum[b] += y_prob_sorted[i] * w_sorted[i]

    bin_true = np.divide(bin_true_sum, bin_weight_sum, out=np.zeros_like(bin_true_sum), where=bin_weight_sum > 0)
    bin_pred = np.divide(bin_prob_sum, bin_weight_sum, out=np.zeros_like(bin_prob_sum), where=bin_weight_sum > 0)

    return bin_true, bin_pred


def get_weighted_calibration_curve(y_true, y_prob, sample_weight=None, n_bins=10, strategy='quantile'):
    bin_true, bin_pred = weighted_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        sample_weight=sample_weight,
        n_bins=n_bins,
        strategy=strategy,
    )
    fig, ax = plt.subplots()
    ax.plot(bin_pred, bin_true, marker='o')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Fraction of true class')
    ax.set_title('Calibration curve (with weights)')
    plt.close()
    return fig