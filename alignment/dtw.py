import numpy as np
from librosa.sequence import dtw
from distances import euclidean
import logging

LOGGER = logging.getLogger(__name__)

DEFAULT_DTW_KWARGS = dict(
    # steps for DTW
    step_sizes_sigma=np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int),
    # Additive weights
    weights_add=np.zeros(3, dtype=np.float32),
    # Multiplicative weights
    weights_mul = np.array([2, 1, 1], dtype=np.float32),
)

def dtw_alignment(perf_features, ref_features,
                  metric='euclidean',
                  dtw_kwargs=DEFAULT_DTW_KWARGS):
    """
    Compute an alignment using Dynamic Time Warping

    Parameters
    ----------
    perf_features : np.ndarray
        The feature representation of the performance. A 2D array of shape (num_frames_perf, num_features).
    ref_features : np.ndarray
        The feature representation of the reference performance. A 2D array of shape (num_frames_ref, num_features)
    metric : str
        Metric to compute the pairwise cost. Default is 'euclidean'.
    dtw_kwargs : dict
        Dictionary of arguments to be passed to `dtw`.

    Returns
    wp : np.ndarray
       Warping path. An array of size (num_frames_perf, 2) with the indices of the corresponding frame
       in perf_features and ref_features (i.e., which index of perf_features corresponds to which index
       of ref_features).
    D : np.ndarray
       Accumulated cost matrix
    """
    
    if metric == 'euclidean':
        pairwise_local_cost = euclidean

    LOGGER.info('Computing pairwise distance')
    C = pairwise_local_cost(perf_features, ref_features)

    LOGGER.info('Computing DTW path')
    D, wp = dtw(C=C, **dtw_kwargs)

    return wp[::-1], D
