""" Utils for faults postprocessing. """

import numpy as np
from numba import njit, prange
from numba.types import bool_

@njit(parallel=True)
def skeletonize(slide, width=5, rel_height=0.5, threshold=0.05):
    """ Perform skeletonize of faults on 2D slide

    Parameters
    ----------
    slide : numpy.ndarray

    width : int, optional
        width of the resulting skeleton, by default 5
    rel_height, threshold : float, optional
        parameters of :meth:~.find_peaks`

    Returns
    -------
    numpy.ndarray
        skeletonized slide
    """
    skeletonized_slide = np.zeros_like(slide)
    for i in prange(slide.shape[1]): #pylint: disable=not-an-iterable
        x = slide[:, i]
        peaks = find_peaks(x, width=width, rel_height=rel_height, threshold=threshold)[0]
        skeletonized_slide[peaks, i] = 1
    return skeletonized_slide

@njit
def find_peaks(x, width=5, rel_height=0.5, threshold=0.05):
    """ See :meth:`scipy.signal.find_peaks`. """
    lmax = (x[1:] - x[:-1] >= 0)
    rmax = (x[:-1] - x[1:] >= 0)
    mask = np.empty(len(x))
    mask[0] = rmax[0]
    mask[-1] = lmax[-1]
    mask[1:-1] = np.logical_and(lmax[:-1], rmax[1:])
    mask = np.logical_and(mask, x >= threshold)
    peaks = np.where(mask)[0]

    prominences, left_bases, right_bases = _peak_prominences(x, peaks, -1)
    widths = _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases)
    return peaks[widths[0] >= width], None

@njit
def _peak_prominences(x, peaks, wlen):
    prominences = np.empty(peaks.shape[0], dtype=np.float32)
    left_bases = np.empty(peaks.shape[0], dtype=np.intp)
    right_bases = np.empty(peaks.shape[0], dtype=np.intp)

    for peak_nr in range(peaks.shape[0]):
        peak = peaks[peak_nr]
        i_min = 0
        i_max = x.shape[0] - 1

        if wlen >= 2:
            i_min = max(peak - wlen // 2, i_min)
            i_max = min(peak + wlen // 2, i_max)

        # Find the left base in interval [i_min, peak]
        i = left_bases[peak_nr] = peak
        left_min = x[peak]
        while i_min <= i and x[i] <= x[peak]:
            if x[i] < left_min:
                left_min = x[i]
                left_bases[peak_nr] = i
            i -= 1

        i = right_bases[peak_nr] = peak
        right_min = x[peak]
        while i <= i_max and x[i] <= x[peak]:
            if x[i] < right_min:
                right_min = x[i]
                right_bases[peak_nr] = i
            i += 1

        prominences[peak_nr] = x[peak] - max(left_min, right_min)

    return prominences, left_bases, right_bases

@njit
def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    widths = np.empty(peaks.shape[0], dtype=np.float64)
    width_heights = np.empty(peaks.shape[0], dtype=np.float64)
    left_ips = np.empty(peaks.shape[0], dtype=np.float64)
    right_ips = np.empty(peaks.shape[0], dtype=np.float64)

    for p in range(peaks.shape[0]):
        i_min = left_bases[p]
        i_max = right_bases[p]
        peak = peaks[p]
        # Validate bounds and order
        height = width_heights[p] = x[peak] - prominences[p] * rel_height

        # Find intersection point on left side
        i = peak
        while i_min < i and height < x[i]:
            i -= 1
        left_ip = i
        if x[i] < height:
            # Interpolate if true intersection height is between samples
            left_ip += (height - x[i]) / (x[i + 1] - x[i])

        # Find intersection point on right side
        i = peak
        while i < i_max and height < x[i]:
            i += 1
        right_ip = i
        if  x[i] < height:
            # Interpolate if true intersection height is between samples
            right_ip -= (height - x[i]) / (x[i - 1] - x[i])

        widths[p] = right_ip - left_ip
        left_ips[p] = left_ip
        right_ips[p] = right_ip

    return widths, width_heights, left_ips, right_ips

@njit
def _select_by_peak_distance(peaks, priority, distance):
    peaks_size = peaks.shape[0]
    distance_ = np.ceil(distance)
    keep = np.ones(peaks_size, bool_)  # Prepare array of flags
    priority_to_position = np.argsort(priority)

    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if keep[j] == 0:
            continue

        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance_:
            keep[k] = 0
            k -= 1

        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = 0
            k += 1
    return keep  # Return as boolean array

def faults_sizes(labels):
    """ Compute sizes of faults.

    Parameters
    ----------
    labels : numpy.ndarray
        array of shape (N, 4) where the first 3 columns are coordinates of points and the last one
        is for labels
    Returns
    -------
    sizes : numpy.ndarray
    """
    sizes = []
    for array in labels:
        i_len = array[:, 0].ptp()
        x_len = array[:, 1].ptp()
        sizes.append((i_len ** 2 + x_len ** 2) ** 0.5)
    return np.array(sizes)
