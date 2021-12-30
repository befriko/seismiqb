""" Utility functions. """
import os
from math import atan

import numpy as np
import torch
from numba import njit, prange
from scipy.ndimage import measurements

from .layers import SemblanceLayer, MovingNormalizationLayer, InstantaneousPhaseLayer, FrequenciesFilterLayer


def file_print(msg, path, mode='w'):
    """ Print to file. """
    with open(path, mode, encoding='utf-8') as file:
        print(msg, file=file)

def fill_defaults(value, default):
    """ #TODO: no longer needed, remove. """
    if value is None:
        value = default
    elif isinstance(value, int):
        value = tuple([value] * 3)
    elif isinstance(value, tuple):
        value = tuple(item if item else default[i] for i, item in enumerate(value))
    return value


def _adjust_shape_for_rotation(shape, angle):
    """ Compute adjusted 2D crop shape to rotate it and get central crop without padding.

    Parameters
    ----------
    shape : tuple
        Target сrop shape.
    angle : float

    Returns
    -------
    tuple
        Adjusted crop shape.
    """
    angle = abs(2 * np.pi * angle / 360)
    limit = atan(shape[1] / shape[0])
    x_max, y_max = shape
    if angle != 0:
        if angle < limit:
            x_max = shape[0] * np.cos(angle) + shape[1] * np.sin(angle) + 1
        else:
            x_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1

        if angle < np.pi / 2 - limit:
            y_max = shape[0] * np.sin(angle) + shape[1] * np.cos(angle) + 1
        else:
            y_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1
    return (int(np.ceil(x_max)), int(np.ceil(y_max)))

def adjust_shape_3d(shape, angle, scale=(1, 1, 1)):
    """ Compute adjusted 3D crop shape to rotate it and get central crop without padding. Adjustments is based on
    proposition that rotation angles are defined as Tait-Bryan angles and the sequence of extrinsic rotations axes
    is (axis_2, axis_0, axis_1) and scale performed after rotation.

    Parameters
    ----------
    shape : tuple
        Target сrop shape.
    angle : float or tuple of floats
        Rotation angles about each axis.
    scale : int or tuple, optional
        Scale for each axis.

    Returns
    -------
    tuple
        Adjusted crop shape.
    """
    angle = angle if isinstance(angle, (tuple, list)) else (angle, 0, 0)
    scale = scale if isinstance(scale, (tuple, list)) else (scale, scale, 1)
    shape = np.ceil(np.array(shape) / np.array(scale)).astype(int)
    if angle[2] != 0:
        shape[2], shape[0] = _adjust_shape_for_rotation((shape[2], shape[0]), angle[2])
    if angle[1] != 0:
        shape[2], shape[1] = _adjust_shape_for_rotation((shape[2], shape[1]), angle[1])
    if angle[0] != 0:
        shape[0], shape[1] = _adjust_shape_for_rotation((shape[0], shape[1]), angle[0])
    return tuple(shape)


@njit
def groupby_mean(array):
    """ Faster version of mean-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1], 1

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1]
            c += 1
        else:
            output[position, :2] = prev
            output[position, -1] = s / c
            position += 1

            prev = curr
            s, c = array[i, -1], 1

    output[position, :2] = prev
    output[position, -1] = s / c
    position += 1
    return output[:position]

@njit
def groupby_min(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = min(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]

@njit
def groupby_max(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = max(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]


@njit(parallel=True)
def filtering_function(points, filtering_matrix):
    """ Remove points where `filtering_matrix` is 1. """
    #pylint: disable=consider-using-enumerate, not-an-iterable
    mask = np.ones(len(points), dtype=np.int32)

    for i in prange(len(points)):
        il, xl = points[i, 0], points[i, 1]
        if filtering_matrix[il, xl] == 1:
            mask[i] = 0
    return points[mask == 1, :]

@njit
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.
    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        if p <= ticks[0]:
            values[i] = ticks[0]
        elif p >= ticks[-1]:
            values[i] = ticks[-1]
        else:
            ix = np.searchsorted(ticks, p)

            if abs(ticks[ix] - p) <= abs(ticks[ix-1] - p):
                values[i] = ticks[ix]
            else:
                values[i] = ticks[ix-1]
    return values

@njit
def find_min_max(array):
    """ Get both min and max values in just one pass through array."""
    n = array.size
    max_val = min_val = array[0]
    for i in range(1, n):
        min_val = min(array[i], min_val)
        max_val = max(array[i], max_val)
    return min_val, max_val


@njit
def filter_simplices(simplices, points, matrix, threshold=5.):
    """ Remove simplices outside of matrix. """
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(simplices), dtype=np.int32)

    for i in range(len(simplices)):
        tri = points[simplices[i]].astype(np.int32)

        middle_i, middle_x = np.mean(tri[:, 0]), np.mean(tri[:, 1])
        heights = np.array([matrix[tri[0, 0], tri[0, 1]],
                            matrix[tri[1, 0], tri[1, 1]],
                            matrix[tri[2, 0], tri[2, 1]]])

        if matrix[int(middle_i), int(middle_x)] < 0 or np.std(heights) > threshold:
            mask[i] = 0

    return simplices[mask == 1]

def compute_attribute(array, window=None, device='cuda:0', attribute='semblance', fill_value=None, **kwargs):
    """ Compute semblance for the cube. """
    if isinstance(window, int):
        window = np.ones(3, dtype=np.int32) * window
    window = np.minimum(np.array(window), array.shape[-3:])
    inputs = torch.Tensor(array).to(device)

    if attribute == 'semblance':
        layer = SemblanceLayer(inputs, window=window, fill_value=fill_value or 1)
    elif attribute == 'moving_normalization':
        layer = MovingNormalizationLayer(inputs, window=window, fill_value=fill_value or 1, **kwargs)
    elif attribute == 'phase':
        layer = InstantaneousPhaseLayer(inputs, **kwargs)
    elif attribute == 'frequencies_filter':
        layer = FrequenciesFilterLayer(inputs, window=window, **kwargs)
    result = layer(inputs)
    return result.cpu().numpy()

def make_bezier_figure(n=7, radius=0.2, sharpness=0.05, scale=1.0, shape=(1, 1),
                       resolution=None, distance=.5, seed=None):
    """ Bezier closed curve coordinates.
    Creates Bezier closed curve which passes through random points.
    Code based on:  https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib

    Parameters
    ----------
    n : int
        Number more than 1 to control amount of angles (key points) in the random figure.
        Must be more than 1.
    radius : float
        Number between 0 and 1 to control the distance of middle points in Bezier algorithm.
    sharpness : float
        Degree of sharpness/edgy. If 0 then a curve will be the smoothest.
    scale : float
        Number between 0 and 1 to control figure scale. Fits to the shape.
    shape : sequence int
        Shape of figure location area.
    resolution : int
        Amount of points in one curve between two key points.
    distance : float
        Number between 0 and 1 to control distance between all key points in a unit square.
    seed: int, optional
        Seed the random numbers generator.
    """
    rng = np.random.default_rng(seed)
    resolution = resolution or int(100 * scale * max(shape))

    # Get key points of figure as random points which are far enough each other
    key_points = rng.random((n, 2))
    squared_distance = distance ** 2

    squared_distances = squared_distance - 1
    while np.any(squared_distances < squared_distance):
        shifted_points = key_points - np.mean(key_points, axis=0)
        angles = np.arctan2(shifted_points[:, 0], shifted_points[:, 1])
        key_points = key_points[np.argsort(angles)]

        squared_distances = np.sum(np.diff(key_points, axis=0)**2, axis=1)
        key_points = rng.random((n, 2))

    key_points *= scale * np.array(shape, float)
    key_points = np.vstack([key_points, key_points[0]])

    # Calculate figure angles in key points
    p = np.arctan(sharpness) / np.pi + .5
    diff_between_points = np.diff(key_points, axis=0)
    angles = np.arctan2(diff_between_points[:, 1], diff_between_points[:, 0])
    angles = angles + 2 * np.pi * (angles < 0)
    rolled_angles = np.roll(angles, 1)
    angles = p * angles + (1 - p) * rolled_angles + np.pi * (np.abs(rolled_angles - angles) > np.pi)
    angles = np.append(angles, angles[0])

    # Create figure part by part: make curves between each pair of points
    curve_segments = []
    # Calculate control points for Bezier curve
    points_distances = np.sqrt(np.sum(diff_between_points ** 2, axis=1))
    radii = radius * points_distances
    middle_control_points_1 = np.transpose(radii * [np.cos(angles[:-1]),
                                                    np.sin(angles[:-1])]) + key_points[:-1]
    middle_control_points_2 = np.transpose(radii * [np.cos(angles[1:] + np.pi),
                                                    np.sin(angles[1:] + np.pi)]) + key_points[1:]
    curve_main_points_arr = np.hstack([key_points[:-1], middle_control_points_1,
                                       middle_control_points_2, key_points[1:]]).reshape(n, 4, -1)

    # Get Bernstein polynomial approximation of each curve
    binom_coefficients = [1, 3, 3, 1]
    for i in range(n):
        bezier_param_t = np.linspace(0, 1, num=resolution)
        current_segment = np.zeros((resolution, 2))
        for point_num, point in enumerate(curve_main_points_arr[i]):
            binom_coefficient = binom_coefficients[point_num]
            polynomial_degree = np.power(bezier_param_t, point_num)
            polynomial_degree *= np.power(1 - bezier_param_t, 3 - point_num)
            bernstein_polynomial = binom_coefficient * polynomial_degree
            current_segment += np.outer(bernstein_polynomial, point)
        curve_segments.extend(current_segment)

    curve_segments = np.array(curve_segments)
    figure_coordinates = np.unique(np.ceil(curve_segments).astype(int), axis=0)
    return figure_coordinates

def trinagular_kernel_1d(length, alpha=.1):
    """ Kernel-function that changes linearly from a center point to alpha on borders. """
    result = np.zeros(length)
    array = np.linspace(alpha, 2, length)
    result[:length // 2] = array[:length // 2]
    result[length // 2:] = 2 + alpha - array[length // 2:]
    return result

def triangular_weights_function_nd(array, alpha=.1):
    """ Weights-function given by a product of 1d triangular kernels. """
    result = 1
    for i, axis_len in enumerate(array.shape):
        if axis_len != 1:
            multiplier_shape = np.ones_like(array.shape)
            multiplier_shape[i] = axis_len
            result = result * trinagular_kernel_1d(axis_len, alpha).reshape(multiplier_shape)
    return result

def get_environ_flag(flag_name, defaults=('0', '1'), convert=int):
    """ Retrive environmental variable, check if it matches expected defaults and optionally convert it. """
    flag = os.environ.get(flag_name, '0')
    if flag not in defaults:
        raise ValueError(f"Expected `{flag_name}` env variable value to be from {defaults}, got {flag} instead.")
    return convert(flag)

def to_list(obj):
    """ Cast an object to a list.
    When default value provided, cast it instead if object value is None.
    Almost identical to `list(obj)` for 1-D objects, except for `str` instances,
    which won't be split into separate letters but transformed into a list of a single element.
    """
    return np.array(obj, dtype=object).ravel().tolist()

def get_class_methods(cls):
    """ Get a list of non-private class methods. """
    return [getattr(cls, func) for func in dir(cls) if not func.startswith("__") and callable(getattr(cls, func))]

def make_savepath(path, name, extension=''):
    """ If given replace asterisk in path with label name and create save dir if it does not already exist. """
    if path.endswith('*'):
        path = path.replace('*', f'{name}{extension}')

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return path

def faults_lengthes(slide, normalize):
    """ Compute sizes (depth length) for each connected object on 2D slide. """
    sizes = slide.copy()
    labels, n_objects = measurements.label(slide > 0, structure=np.ones((3, 3)))
    for i in range(1, n_objects+1):
        size = np.where(labels == i)[-1].ptp()
        if normalize:
            size /= slide.shape[-1]
        sizes[labels == i] = size
    return sizes

@njit
def concat_sorted(first_array, second_array):
    """ Merge two sorted arrays into sorted array. """
    buffer = np.zeros((len(first_array) + len(second_array), 3), dtype=np.int32)
    c = 0
    i, j = 0, 0

    # Condition is re-evaluated to swapped arrays
    while i < len(first_array) and j < len(second_array):
        first_ = first_array[i]
        second_ = second_array[j]

        # Need to swap or not
        if second_[0] == first_[0]:
            if second_[1] < first_[1]:
                first_array, second_array = second_array, first_array
                i, j = j, i
            elif second_[1] == first_[1]:
                # Same value: no need to duplicate
                j += 1

        elif second_[0] < first_[0]:
            first_array, second_array = second_array, first_array
            i, j = j, i

        buffer[c] = first_array[i]
        c += 1
        i += 1

    return buffer[:c]

def split_array(array, labels):
    """ Split (groupby) array by values from labels. Labels must be sorted and all groups must be contiguous. """
    return np.split(array, np.unique(labels, return_index=True)[1][1:])


@njit
def get_adjancency_matrix(arr, n):
    """ Produce an adjacency matrix for a labeled integer-valued array with the provided amount of unique elements. """
    res = np.zeros((n, n), dtype=np.int32)

    for i in range(0, arr.shape[0] - 1, 1):
        for j in range(0, arr.shape[1] - 1, 1):

            a = arr[i, j]
            b = arr[i, j + 1]
            c = arr[i + 1, j]

            if a != b:
                res[a, b] = 1
                res[b, a] = 1

            if c not in (a, b):
                res[a, c] = 1
                res[c, a] = 1

    return res
