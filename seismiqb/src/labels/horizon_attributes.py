""" Mixin with computed along horizon geological attributes. """
# pylint: disable=too-many-statements
import numpy as np

from cv2 import dilate
from scipy.signal import hilbert, ricker
from scipy.ndimage import convolve
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion, binary_dilation
from skimage.measure import label
from sklearn.decomposition import PCA

from ..functional import smooth_out, special_convolve
from ..utils import transformable, lru_cache, to_list



class AttributesMixin:
    """ Geological attributes along horizon:
    - scalars computed from its depth map only: number of holes, perimeter, coverage
    - matrices computed from its depth map only: presence matrix, gradients along directions, etc
    - properties of a carcass
    - methods to cut data from the cube along horizon
    - matrices derived from amplitudes along horizon: instant amplitudes/phases, decompositions, etc.

    Also changes the `__getattr__` of a horizon by allowing the `full_` prefix to apply `:meth:~.put_on_full`.
    For example, `full_binary_matrix` would return the result of `binary_matrix`, wrapped with `:meth:~.put_on_full`.

    Method for getting desired attributes is `load_attribute`. It works with nested keys, i.e. one can get attributes
    of horizon subsets. Address method documentation for further details.
    """
    #pylint: disable=unexpected-keyword-arg
    def __getattr__(self, key):
        if key.startswith('full_'):
            key = key.replace('full_', '')
            matrix = getattr(self, key)
            return self.matrix_put_on_full(matrix)
        raise AttributeError(key)

    # Modify computed matrices
    def _dtype_to_fill_value(self, dtype):
        if dtype == np.int32:
            fill_value = self.FILL_VALUE
        elif dtype == np.float32:
            fill_value = np.nan
        elif np.issubdtype(dtype, np.bool):
            fill_value = False
        else:
            raise TypeError(f'Incorrect dtype: `{dtype}`')
        return fill_value

    def matrix_set_dtype(self, matrix, dtype):
        """ Change the dtype and fill_value to match it. """
        mask = (matrix == self.FILL_VALUE) | np.isnan(matrix)
        matrix = matrix.astype(dtype)

        matrix[mask] = self._dtype_to_fill_value(dtype)
        return matrix

    def matrix_put_on_full(self, matrix):
        """ Convert matrix from being horizon-shaped to cube-shaped. """
        if matrix.shape[:2] != self.field.spatial_shape:
            background = np.full(self.field.spatial_shape, self._dtype_to_fill_value(matrix.dtype), dtype=matrix.dtype)
            background[self.i_min:self.i_max + 1, self.x_min:self.x_max + 1] = matrix
        else:
            background = matrix
        return background

    def matrix_fill_to_num(self, matrix, value):
        """ Change the matrix values at points where horizon is absent to a supplied one. """
        if matrix.shape == self.shape:
            mask = ~self.binary_matrix
        elif matrix.shape == self.field.spatial_shape:
            mask = ~self.full_binary_matrix
        else:
            msg = f"Can't define horizon absence mask with respect to provided matrix since its shape {matrix.shape} "\
                  f"doesn't coincide with either horizon shape {self.shape} or field shape {self.field.spatial_shape}."
            raise ValueError(msg)

        matrix[mask] = value

        return matrix

    def matrix_num_to_fill(self, matrix, value):
        """ Mark points equal to value as absent ones. """
        if value is np.nan:
            mask = np.isnan(matrix)
        else:
            mask = (matrix == value)

        matrix[mask] = self._dtype_to_fill_value(matrix.dtype)
        return matrix

    def matrix_normalize(self, matrix, mode):
        """ Normalize matrix values.

        Parameters
        ----------
        mode : bool, str, optional
            If `min-max` or True, then use min-max scaling.
            If `mean-std`, then use mean-std scaling.
            If False, don't scale matrix.
        """
        values = matrix[self.presence_matrix]

        if mode in ['min-max', True]:
            min_, max_ = np.nanmin(values), np.nanmax(values)
            matrix = (matrix - min_) / (max_ - min_)
        elif mode == 'mean-std':
            mean, std = np.nanmean(values), np.nanstd(values)
            matrix = (matrix - mean) / std
        else:
            raise ValueError(f'Unknown normalization mode `{mode}`.')
        return matrix


    def matrix_smooth_out(self, matrix, kernel=None, kernel_size=7, sigma=2., margin=5, iters=1):
        """ Smooth the depth matrix to produce floating point numbers. """
        smoothed = smooth_out(matrix, kernel=kernel, kernel_size=kernel_size, sigma=sigma,
                              margin=margin, fill_value=self.FILL_VALUE, preserve=True, iters=iters)
        return smoothed

    def matrix_enlarge(self, matrix, width=3):
        """ Increase visibility of a sparse carcass metric. Should be used only for visualization purposes. """
        if matrix.ndim == 3 and matrix.shape[-1] != 1:
            return matrix

        # Convert all the nans to a number, so that `dilate` can work with it
        matrix = matrix.copy().astype(np.float32).squeeze()
        matrix[np.isnan(matrix)] = self.FILL_VALUE

        # Apply dilations along both axis
        structure = np.ones((1, 3), dtype=np.uint8)
        dilated1 = dilate(matrix, structure, iterations=width)
        dilated2 = dilate(matrix, structure.T, iterations=width)

        # Mix matrices
        matrix = np.full_like(matrix, np.nan)
        matrix[dilated1 != self.FILL_VALUE] = dilated1[dilated1 != self.FILL_VALUE]
        matrix[dilated2 != self.FILL_VALUE] = dilated2[dilated2 != self.FILL_VALUE]

        mask = (dilated1 != self.FILL_VALUE) & (dilated2 != self.FILL_VALUE)
        matrix[mask] = (dilated1[mask] + dilated2[mask]) / 2

        # Fix zero traces
        matrix[np.isnan(self.field.std_matrix)] = np.nan
        return matrix

    @staticmethod
    def pca_transform(data, n_components=3, **kwargs):
        """ Reduce number of channels along the depth axis. """
        flattened = data.reshape(-1, data.shape[-1])
        mask = np.isnan(flattened).any(axis=-1)

        pca = PCA(n_components, **kwargs)
        transformed = pca.fit_transform(flattened[~mask])
        n_components = transformed.shape[-1]

        result = np.full((*data.shape[:2], n_components), np.nan).reshape(-1, n_components)
        result[~mask] = transformed
        result = result.reshape(*data.shape[:2], n_components)
        return result


    # Technical matrices
    @property
    def binary_matrix(self):
        """ Boolean matrix with `true` values at places where horizon is present and `false` everywhere else. """
        return (self.matrix > 0).astype(np.bool)

    @property
    def presence_matrix(self):
        """ A convenient alias for binary matrix in cubic coordinate system. """
        return self._presence_matrix()

    @lru_cache(maxsize=1)
    def _presence_matrix(self):
        """ A method for getting binary matrix in cubic coordinates. Allows for introspectable cache. """
        return self.full_binary_matrix


    # Scalars computed from depth map
    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        return len(self) / (np.prod(self.field.spatial_shape) - np.sum(self.field.zero_traces))

    @property
    def number_of_holes(self):
        """ Number of holes inside horizon borders. """
        holes_array = self.filled_matrix != self.binary_matrix
        _, num = label(holes_array, connectivity=2, return_num=True, background=0)
        return num

    @property
    def perimeter(self):
        """ Number of points in the borders. """
        return np.sum((self.borders_matrix == 1).astype(np.int32))

    @property
    def solidity(self):
        """ Ratio of area covered by horizon to total area inside borders. """
        return len(self) / np.sum(self.filled_matrix)


    # Matrices computed from depth map
    @property
    def borders_matrix(self):
        """ Borders of horizons (borders of holes inside are not included). """
        filled_matrix = self.filled_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(filled_matrix, structure, border_value=0)
        return filled_matrix ^ eroded # binary difference operation

    @property
    def boundaries_matrix(self):
        """ Borders of horizons (borders of holes inside included). """
        binary_matrix = self.binary_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(binary_matrix, structure, border_value=0)
        return binary_matrix ^ eroded # binary difference operation

    @property
    def filled_matrix(self):
        """ Binary matrix with filled holes. """
        structure = np.ones((3, 3))
        filled_matrix = binary_fill_holes(self.binary_matrix, structure)
        return filled_matrix


    def grad_along_axis(self, axis=0):
        """ Change of heights along specified direction. """
        grad = np.diff(self.matrix, axis=axis, prepend=np.int32(0))
        grad[np.abs(grad) > self.h_min] = self.FILL_VALUE
        grad[self.matrix == self.FILL_VALUE] = self.FILL_VALUE
        return grad

    @property
    def grad_i(self):
        """ Change of heights along iline direction. """
        return self.grad_along_axis(0)

    @property
    def grad_x(self):
        """ Change of heights along xline direction. """
        return self.grad_along_axis(1)


    # Carcass properties: should be used only if the horizon is a carcass
    @property
    def is_carcass(self):
        """ Check if the horizon is a sparse carcass. """
        return len(self) / self.filled_matrix.sum() < 0.5

    @property
    def carcass_ilines(self):
        """ Labeled inlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        return uniques[counts > 256]

    @property
    def carcass_xlines(self):
        """ Labeled xlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        return uniques[counts > 256]


    # Retrieve data from seismic along horizon
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_cube_values(self, window=1, offset=0, chunk_size=256, **_):
        """ Get values from the cube along the horizon.

        Parameters
        ----------
        window : int
            Width of data slice along the horizon.
        offset : int
            Offset of data slice with respect to horizon heights matrix.
        chunk_size : int
            Size of data along height axis processed at a time.
        """
        low = window // 2
        high = max(window - low, 0)
        chunk_size = min(chunk_size, self.h_max - self.h_min + window)

        background = np.zeros((self.field.ilines_len, self.field.xlines_len, window), dtype=np.float32)

        for h_start in range(max(low, self.h_min), self.h_max + 1, chunk_size):
            h_end = min(h_start + chunk_size, self.h_max + 1)

            # Get chunk from the cube (depth-wise)
            location = (slice(None), slice(None),
                        slice(h_start - low, min(h_end + high, self.field.depth)))
            data_chunk = self.field.geometry.load_crop(location, use_cache=False)

            # Check which points of the horizon are in the current chunk (and present)
            idx_i, idx_x = np.asarray((self.matrix != self.FILL_VALUE) &
                                      (self.matrix >= h_start) &
                                      (self.matrix < h_end)).nonzero()
            heights = self.matrix[idx_i, idx_x]

            # Convert spatial coordinates to cubic, convert height to current chunk local system
            idx_i += self.i_min
            idx_x += self.x_min
            heights -= (h_start - offset)

            # Subsequently add values from the cube to background, then shift horizon 1 unit lower
            for j in range(window):
                background[idx_i, idx_x, np.full_like(heights, j)] = data_chunk[idx_i, idx_x, heights]
                heights += 1
                mask = heights < data_chunk.shape[2]
                idx_i = idx_i[mask]
                idx_x = idx_x[mask]
                heights = heights[mask]

        background[~self.presence_matrix] = np.nan
        return background


    def get_array_values(self, array, shifts=None, grid_info=None, width=5, axes=(2, 1, 0)):
        """ Get values from an external array along the horizon.

        Parameters
        ----------
        array : np.ndarray
            A data-array to make a cut from.
        shifts : tuple or None
            an offset defining the location of given array with respect to the horizon.
            If None, `grid_info` with key `range` must be supplied.
        grid_info : dict
            Whenever passed, must contain key `range`.
            Used for infering shifts of the array with respect to horizon.
        width : int
            required width of the resulting cut.
        axes : tuple
            if not None, axes-transposition with the required axes-order is used.
        """
        if shifts is None and grid_info is None:
            raise ValueError('Either shifts or dataset with filled grid_info must be supplied!')

        if shifts is None:
            shifts = [grid_info['range'][i][0] for i in range(3)]

        shifts = np.array(shifts)
        horizon_shift = np.array((self.bbox[0, 0], self.bbox[1, 0]))

        if axes is not None:
            array = np.transpose(array, axes=axes)

        # compute start and end-points of the ilines-xlines overlap between
        # array and matrix in horizon and array-coordinates
        horizon_shift, shifts = np.array(horizon_shift), np.array(shifts)
        horizon_max = horizon_shift[:2] + np.array(self.matrix.shape)
        array_max = np.array(array.shape[:2]) + shifts[:2]
        overlap_shape = np.minimum(horizon_max[:2], array_max[:2]) - np.maximum(horizon_shift[:2], shifts[:2])
        overlap_start = np.maximum(0, horizon_shift[:2] - shifts[:2])
        heights_start = np.maximum(shifts[:2] - horizon_shift[:2], 0)

        # recompute horizon-matrix in array-coordinates
        slc_array = [slice(l, h) for l, h in zip(overlap_start, overlap_start + overlap_shape)]
        slc_horizon = [slice(l, h) for l, h in zip(heights_start, heights_start + overlap_shape)]
        overlap_matrix = np.full(array.shape[:2], fill_value=self.FILL_VALUE, dtype=np.float32)
        overlap_matrix[slc_array] = self.matrix[slc_horizon]
        overlap_matrix -= shifts[-1]

        # make the cut-array and fill it with array-data located on needed heights
        result = np.full(array.shape[:2] + (width, ), np.nan, dtype=np.float32)
        iterator = [overlap_matrix + shift for shift in range(-width // 2 + 1, width // 2 + 1)]

        for i, surface_level in enumerate(np.array(iterator)):
            mask = (surface_level >= 0) & (surface_level < array.shape[-1]) & (surface_level !=
                                                                               self.FILL_VALUE - shifts[-1])
            mask_where = np.where(mask)
            result[mask_where[0], mask_where[1], i] = array[mask_where[0], mask_where[1],
                                                            surface_level[mask_where].astype(np.int)]

        return result


    # Generic attributes loading
    ATTRIBUTE_TO_ALIAS = {
        # Properties
        'full_matrix': ['full_matrix', 'heights', 'depths'],
        'full_binary_matrix': ['full_binary_matrix', 'presence_matrix', 'masks'],

        # Created by `get_*` methods
        'amplitudes': ['amplitudes', 'cube_values'],
        'metric': ['metric', 'metrics'],
        'instant_phases': ['instant_phases', 'iphases'],
        'instant_amplitudes': ['instant_amplitudes', 'iamplitudes'],
        'fourier_decomposition': ['fourier', 'fourier_decomposition'],
        'wavelet_decomposition': ['wavelet', 'wavelet_decomposition'],
        'median_diff': ['median_diff', 'mdiff'],
        'grad': ['grad', 'gradient'],
        'spikes': ['spikes'],
    }
    ALIAS_TO_ATTRIBUTE = {alias: name for name, aliases in ATTRIBUTE_TO_ALIAS.items() for alias in aliases}

    ATTRIBUTE_TO_METHOD = {
        'amplitudes' : 'get_cube_values',
        'metric' : 'get_metric',
        'instant_phases' : 'get_instantaneous_phases',
        'instant_amplitudes' : 'get_instantaneous_amplitudes',
        'fourier_decomposition' : 'get_fourier_decomposition',
        'wavelet_decomposition' : 'get_wavelet_decomposition',
        'median_diff': 'get_median_diff_map',
        'grad': 'get_gradient_map',
        'spikes': 'get_spikes_map',
    }

    def load_attribute(self, src, location=None, use_cache=True, enlarge=False, **kwargs):
        """ Load horizon attribute values at requested location.
        This is the intended interface of loading matrices along the horizon, and should be preffered in all scenarios.

        To retrieve the attribute, we either use `:meth:~.get_property` or `:meth:~.get_*` methods: as all of them are
        wrapped with `:func:~.transformable` decorator, you can use its arguments to modify the behaviour.

        Parameters
        ----------
        src : str
            Key of the desired attribute. Valid attributes are either properties or aliases, defined
            by `ALIAS_TO_ATTRIBUTE` mapping, for example:

            - 'cube_values' or 'amplitudes': cube values;
            - 'depths' or 'full_matrix': horizon depth map in cubic coordinates;
            - 'metrics': random support metrics matrix.
            - 'instant_phases': instantaneous phase;
            - 'instant_amplitudes': instantaneous amplitude;
            - 'fourier' or 'fourier_decomposition': fourier transform with optional PCA;
            - 'wavelet' or 'wavelet decomposition': wavelet transform with optional PCA;
            - 'masks' or 'full_binary_matrix': mask of horizon;
        location : sequence of 3 slices
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is not used, since points are sampled exactly on horizon.
            If None, `src` is returned uncropped.
        enlarge : bool, optional
            Whether to enlarge carcass maps. Defaults to True, if the horizon is a carcass, False otherwise.
            Should be used only for visualization purposes.
        kwargs :
            Passed directly to attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src`.

        Examples
        --------
        Load 'depths' attribute for whole horizon:
        >>> horizon.load_attribute('depths')

        Load 'cube_values' attribute for requested slice of fixed width:
        >>> horizon.load_attribute('cube_values', (x_slice, i_slice, 1), window=10)

        Load 'metrics' attribute with specific evaluation parameter and following normalization.
        >>> horizon.load_attribute('metrics', metric='local_corrs', normalize='min-max')
        """
        src = self.ALIAS_TO_ATTRIBUTE.get(src, src)
        enlarge = enlarge and self.is_carcass

        if src in self.ATTRIBUTE_TO_METHOD:
            method = self.ATTRIBUTE_TO_METHOD[src]
            data = getattr(self, method)(use_cache=use_cache, enlarge=enlarge, **kwargs)
        else:
            data = self.get_property(src, enlarge=enlarge, **kwargs)

        # TODO: Someday, we would need to re-write attribute loading methods
        # so they use locations not to crop the loaded result, but to load attribute only at location.
        if location is not None:
            i_slice, x_slice, _ = location
            data = data[i_slice, x_slice]
        return data


    # Specific attributes loading
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_property(self, src, **_):
        """ Load a desired instance attribute. Decorated to allow additional postprocessing steps. """
        data = getattr(self, src, None)
        if data is None:
            aliases = list(self.ALIAS_TO_ATTRIBUTE.keys())
            raise ValueError(f'Unknown `src` {src}. Expected a matrix-property or one of {aliases}.')
        return data

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_instantaneous_amplitudes(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous amplitude along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values`.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_amplitudes(depths=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs)
        result = np.abs(hilbert(amplitudes)).astype(np.float32)[:, :, depths]
        return result

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_instantaneous_phases(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous phase along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values`.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_phases(depths=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs)
        result = np.angle(hilbert(amplitudes)).astype(np.float32)[:, :, depths]
        return result

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_metric(self, metric='support_corrs', supports=50, agg='nanmean', **kwargs):
        """ Cached metrics calcucaltion with disabled plotting option.

        Parameters
        ----------
        metric, supports, agg :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        kwargs :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        """
        metrics = self.metrics.evaluate(metric=metric, supports=supports, agg=agg,
                                        enlarge=False, plot=False, savepath=None, **kwargs)
        return metrics


    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_fourier_decomposition(self, window=50, **_):
        """ Cached fourier transform calculation follower by dimensionaluty reduction via PCA.

        Parameters
        ----------
        window : int
            Width of amplitudes slice to calculate fourier transform on.
        """
        amplitudes = self.load_attribute('amplitudes', window=window)
        result = np.abs(np.fft.rfft(amplitudes))
        return result

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_wavelet_decomposition(self, widths=range(1, 14, 3), window=50, **_):
        """ Cached wavelet transform calculation followed by dimensionaluty reduction via PCA.

        Parameters
        ----------
        widths : list of numbers
            Widths of wavelets to calculate decomposition for.
        window : int
            Width of amplitudes slice to calculate wavelet transform on.
        """
        amplitudes = self.load_attribute('amplitudes', window=window)
        result_shape = *amplitudes.shape[:2], len(widths)
        result = np.empty(result_shape, dtype=np.float32)
        for idx, width in enumerate(widths):
            wavelet = ricker(window, width).reshape(1, 1, -1)
            result[:, :, idx] = convolve(amplitudes, wavelet, mode='constant')[:, :, window // 2]

        return result


    def get_zerocrossings(self, side, window=15):
        """ Get matrix of depths shifted to nearest point of sign change in cube values.

        Parameters
        ----------
        side : -1 or 1
            Whether to look for sign change above the horizon (-1) or below (1).
        window : positive int
            Width of data slice above/below the horizon made along its surface.
        """
        values = self.get_cube_values(window=window, offset=window // 2 * side, fill_value=0)
        # reverse array along depth axis for invariance
        values = values[:, :, ::side]

        sign = np.sign(values)
        # value 2 in the array below mark cube values sign change along depth axis
        cross = np.abs(np.diff(sign, axis=-1))

        # put 2 at points, where cube values are precisely equal to zero
        zeros = sign[:, :, :-1] == 0
        cross[zeros] = 2

        # obtain indices of first sign change occurences for every trace
        # if trace doesn't change sign, corresponding index of sign change is 0
        cross_indices = np.argmax(cross == 2, axis=-1)

        # get cube values before sign change
        start_points = self.matrix_to_points(cross_indices).T
        start_values = values[tuple(start_points)]

        # get cube values after sign change
        stop_points = start_points + np.array([[0], [0], [1]])
        stop_values = values[tuple(stop_points)]

        # calculate additional float shifts towards true zero-crossing point
        float_shift = start_values - stop_values
        # do not perform division at points, where both 'start' and 'stop' values are 0
        np.divide(start_values, float_shift, out=float_shift, where=float_shift != 0)

        # treat obtained indices as shifts for label depths matrix
        shift = cross_indices.astype(np.float32)
        # apply additional float shifts to shift matrix
        shift += float_shift.reshape(shift.shape)
        # account for shift matrix sign change
        shift *= side

        result = self.full_matrix + shift
        return result


    # Despiking maps
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_median_diff_map(self, convolve_mode='m', kernel_size=11, kernel=None, margin=0, iters=2, threshold=2, **_):
        """ Compute difference between depth map and its median filtered counterpart. """
        convolved = special_convolve(self.full_matrix, mode=convolve_mode, kernel=kernel, kernel_size=kernel_size,
                                     margin=margin, iters=iters, fill_value=self.FILL_VALUE)
        spikes = self.full_matrix - convolved

        if threshold is not None:
            spikes[np.abs(spikes) < threshold] = 0
        spikes[self.field.zero_traces == 1] = np.nan
        return spikes

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_gradient_map(self, threshold=0, **_):
        """ Compute combined gradient map along both directions.

        Parameters
        ----------
        threshold : number
            Threshold to consider a difference to be a spike.
        """
        grad_i = self.load_attribute('grad_i', on_full=True, dtype=np.float32, use_cache=False)
        grad_x = self.load_attribute('grad_x', on_full=True, dtype=np.float32, use_cache=False)

        grad_i[np.abs(grad_i) <= threshold] = 0
        grad_x[np.abs(grad_x) <= threshold] = 0

        grad = grad_i + grad_x
        grad[self.field.zero_traces == 1] = np.nan
        return grad


    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    @transformable
    def get_spikes_map(self, spikes_mode='median', threshold=1., dilation=5,
                       kernel_size=11, kernel=None, margin=0, iters=2, **_):
        """ Locate spikes on a horizon.

        Parameters
        ----------
        mode : str
            If 'gradient', then use gradient map to locate spikes.
            If 'median', then use median diffs to locate spikes.
        threshold : number
            Threshold to consider a difference to be a spike.
        dilation : int
            Number of iterations for binary dilation algorithm to increase the spikes.
        kernel_size, kernel, margin, iters
            Parameters for median differences computation.
        """
        if spikes_mode.startswith('m'):
            spikes = self.load_attribute('median_diff', mode='m', kernel=kernel, kernel_size=kernel_size,
                                         margin=margin, iters=iters, threshold=threshold)
        elif spikes_mode.startswith('g'):
            spikes = self.load_attribute('gradient', threshold=threshold)
        else:
            raise ValueError(f'Wrong mode passed, {spikes_mode}')

        if dilation:
            spikes = np.nan_to_num(spikes)
            spikes = binary_dilation(spikes, iterations=dilation).astype(np.float32)
            spikes[self.field.zero_traces == 1] = np.nan
        return spikes

    # Clustering
    @transformable
    def cluster(self, clusterer, attributes, scales=None, add_coordinates=True, **kwargs):
        """ !!. """
        attributes = to_list(attributes) if isinstance(attributes, dict) else [{'src': attributes}]
        scales = [1] * len(attributes) if scales is None else to_list(scales)

        i, x = self.points.T[:2]
        features = [self.load_attribute(dtype=np.float32, **attribute)[i, x] * scale
                    for attribute, scale in zip(attributes, scales)]
        if add_coordinates:
            features += [self.points.astype(np.float32)]
        features = np.concatenate(features, axis=1)

        cluster_labels = clusterer.fit_predict(features, **kwargs)

        cluster_map = np.full(self.field.spatial_shape, np.nan)
        cluster_map[i, x] = cluster_labels

        return cluster_map
