""" Horizon class for POST-STACK data. """
import os
from copy import copy
from textwrap import dedent
from functools import partialmethod

import numpy as np

from skimage.measure import label
from scipy.ndimage import find_objects
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from .horizon_attributes import AttributesMixin
from .horizon_extraction import ExtractionMixin
from .horizon_visualization import VisualizationMixin
from ..utils import CacheMixin, CharismaMixin
from ..utils import groupby_mean, groupby_min, groupby_max, filtering_function
from ..utils import make_bezier_figure
from ..utils import MetaDict
from ..functional import smooth_out



class Horizon(AttributesMixin, CacheMixin, CharismaMixin, ExtractionMixin, VisualizationMixin):
    """ Contains spatially-structured horizon: each point describes a height on a particular (iline, xline).

    Initialized from `storage` and `geometry`, where storage can be one of:
        - csv-like file in CHARISMA or REDUCED_CHARISMA format.
        - ndarray of (N, 3) shape.
        - ndarray of (ilines_len, xlines_len) shape.
        - dictionary: a mapping from (iline, xline) -> height.
        - mask: ndarray of (ilines_len, xlines_len, depth) with 1's at places of horizon location.

    Main storages are `matrix` and `points` attributes:
        - `matrix` is a depth map, ndarray of (ilines_len, xlines_len) shape with each point
          corresponding to horizon height at this point. Note that shape of the matrix is generally smaller
          than cube spatial range: that allows to save space.
          Attributes `i_min` and `x_min` describe position of the matrix in relation to the cube spatial range.
          Each point with absent horizon is filled with `FILL_VALUE`.
          Note that since the dtype of `matrix` is `np.int32`, we can't use `np.nan` as the fill value.
          In order to initialize from this storage, one must supply `matrix`, `i_min`, `x_min`.

        - `points` is a (N, 3) ndarray with every row being (iline, xline, height). Note that (iline, xline) are
          stored in cube coordinates that range from 0 to `ilines_len` and 0 to `xlines_len` respectively.
          Stored height is corrected on `time_delay` and `sample_rate` of the cube.
          In order to initialize from this storage, one must supply (N, 3) ndarray.

    Depending on which attribute was created at initialization (`matrix` or `points`), the other is computed lazily
    at the time of the first access. This way, we can greatly amortize computations when dealing with huge number of
    `Horizon` instances, i.e. when extracting surfaces from predicted masks.

    Independently of type of initial storage, Horizon provides following:
        - Attributes `i_min`, `x_min`, `i_max`, `x_max`, `h_min`, `h_max`, `h_mean`, `h_std`, `bbox`,
          to completely describe location of the horizon in the 3D volume of the seismic cube.

        - Convenient methods of changing the horizon, `apply_to_matrix` and `apply_to_points`:
          these methods must be used instead of manually permuting `matrix` and `points` attributes.
          For example, filtration or smoothing of a horizon can be done with their help.

        - Method `add_to_mask` puts 1's on the `location` of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - `evaluate` allows to quickly assess the quality of a seismic reflection;
          for more metrics, check :class:`~.HorizonMetrics`.

        - A number of properties that describe geometrical, geological and mathematical characteristics of a horizon.
          For example, `borders_matrix` and `boundaries_matrix`: the latter containes outer and inner borders;
          `coverage` is the ratio between labeled traces and non-zero traces in the seismic cube;
          `solidity` is the ratio between labeled traces and traces inside the hull of the horizon;
          `perimeter` and `number_of_holes` speak for themselves.

        - Multiple instances of Horizon can be compared against one another and, if needed,
          merged into one (either in-place or not) via `check_proximity`, `overlap_merge`, `adjacent_merge` methods.
          These methods are highly optimized in their accesses to inner attributes that are computed lazily.

        - A wealth of visualization methods: view from above, slices along iline/xline axis, etc.
    """
    #pylint: disable=too-many-public-methods, import-outside-toplevel, redefined-builtin

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999


    def __init__(self, storage, field, name=None, dtype=np.int32, force_format=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.dtype = dtype
        self.format = None

        # Location of the horizon inside cube spatial range
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.bbox = None
        self._len = None

        # Underlying data storages
        self._matrix = None
        self._points = None
        self._depths = None

        # Heights information
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None

        # Field reference
        self.field = field

        # Check format of storage, then use it to populate attributes
        if force_format is not None:
            self.format = force_format

        elif isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, dict):
            # mapping from (iline, xline) to (height)
            self.format = 'dict'

        elif isinstance(storage, np.ndarray):
            if storage.ndim == 2 and storage.shape[1] == 3:
                # array with row in (iline, xline, height) format
                self.format = 'points'

            elif storage.ndim == 2 and (storage.shape == self.field.spatial_shape):
                # matrix of (iline, xline) shape with every value being height
                self.format = 'full_matrix'

            elif storage.ndim == 2:
                # matrix of (iline, xline) shape with every value being height
                self.format = 'matrix'

        getattr(self, f'from_{self.format}')(storage, **kwargs)


    # Logic of lazy computation of `points` or `matrix` from the other available storage; cache management
    @property
    def points(self):
        """ Storage of horizon data as (N, 3) array of (iline, xline, height) in cubic coordinates.
        If the horizon is created not from (N, 3) array, evaluated at the time of the first access.
        """
        if self._points is None and self.matrix is not None:
            points = self.matrix_to_points(self.matrix).astype(self.dtype)
            points += np.array([self.i_min, self.x_min, 0], dtype=self.dtype)
            self._points = points
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.nonzero(matrix != Horizon.FILL_VALUE)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points


    @property
    def matrix(self):
        """ Storage of horizon data as depth map: matrix of (ilines_length, xlines_length) with each point
        corresponding to height. Matrix is shifted to a (i_min, x_min) point so it takes less space.
        If the horizon is created not from matrix, evaluated at the time of the first access.
        """
        if self._matrix is None and self.points is not None:
            self._matrix = self.points_to_matrix(points=self.points, i_min=self.i_min, x_min=self.x_min,
                                                 i_length=self.i_length, x_length=self.x_length, dtype=self.dtype)
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @staticmethod
    def points_to_matrix(points, i_min, x_min, i_length, x_length, dtype=np.int32):
        """ Convert array of (N, 3) shape to a depth map (matrix). """
        matrix = np.full((i_length, x_length), Horizon.FILL_VALUE, dtype)

        matrix[points[:, 0].astype(np.int32) - i_min,
               points[:, 1].astype(np.int32) - x_min] = points[:, 2]

        return matrix

    @property
    def depths(self):
        """ Array of depth only. Useful for faster stats computation when initialized from a matrix. """
        if self._depths is None:
            if self._points is not None:
                self._depths = self.points[:, -1]
            else:
                self._depths = self.matrix[self.matrix != self.FILL_VALUE]
        return self._depths


    def reset_storage(self, storage=None):
        """ Reset storage along with depth-wise lazy computed stats. """
        self._depths = None
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None
        self._len = None

        if storage == 'matrix':
            self._matrix = None
            if len(self.points) > 0:
                self._h_min = self.points[:, 2].min().astype(self.dtype)
                self._h_max = self.points[:, 2].max().astype(self.dtype)

                i_min, x_min, _ = np.min(self.points, axis=0)
                i_max, x_max, _ = np.max(self.points, axis=0)
                self.i_min, self.i_max, self.x_min, self.x_max = int(i_min), int(i_max), int(x_min), int(x_max)

                self.i_length = (self.i_max - self.i_min) + 1
                self.x_length = (self.x_max - self.x_min) + 1
                self.bbox = np.array([[self.i_min, self.i_max],
                                    [self.x_min, self.x_max],
                                    [self.h_min, self.h_max]],
                                    dtype=np.int32)
        elif storage == 'points':
            self._points = None

    def __copy__(self):
        """ Create a new horizon with the same data.

        Returns
        -------
        A horizon object with new matrix object and a reference to the same field.
        """
        return type(self)(storage=np.copy(self.matrix), field=self.field, i_min=self.i_min, x_min=self.x_min,
                          name=f"copy_of_{self.name}")

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Operands types do not match. Got {type(self)} and {type(other)}.")

        presence = other.full_binary_matrix
        discrepancies = self.full_matrix[presence] != other.full_matrix[presence]
        if discrepancies.any():
            raise ValueError("Horizons have different depths where present.")

        res_matrix = self.full_matrix.copy()
        res_matrix[presence] = self.FILL_VALUE
        name = f"~{other.name}"
        result = type(self)(storage=res_matrix, field=self.field, name=name)

        return result


    # Properties, computed from lazy evaluated attributes
    @property
    def h_min(self):
        """ Minimum depth value. """
        if self._h_min is None:
            self._h_min = np.min(self.depths)
        return self._h_min

    @property
    def h_max(self):
        """ Maximum depth value. """
        if self._h_max is None:
            self._h_max = np.max(self.depths)
        return self._h_max

    @property
    def h_mean(self):
        """ Average depth value. """
        if self._h_mean is None:
            self._h_mean = np.mean(self.depths)
        return self._h_mean

    @property
    def h_std(self):
        """ Std of depths. """
        if self._h_std is None:
            self._h_std = np.std(self.depths)
        return self._h_std

    def __len__(self):
        """ Number of labeled traces. """
        if self._len is None:
            if self._points is not None:
                self._len = len(self.points)
            else:
                self._len = len(self.depths)
        return self._len


    # Initialization from different containers
    def from_points(self, points, verify=True, dst='points', reset='matrix', **kwargs):
        """ Base initialization: from point cloud array of (N, 3) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: two spatial coordinates and depth.
        verify : bool
            Whether to remove points outside of the cube range.
        dst : str
            Attribute to save result.
        reset : str or None
            Storage to reset.
        """
        _ = kwargs

        if verify:
            idx = np.where((points[:, 0] >= 0) &
                           (points[:, 1] >= 0) &
                           (points[:, 2] >= 0) &
                           (points[:, 0] < self.field.shape[0]) &
                           (points[:, 1] < self.field.shape[1]) &
                           (points[:, 2] < self.field.shape[2]))[0]
            points = points[idx]

        if self.dtype == np.int32:
            points = np.rint(points)
        setattr(self, dst, points.astype(self.dtype))

        # Collect stats on separate axes. Note that depth stats are properties
        if reset:
            self.reset_storage(reset)


    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file. """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path) if self.name is None else self.name

        points = self.load_charisma(path=path, dtype=np.int32, format='points',
                                    fill_value=Horizon.FILL_VALUE, transform=transform,
                                    verify=True)

        self.from_points(points, verify=False, **kwargs)


    def from_matrix(self, matrix, i_min, x_min, h_min=None, h_max=None, length=None, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        if matrix.dtype != self.dtype:
            if np.issubdtype(self.dtype, np.integer):
                matrix = np.rint(matrix)
            matrix = matrix.astype(self.dtype)
        self.matrix = matrix

        self.i_min, self.x_min = i_min, x_min
        self.i_max, self.x_max = i_min + matrix.shape[0] - 1, x_min + matrix.shape[1] - 1

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.reset_storage('points')

        # Populate lazy properties with supplied values
        self._h_min, self._h_max, self._len = h_min, h_max, length
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max],
                              [self.h_min, self.h_max]],
                             dtype=np.int32)


    def from_full_matrix(self, matrix, **kwargs):
        """ Init from matrix that covers the whole cube. """
        kwargs = {
            'i_min': 0,
            'x_min': 0,
            **kwargs
        }
        self.from_matrix(matrix, **kwargs)


    def from_dict(self, dictionary, transform=True, **kwargs):
        """ Init from mapping from (iline, xline) to depths. """
        _ = kwargs

        points = self.dict_to_points(dictionary)

        if transform:
            points = self.field.lines_to_cubic(points)

        self.from_points(points)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    @staticmethod
    def from_mask(mask, field=None, origin=None, connectivity=3,
                  mode='mean', threshold=0.5, minsize=0, prefix='predict', **kwargs):
        """ Convert mask to a list of horizons.
        Returned list is sorted by length of horizons.

        Parameters
        ----------
        field : Field
            Horizon parent field.
        origin : sequence
            The upper left coordinate of a `mask` in the cube coordinates.
        threshold : float
            Parameter of mask-thresholding.
        mode : str
            Method used for finding the point of a horizon for each iline, xline.
        minsize : int
            Minimum length of a horizon to be extracted.
        prefix : str
            Name of horizon to use.
        """
        _ = kwargs
        if mode in ['mean', 'avg']:
            group_function = groupby_mean
        elif mode in ['min']:
            group_function = groupby_min
        elif mode in ['max']:
            group_function = groupby_max

        # Labeled connected regions with an integer
        labeled = label(mask >= threshold, connectivity=connectivity)
        objects = find_objects(labeled)

        # Create an instance of Horizon for each separate region
        horizons = []
        for i, sl in enumerate(objects):
            max_possible_length = 1
            for j in range(3):
                max_possible_length *= sl[j].stop - sl[j].start

            if max_possible_length >= minsize:
                indices = np.nonzero(labeled[sl] == i + 1)

                if len(indices[0]) >= minsize:
                    coords = np.vstack([indices[i] + sl[i].start for i in range(3)]).T

                    points = group_function(coords) + origin
                    horizons.append(Horizon(storage=points, field=field, name=f'{prefix}_{i}'))

        horizons.sort(key=len)
        horizons = [horizon for horizon in horizons if len(horizon) != 0]
        return horizons

    def from_subset(self, matrix, name=None):
        """ Make new label with points matrix filtered by given presense matrix.

        Parameters
        ----------
        matrix : np.array
            Presense matrix of labels points. Must be in full cubes coordinates.
            If consists of 0 and 1, keep points only where values are 1.
            If consists of values from [0, 1] interval, keep points where values are greater than 0.5.
        name : str or None
            Name for new label. If None, original label name used.

        Returns
        -------
        New `Horizon` instance with filtered points matrix.
        """
        result = copy(self)
        result.name = name or self.name

        filtering_matrix = (matrix < 0.5).astype(int)
        result.filter_matrix(filtering_matrix)

        return result

    # Basic properties
    @property
    def shape(self):
        """ Tuple of horizon dimensions."""
        return (self.i_length, self.x_length)

    @property
    def size(self):
        """ Number of elements in the full horizon matrix."""
        return self.i_length * self.x_length

    @property
    def short_name(self):
        """ Name without extension. """
        return self.name.split('.')[0]

    @property
    def displayed_name(self):
        """ Alias for `short_name`. """
        return self.short_name


    # Functions to use to change the horizon
    def apply_to_matrix(self, function, **kwargs):
        """ Apply passed function to matrix storage.
        Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to matrix storage directly.
            Can return either new_matrix, new_i_min, new_x_min or new_matrix only.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        result = function(self.matrix, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            matrix, i_min, x_min = result
        else:
            matrix, i_min, x_min = result, self.i_min, self.x_min
        self.matrix, self.i_min, self.x_min = matrix, i_min, x_min

        self.reset_storage('points') # applied to matrix, so we need to re-create points
        self.reset_cache()

    def apply_to_points(self, function, **kwargs):
        """ Apply passed function to points storage.
        Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to points storage directly.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        self.points = function(self.points, **kwargs)
        self.reset_storage('matrix') # applied to points, so we need to re-create matrix
        self.reset_cache()


    def filter_points(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from points storage. """
        if filtering_matrix is None:
            filtering_matrix = self.field.zero_traces

        def _filtering_function(points, **kwds):
            _ = kwds
            return filtering_function(points, filtering_matrix)

        self.apply_to_points(_filtering_function, **kwargs)

    def filter_matrix(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from matrix storage. """
        if filtering_matrix is None:
            filtering_matrix = self.field.zero_traces

        idx_i, idx_x = np.asarray(filtering_matrix[self.i_min:self.i_max + 1,
                                                   self.x_min:self.x_max + 1] == 1).nonzero()

        def _filtering_function(matrix, **kwds):
            _ = kwds
            matrix[idx_i, idx_x] = self.FILL_VALUE
            return matrix

        self.apply_to_matrix(_filtering_function, **kwargs)

    filter = filter_points

    def filter_spikes(self, mode='gradient', threshold=1., dilation=5, kernel_size=11, kernel=None, margin=0, iters=2):
        """ Remove spikes from horizon. Works inplace.

        Parameters
        ----------
        mode : str
            If 'gradient', then use gradient map to locate spikes.
            If 'median', then use median diffs to locate spikes.
        threshold : number
            Threshold to consider a difference to be a spike,
        dilation : int
            Number of iterations for binary dilation algorithm to increase the spikes.
        kernel_size, kernel, margin, iters
            Parameters for median differences computation.
        """
        spikes = self.load_attribute('spikes', spikes_mode=mode, threshold=threshold, dilation=dilation,
                                     kernel_size=kernel_size, kernel=kernel, margin=margin, iters=iters)
        self.filter(spikes)

    despike = filter_spikes

    def filter_disconnected_regions(self):
        """ Remove regions, not connected to the largest component of a horizon. """
        labeled = label(self.full_binary_matrix)
        values, counts = np.unique(labeled, return_counts=True)
        counts = counts[values != 0]
        values = values[values != 0]

        object_id = values[np.argmax(counts)]

        filtering_matrix = self.full_binary_matrix.copy()
        filtering_matrix[labeled == object_id] = 0
        self.filter(filtering_matrix)


    # Pre-defined transforms of a horizon
    def thin_out(self, factor=1, threshold=256):
        """ Thin out the horizon by keeping only each `factor`-th line.

        Parameters
        ----------
        factor : integer or sequence of two integers
            Frequency of lines to keep along ilines and xlines direction.
        threshold : integer
            Minimal amount of points in a line to keep.
        """
        if isinstance(factor, int):
            factor = (factor, factor)

        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        mask_i = np.isin(self.points[:, 0], uniques[counts > threshold][::factor[0]])

        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        mask_x = np.isin(self.points[:, 1], uniques[counts > threshold][::factor[1]])

        self.points = self.points[mask_i + mask_x]
        self.reset_storage('matrix')

    def smooth_out(self, kernel=None, kernel_size=3, sigma=0.8, iters=1, preserve_borders=True, margin=5, **kwargs):
        """ Convolve the horizon with gaussian kernel with special treatment to absent points:
        if the point was present in the original horizon, then it is changed to a weighted sum of all
        present points nearby;
        if the point was absent in the original horizon and there is at least one non-fill point nearby,
        then it is changed to a weighted sum of all present points nearby.

        Parameters
        ----------
        kernel : ndarray or None
            If passed, then ready-to-use kernel. Otherwise, gaussian kernel will be created.
        kernel_size : int
            Size of gaussian filter.
        sigma : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        iters : int
            Number of times to apply smoothing filter.
        preserve_borders : bool
            Whether or not to allow method label additional points.
        """
        def smoothing_function(matrix):
            smoothed = smooth_out(matrix, kernel=kernel,
                                  kernel_size=kernel_size, sigma=sigma, margin=margin,
                                  fill_value=self.FILL_VALUE, preserve=preserve_borders, iters=iters)
            smoothed = np.rint(smoothed).astype(np.int32)
            smoothed[self.field.zero_traces[self.i_min:self.i_max + 1,
                                            self.x_min:self.x_max + 1] == 1] = self.FILL_VALUE
            return smoothed

        self.apply_to_matrix(smoothing_function, **kwargs)

    interpolate = partialmethod(smooth_out, preserve_borders=False)


    def make_carcass(self, frequencies=100, regular=True, margin=50, apply_smoothing=False, **kwargs):
        """ Cut carcass out of a horizon. Returns a new instance.

        Parameters
        ----------
        frequencies : int or sequence of ints
            Frequencies of carcass lines.
        regular : bool
            Whether to make regular lines or base lines on geometry quality map.
        margin : int
            Margin from geometry edges to exclude from carcass.
        apply_smoothing : bool
            Whether to smooth out the result.
        kwargs : dict
            Other parameters for grid creation, see `:meth:~.SeismicGeometry.make_quality_grid`.
        """
        frequencies = frequencies if isinstance(frequencies, (tuple, list)) else [frequencies]
        carcass = copy(self)
        carcass.name = carcass.name.replace('copy', 'carcass')

        if regular:
            from ..metrics import GeometryMetrics
            gm = GeometryMetrics(self.field.geometry)
            grid = gm.make_grid(1 - self.field.zero_traces, frequencies=frequencies, margin=margin, **kwargs)
        else:
            grid = self.field.geometry.make_quality_grid(frequencies, margin=margin, **kwargs)

        carcass.filter(filtering_matrix=1-grid)
        if apply_smoothing:
            carcass.smooth_out(preserve_borders=False)
        return carcass

    def make_random_holes_matrix(self, n=10, scale=1.0, max_scale=.25,
                                 max_angles_amount=4, max_sharpness=5.0, locations=None,
                                 points_proportion=1e-5, points_shape=1,
                                 noise_level=0, seed=None):
        """ Create matrix of random holes for horizon.

        Holes can be bezier-like figures or points-like.
        We can control bezier-like and points-like holes amount by `n` and `points_proportion` parameters respectively.
        We also do some noise amplifying with `noise_level` parameter.

        Parameters
        ----------
        n : int
            Amount of bezier-like holes on horizon.
        points_proportion : float
            Proportion of point-like holes on the horizon. A number between 0 and 1.
        points_shape : int or sequence of int
            Shape of point-like holes.
        noise_level : int
            Radius of noise scattering near the borders of holes.
        scale : float or sequence of float
            If float, each bezier-like hole will have a random scale from exponential distribution with parameter scale.
            If sequence, each bezier-like hole will have a provided scale.
        max_scale : float
            Maximum bezier-like hole scale.
        max_angles_amount : int
            Maximum amount of angles in each bezier-like hole.
        max_sharpness : float
            Maximum value of bezier-like holes sharpness.
        locations : ndarray
            If provided, an array of desired locations of bezier-like holes.
        seed : int, optional
            Seed the random numbers generator.
        """
        rng = np.random.default_rng(seed)
        filtering_matrix = np.zeros_like(self.full_matrix)

        # Generate bezier-like holes
        if isinstance(scale, float):
            scales = []
            sampling_scale = int(
                np.ceil(1.0 / (1 - np.exp(-scale * max_scale)))
            ) # inverse probability of scales < max_scales
            while len(scales) < n:
                new_scales = rng.exponential(scale, size=sampling_scale*(n - len(scales)))
                new_scales = new_scales[new_scales <= max_scale]
                scales.extend(new_scales)
            scales = scales[:n]
        else:
            scales = scale

        if locations is None:
            idxs = rng.choice(len(self), size=n)
            locations = self.points[idxs, :2]

        coordinates = [] # container for all types of holes, represented by their coordinates
        for location, figure_scale in zip(locations, scales):
            n_key_points = rng.integers(2, max_angles_amount + 1)
            radius = rng.random()
            sharpness = rng.random() * rng.integers(1, max_sharpness)

            figure_coordinates = make_bezier_figure(n=n_key_points, radius=radius, sharpness=sharpness,
                                                    scale=figure_scale, shape=self.shape, seed=seed)
            figure_coordinates += location

            negative_coords_shift = np.min(np.vstack([figure_coordinates, [0, 0]]), axis=0)
            huge_coords_shift = np.max(np.vstack([figure_coordinates - self.shape, [0, 0]]), axis=0)
            figure_coordinates -= (huge_coords_shift + negative_coords_shift + 1)

            coordinates.append(figure_coordinates)

        # Generate points-like holes
        if points_proportion:
            points_n = int(points_proportion * len(self))
            idxs = rng.choice(len(self), size=points_n)
            locations = self.points[idxs, :2]

            filtering_matrix[locations[:, 0], locations[:, 1]] = 1
            if isinstance(points_shape, int):
                points_shape = (points_shape, points_shape)
            filtering_matrix = binary_dilation(filtering_matrix, np.ones(points_shape))
            coordinates.append(np.argwhere(filtering_matrix > 0))
        coordinates = np.concatenate(coordinates)

        # Add noise and filtering matrix transformations
        if noise_level:
            noise = rng.normal(loc=coordinates,
                               scale=noise_level,
                               size=coordinates.shape)
            coordinates = np.unique(np.vstack([coordinates, noise.astype(int)]), axis=0)

        idx = np.where((coordinates[:, 0] >= 0) &
                       (coordinates[:, 1] >= 0) &
                       (coordinates[:, 0] < self.i_length) &
                       (coordinates[:, 1] < self.x_length))[0]
        coordinates = coordinates[idx]

        filtering_matrix[coordinates[:, 0], coordinates[:, 1]] = 1
        filtering_matrix = binary_fill_holes(filtering_matrix)
        filtering_matrix = binary_dilation(filtering_matrix, iterations=4)
        return filtering_matrix


    # Horizon usage: mask generation
    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : ndarray
            Where the mask is located.
        width : int
            Width of an added horizon.
        alpha : number
            Value to fill background with at horizon location.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        #TODO: add clear explanation about usage of advanced index in Horizon
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)

        if i_max > i_min and x_max > x_min:
            overlap = self.matrix[i_min - self.i_min : i_max - self.i_min,
                                  x_min - self.x_min : x_max - self.x_min]

            # Coordinates of points to use in overlap local system
            idx_i, idx_x = np.asarray((overlap != self.FILL_VALUE) &
                                      (overlap >= mask_h_min + low) &
                                      (overlap <= mask_h_max - high)).nonzero()
            heights = overlap[idx_i, idx_x]

            # Convert coordinates to mask local system
            idx_i += i_min - mask_i_min
            idx_x += x_min - mask_x_min
            heights -= (mask_h_min + low)

            for shift in range(width):
                mask[idx_i, idx_x, heights + shift] = alpha

        return mask

    def load_slide(self, loc, axis=0, width=3):
        """ Create a mask at desired location along supplied axis. """
        axis = self.field.geometry.parse_axis(axis)
        locations = self.field.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])
        width = width or max(5, shape[-1] // 100)

        mask = np.zeros(shape, dtype=np.float32)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        return np.squeeze(mask)


    # Evaluate horizon on its own / against other(s)
    @property
    def metrics(self):
        """ Calculate :class:`~HorizonMetrics` on demand. """
        # pylint: disable=import-outside-toplevel
        from ..metrics import HorizonMetrics
        return HorizonMetrics(self)

    def evaluate(self, compute_metric=True, supports=50, plot=True, savepath=None, printer=print, **kwargs):
        """ Compute crucial metrics of a horizon.

        Parameters
        ----------
        compute_metrics : bool
            Whether to compute correlation map of a horizon.
        supports, savepath, plot, kwargs
            Passed directly to :meth:`HorizonMetrics.evaluate`.
        printer : callable
            Function to display message with metrics.
        """
        msg = f"""
        Number of labeled points:                         {len(self)}
        Number of points inside borders:                  {np.sum(self.filled_matrix)}
        Perimeter (length of borders):                    {self.perimeter}
        Percentage of labeled non-bad traces:             {self.coverage}
        Percentage of labeled traces inside borders:      {self.solidity}
        Number of holes inside borders:                   {self.number_of_holes}
        """
        printer(dedent(msg))
        if compute_metric:
            return self.metrics.evaluate('support_corrs', supports=supports, agg='nanmean',
                                         plot=plot, savepath=savepath, **kwargs)
        return None

    def compare(self, other, offset=0, absolute=True, printer=print, hist=True, plot=True):
        """ Compare quality of self against another horizon or sequence of horizons. """
        from ..metrics import HorizonMetrics
        HorizonMetrics([self, other]).evaluate('compare', absolute=absolute, offset=offset,
                                               printer=printer, hist=hist, plot=plot)

    def check_proximity(self, other):
        """ Compute a number of stats of location of `self` relative to the `other` Horizons.

        Parameters
        ----------
        self, other : Horizon
            Horizons to compare.

        Returns
        -------
        dictionary with following keys:
            - `difference_matrix` with matrix of depth differences
            - `difference_mean` for average distance
            - `difference_abs_mean` for average of absolute values of point-wise distances
            - `difference_max`, `difference_abs_max`, `difference_std`, `difference_abs_std`
            - `overlap_size` with number of overlapping points
            - `window_rate` for percentage of traces that are in 5ms from one horizon to the other
        """
        difference = np.where((self.full_matrix != self.FILL_VALUE) & (other.full_matrix != self.FILL_VALUE),
                              self.full_matrix - other.full_matrix, np.nan)
        abs_difference = np.abs(difference)

        overlap_size = np.nansum(~np.isnan(difference))
        window_rate = np.nansum(abs_difference < (5 / self.field.sample_rate)) / overlap_size

        present_at_1_absent_at_2 = ((self.full_matrix != self.FILL_VALUE)
                                    & (other.full_matrix == self.FILL_VALUE)).sum()
        present_at_2_absent_at_1 = ((self.full_matrix == self.FILL_VALUE)
                                    & (other.full_matrix != self.FILL_VALUE)).sum()

        info_dict = {
            'difference_matrix' : difference,
            'difference_mean' : np.nanmean(difference),
            'difference_max' : np.nanmax(difference),
            'difference_min' : np.nanmin(difference),
            'difference_std' : np.nanstd(difference),

            'abs_difference_mean' : np.nanmean(abs_difference),
            'abs_difference_max' : np.nanmax(abs_difference),
            'abs_difference_std' : np.nanstd(abs_difference),

            'overlap_size' : overlap_size,
            'window_rate' : window_rate,

            'present_at_1_absent_at_2' : present_at_1_absent_at_2,
            'present_at_2_absent_at_1' : present_at_2_absent_at_1,
        }
        return MetaDict(info_dict)

    def equal(self, other, threshold_missing=0):
        """ Return True if the horizons are considered equal, False otherwise.
        If the `threshold_missing` is zero, then check if the points of `self` and `other` are the same.
        If the `threshold_missing` is positive, then check that in overlapping points values are the same,
        and number of missing traces is smaller than allowed.
        """
        if threshold_missing == 0:
            return np.array_equal(self.points, other.points)

        info = self.check_proximity(other)
        n_missing = max(info['present_at_1_absent_at_2'], info['present_at_2_absent_at_1'])
        return info['difference_mean'] == 0 and n_missing < threshold_missing


    # Save horizon to disk
    def dump(self, path, transform=None):
        """ Save horizon points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        self.dump_charisma(data=copy(self.points), path=path, format='points',
                           name=self.short_name, transform=transform)

    def dump_float(self, path, transform=None, kernel_size=7, sigma=2., margin=5):
        """ Smooth out the horizon values, producing floating-point numbers, and dump to the disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        kernel_size : int
            Size of the filtering kernel.
        sigma : number
            Standard deviation of the Gaussian kernel.
        margin : number
            During the filtering, not include in the computation all the points that are
            further away from the current, than the margin.
        """
        matrix = self.matrix_smooth_out(matrix=self.full_matrix, kernel_size=kernel_size, sigma=sigma, margin=margin)
        points = self.matrix_to_points(matrix)
        self.dump_charisma(data=points, path=path, format='points', name=self.short_name, transform=transform)
