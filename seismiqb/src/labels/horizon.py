""" Horizon class for POST-STACK data. """
import os
from copy import copy
from textwrap import dedent
from functools import partialmethod

import numpy as np
import pandas as pd

from cv2 import dilate
from skimage.measure import label
from scipy.ndimage import find_objects
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from .horizon_attributes import AttributesMixin
from .horizon_visualization import VisualizationMixin
from ..utils import CacheMixin
from ..utils import groupby_mean, groupby_min, groupby_max, filtering_function
from ..utils import make_bezier_figure
from ..functional import smooth_out



class Horizon(AttributesMixin, CacheMixin, VisualizationMixin):
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
    #pylint: disable=too-many-public-methods, import-outside-toplevel

    # CHARISMA: default seismic format of storing surfaces inside the 3D volume
    CHARISMA_SPEC = ['INLINE', '_', 'iline', 'XLINE', '__', 'xline', 'cdp_x', 'cdp_y', 'height']

    # REDUCED_CHARISMA: CHARISMA without redundant columns
    REDUCED_CHARISMA_SPEC = ['iline', 'xline', 'height']

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999


    def __init__(self, storage, field, name=None, dtype=np.int32, **kwargs):
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
        if isinstance(storage, str):
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
            points += np.array([self.i_min, self.x_min, 0])
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
            self._matrix = self.points_to_matrix(self.points, self.i_min, self.x_min,
                                                 self.i_length, self.x_length, self.dtype)
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
                self.i_min, self.x_min, _ = np.min(self.points, axis=0).astype(np.int32)
                self.i_max, self.x_max, _ = np.max(self.points, axis=0).astype(np.int32)

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


    # Coordinate transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] -= self.field.ilines_offset
        array[:, 1] -= self.field.xlines_offset
        array[:, 2] -= self.field.delay
        array[:, 2] /= self.field.sample_rate
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array = array.astype(np.float32)
        array[:, 0] += self.field.ilines_offset
        array[:, 1] += self.field.xlines_offset
        array[:, 2] *= self.field.sample_rate
        array[:, 2] += self.field.delay
        return array


    # Initialization from different containers
    def from_points(self, points, transform=False, verify=True, dst='points', reset='matrix', **kwargs):
        """ Base initialization: from point cloud array of (N, 3) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: two spatial coordinates and depth.
        transform : bool
            Whether transform from line coordinates (ilines, xlines) to cubic system.
        verify : bool
            Whether to remove points outside of the cube range.
        dst : str
            Attribute to save result.
        reset : str or None
            Storage to reset.
        """
        _ = kwargs

        # Transform to cubic coordinates, if needed
        if transform:
            points = self.lines_to_cubic(points)
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
        _ = kwargs
        path = self.field.make_path(path, makedirs=False)

        self.path = path
        self.name = os.path.basename(path) if self.name is None else self.name
        points = self.file_to_points(path)
        self.from_points(points, transform, **kwargs)

    @classmethod
    def file_to_points(cls, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path, encoding='utf-8') as file:
            line_len = len(file.readline().split(' '))
        if line_len == 3:
            names = cls.REDUCED_CHARISMA_SPEC
        elif line_len >= 9:
            names = cls.CHARISMA_SPEC
        else:
            raise ValueError('Horizon labels must be in CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep=r'\s+', names=names, usecols=cls.COLUMNS)
        df.sort_values(cls.COLUMNS, inplace=True)
        return df.values


    def from_matrix(self, matrix, i_min, x_min, length=None, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        if np.issubdtype(self.dtype, np.integer):
            matrix = np.rint(matrix)
        self.matrix = matrix.astype(self.dtype)

        self.i_min, self.x_min = i_min, x_min
        self.i_max, self.x_max = i_min + matrix.shape[0] - 1, x_min + matrix.shape[1] - 1

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max],
                              [self.h_min, self.h_max]],
                             dtype=np.int32)

        self.reset_storage('points')
        self._len = length

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
        self.from_points(points, transform=transform)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    @staticmethod
    def from_mask(mask, field=None, origin=None,
                  mode='mean', threshold=0.5, minsize=0, prefix='predict', **kwargs):
        """ Convert mask to a list of horizons.
        Returned list is sorted on length of horizons.

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
            Minimum length of a horizon to be saved.
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
        labeled = label(mask >= threshold)
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
            Other parameters for grid creation, see `:meth:~.SeismicGeometry.make_grid`.
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
        filtering_matrix = np.zeros_like(self.matrix)

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

    def check_proximity(self, other, offset=0):
        """ Compute a number of stats on location of `self` relative to the `other` Horizons.
        This method can be used as either bound or static method.

        Parameters
        ----------
        self, other : Horizon
            Horizons to compare.
        offset : number
            Value to shift the first horizon down.

        Returns
        -------
        dictionary with following keys:
            - `mean` for average distance
            - `abs_mean` for average of absolute values of point-wise distances
            - `max`, `abs_max`, `std`, `abs_std`
            - `window_rate` for percentage of traces that are in 5ms from one horizon to the other
            - `offset_diffs` with point-wise differences
        """
        _, overlap_info = self.verify_merge(other)
        diffs = overlap_info.get('diffs', 999) + offset

        overlap_info = {
            **overlap_info,
            'mean': np.mean(diffs),
            'abs_mean': np.mean(np.abs(diffs)),
            'max': np.max(diffs),
            'abs_max': np.max(np.abs(diffs)),
            'std': np.std(diffs),
            'abs_std': np.std(np.abs(diffs)),
            'window_rate': np.mean(np.abs(diffs) < (5 / self.field.sample_rate)),
            'offset_diffs': diffs,
        }
        return overlap_info


    # Merge functions
    def verify_merge(self, other, mean_threshold=3.0, adjacency=0):
        """ Collect stats of overlapping of two horizons.

        Returns a number that encodes position of two horizons, as well as dictionary with collected statistics.
        If code is 0, then horizons are too far away from each other (heights-wise), and therefore are not mergeable.
        If code is 1, then horizons are too far away from each other (spatially) even with adjacency, and therefore
        are not mergeable.
        If code is 2, then horizons are close enough spatially (with adjacency), but are not overlapping, and therefore
        an additional check (`adjacent_merge`) is needed.
        If code is 3, then horizons are definitely overlapping and are close enough to meet all the thresholds, and
        therefore are mergeable without any additional checks.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to compare.
        mean_threshold : number
            Height threshold for mean distances.
        adjacency : int
            Margin to consider horizons to be close (spatially).
        """
        overlap_info = {}

        # Overlap bbox
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        i_range = overlap_i_min - overlap_i_max
        x_range = overlap_x_min - overlap_x_max

        # Simplest possible check: horizon bboxes are too far from each other
        if i_range >= adjacency or x_range >= adjacency:
            merge_code = 1
            spatial_position = 'distant'
        else:
            merge_code = 2
            spatial_position = 'adjacent'

        # Compare matrices on overlap without adjacency:
        if merge_code != 1 and i_range < 0 and x_range < 0:
            self_overlap = self.matrix[overlap_i_min - self.i_min:overlap_i_max - self.i_min,
                                       overlap_x_min - self.x_min:overlap_x_max - self.x_min]

            other_overlap = other.matrix[overlap_i_min - other.i_min:overlap_i_max - other.i_min,
                                         overlap_x_min - other.x_min:overlap_x_max - other.x_min]

            self_mask = self_overlap != self.FILL_VALUE
            other_mask = other_overlap != self.FILL_VALUE
            mask = self_mask & other_mask
            diffs_on_overlap = self_overlap[mask] - other_overlap[mask]

            if len(diffs_on_overlap) == 0:
                # bboxes are overlapping, but horizons are not
                merge_code = 2
                spatial_position = 'adjacent'
            else:
                abs_diffs = np.abs(diffs_on_overlap)
                mean_on_overlap = np.mean(abs_diffs)
                if mean_on_overlap < mean_threshold:
                    merge_code = 3
                    spatial_position = 'overlap'
                else:
                    merge_code = 0
                    spatial_position = 'separated'

                overlap_info.update({'mean': mean_on_overlap,
                                     'diffs': diffs_on_overlap})

        overlap_info['spatial_position'] = spatial_position
        return merge_code, overlap_info

    def overlap_merge(self, other, inplace=False):
        """ Merge two horizons into one.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.
        """
        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Add both horizons to the background
        background[shared_self_i_min:shared_self_i_min+self.i_length,
                   shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Correct overlapping points
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        overlap_i_min -= shared_i_min
        overlap_i_max -= shared_i_min
        overlap_x_min -= shared_x_min
        overlap_x_max -= shared_x_min

        overlap = background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max]
        mask = overlap >= 0
        overlap[mask] //= 2
        overlap[~mask] -= self.FILL_VALUE
        background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max] = overlap

        background[background == 0] = self.FILL_VALUE
        length = len(self) + len(other) - mask.sum()
        # Create new instance or change `self`
        if inplace:
            # Change `self` inplace
            self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
            merged = True
        else:
            # Return a new instance of horizon
            merged = Horizon(storage=background, field=self.field, name=self.name,
                             i_min=shared_i_min, x_min=shared_x_min, length=length)
        return merged

    def adjacent_merge(self, other, mean_threshold=3.0, adjacency=3, inplace=False):
        """ Check if adjacent merge (that is merge with some margin) is possible, and, if needed, merge horizons.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to merge.
        mean_threshold : number
            Height threshold for mean distances.
        adjacency : int
            Margin to consider horizons close (spatially).
        inplace : bool
            Whether to create new instance or update `self`.
        """
        # Simplest possible check: horizons are too far away from one another (depth-wise)
        overlap_h_min, overlap_h_max = max(self.h_min, other.h_min), min(self.h_max, other.h_max)
        if overlap_h_max - overlap_h_min < 0:
            return False

        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Put the second of the horizons on background
        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Enlarge the image to account for adjacency
        kernel = np.ones((3, 3), np.float32)
        dilated_background = dilate(background.astype(np.float32), kernel,
                                    iterations=adjacency).astype(np.int32)

        # Make counts: number of horizons in each point; create indices of overlap
        counts = (dilated_background != 0).astype(np.int32)
        counts[shared_self_i_min:shared_self_i_min+self.i_length,
               shared_self_x_min:shared_self_x_min+self.x_length] += 1
        counts_idx = counts == 2

        # Determine whether horizon can be merged (adjacent and height-close) or not
        mergeable = False
        if counts_idx.any():
            # Put the first horizon on dilated background, compute mean
            background[shared_self_i_min:shared_self_i_min+self.i_length,
                       shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

            # Compute diffs on overlap
            diffs = background[counts_idx] - dilated_background[counts_idx]
            diffs = np.abs(diffs)
            diffs = diffs[diffs < (-self.FILL_VALUE // 2)]

            if len(diffs) != 0 and np.mean(diffs) < mean_threshold:
                mergeable = True

        if mergeable:
            background[(background < 0) & (background != self.FILL_VALUE)] -= self.FILL_VALUE
            background[background == 0] = self.FILL_VALUE

            length = len(self) + len(other) # since there is no direct overlap

            # Create new instance or change `self`
            if inplace:
                # Change `self` inplace
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
                merged = True
            else:
                # Return a new instance of horizon
                merged = Horizon(storage=background, field=self.field, name=self.name,
                                 i_min=shared_i_min, x_min=shared_x_min, length=length)
            return merged
        return False

    @staticmethod
    def merge_list(horizons, mean_threshold=2.0, adjacency=3, minsize=50):
        """ Iteratively try to merge every horizon in a list to every other, until there are no possible merges.
        Parameters are passed directly to :meth:`~.verify_merge`, :meth:`~.overlap_merge` and :meth:`~.adjacent_merge`.
        """
        horizons = [horizon for horizon in horizons if len(horizon) >= minsize]

        # Iterate over the list of horizons to merge everything that can be merged
        i = 0
        flag = True
        while flag:
            # Continue while at least one pair of horizons was merged at previous iteration
            flag = False
            while True:
                if i >= len(horizons):
                    break

                j = i + 1
                while True:
                    # Attempt to merge j-th horizon to i-th horizon
                    if j >= len(horizons):
                        break

                    merge_code, _ = Horizon.verify_merge(horizons[i], horizons[j],
                                                         mean_threshold=mean_threshold,
                                                         adjacency=adjacency)
                    if merge_code == 3:
                        merged = Horizon.overlap_merge(horizons[i], horizons[j], inplace=True)
                    elif merge_code == 2:
                        merged = Horizon.adjacent_merge(horizons[i], horizons[j], inplace=True,
                                                        mean_threshold=mean_threshold,
                                                        adjacency=adjacency)
                    else:
                        merged = False

                    if merged:
                        _ = horizons.pop(j)
                        flag = True
                    else:
                        j += 1
                i += 1
        return sorted(horizons, key=len, reverse=True)


    @staticmethod
    def average_horizons(horizons):
        """ Average list of horizons into one surface. """
        field = horizons[0].field
        horizon_matrix = np.zeros(field.spatial_shape, dtype=np.float32)
        std_matrix = np.zeros(field.spatial_shape, dtype=np.float32)
        counts_matrix = np.zeros(field.spatial_shape, dtype=np.int32)

        for horizon in horizons:
            fm = horizon.full_matrix
            horizon_matrix[fm != Horizon.FILL_VALUE] += fm[fm != Horizon.FILL_VALUE]
            std_matrix[fm != Horizon.FILL_VALUE] += fm[fm != Horizon.FILL_VALUE] ** 2
            counts_matrix[fm != Horizon.FILL_VALUE] += 1

        horizon_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        horizon_matrix[counts_matrix == 0] = Horizon.FILL_VALUE

        std_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        std_matrix -= horizon_matrix ** 2
        std_matrix = np.sqrt(std_matrix)
        std_matrix[counts_matrix == 0] = np.nan

        averaged_horizon = Horizon(horizon_matrix.astype(np.int32), field=field)
        return averaged_horizon, {
            'matrix': horizon_matrix,
            'std_matrix': std_matrix,
        }


    # Save horizon to disk
    def dump_charisma(self, points, path, transform=None):
        """ Save (N, 3) array of points to disk in CHARISMA-compatible format.

        Parameters
        ----------
        points : ndarray
            Array of (N, 3) shape.
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        points = points if transform is None else transform(points)
        path = self.field.make_path(path, name=self.short_name)

        df = pd.DataFrame(points, columns=Horizon.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)
        df = df.astype({'iline': np.int32, 'xline': np.int32, 'height': np.float32})
        df.to_csv(path, sep=' ', columns=Horizon.COLUMNS, index=False, header=False)

    def dump_matrix(self, matrix, path, transform=None):
        """ Save (N_ILINES, N_CROSSLINES) matrix in CHARISMA-compatible format.

        Parameters
        ----------
        matrix : ndarray
            Array of (N_ILINES, N_CROSSLINES) shape with depth values.
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        points = Horizon.matrix_to_points(matrix)
        points = self.cubic_to_lines(points)
        Horizon.dump_charisma(points, path, transform)

    def dump(self, path, transform=None):
        """ Save horizon points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        points = self.cubic_to_lines(copy(self.points))
        self.dump_charisma(points, path, transform)

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
        points = self.cubic_to_lines(points)
        self.dump_charisma(points, path, transform)


    # Utility
    @staticmethod
    def is_charisma_like(path, bad_extensions=None, size_threshold=100):
        """ Check if the path looks like the horizon file.

        Parameters
        ----------
        path : str
            Path of file to check.
        bad_extensions : list, optional
            If provided, then list of extensions to consider file not charisma-like.
        size_threshold : number
            If file size in kilobytes is less, than the threshold, then file is considered not charisma-like.
        """
        bad_extensions = bad_extensions or []
        bad_extensions.extend(['.py', '.ipynb', '.ckpt',
                            '.png', '.jpg',
                            '.log', '.txt', '.torch'])

        try:
            if os.path.isdir(path):
                return False

            if max([path.endswith(ext) for ext in bad_extensions]):
                return False

            if (os.path.getsize(path) / 1024) < size_threshold:
                return False
            with open(path, encoding='utf-8') as file:
                line = file.readline()
                n = len(line.split(' '))
            return (n == 3) or (n >= 9 and 'INLINE' in line)

        except UnicodeDecodeError:
            return False
