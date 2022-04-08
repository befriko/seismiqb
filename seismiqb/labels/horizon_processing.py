""" Mixin for horizon processing. """
import numpy as np

from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion

from ..functional import smooth_out, interpolate
from ..utils import make_bezier_figure

class ProcessingMixin:
    """ Methods for horizon processing.

    This class contains methods which can be divided into the following groups:
        - Filtering methods to cut out some surface regions.
        - Horizon transformations such as smoothing and thinning.
        - Surface distortions such as holes or carcass creation.
    All of these methods affect horizon structure, so be careful with them: their result is a new horizon surface.
    """
    # Filtering methods
    def filter_points(self, filtering_matrix=None, margin=0, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from points storage. """
        if filtering_matrix is None:
            filtering_matrix = self.field.zero_traces
        if margin > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure=np.ones((margin, margin)))
            filtering_matrix[:margin, :] = 1
            filtering_matrix[:, :margin] = 1
            filtering_matrix[-margin:, :] = 1
            filtering_matrix[:, -margin:] = 1

        mask = filtering_matrix[self.points[:, 0], self.points[:, 1]]
        self.points = self.points[mask == 0]
        self.reset_storage('matrix')

    def filter_matrix(self, filtering_matrix=None, margin=0, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from matrix storage. """
        if filtering_matrix is None:
            filtering_matrix = self.field.zero_traces
        if margin > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure=np.ones((margin, margin)))
            filtering_matrix[:margin, :] = 1
            filtering_matrix[:, :margin] = 1
            filtering_matrix[-margin:, :] = 1
            filtering_matrix[:, -margin:] = 1

        idx_i, idx_x = np.asarray(filtering_matrix[self.i_min:self.i_max + 1,
                                                   self.x_min:self.x_max + 1] == 1).nonzero()

        self.matrix[idx_i, idx_x] = self.FILL_VALUE
        self.reset_storage('points')

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

    def filter_disconnected_regions(self, erosion_rate=0):
        """ Remove regions, not connected to the largest component of a horizon. """
        if erosion_rate > 0:
            structure = np.ones((3, 3))
            matrix = binary_erosion(self.presence_matrix, structure, iterations=erosion_rate)
        else:
            matrix = self.presence_matrix

        labeled = label(matrix)
        values, counts = np.unique(labeled, return_counts=True)
        counts = counts[values != 0]
        values = values[values != 0]

        object_id = values[np.argmax(counts)]

        filtering_matrix = np.zeros_like(self.presence_matrix)
        filtering_matrix[labeled == object_id] = 1

        if erosion_rate > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure, iterations=erosion_rate)

        filtering_matrix = filtering_matrix == 0

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

    def smooth_out(self, kernel=None, kernel_size=3, sigma=0.8, iters=1, preserve_borders=True, margin=5, **_):
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
        smoothed = smooth_out(self.matrix, kernel=kernel,
                              kernel_size=kernel_size, sigma=sigma, margin=margin,
                              fill_value=self.FILL_VALUE, preserve=preserve_borders, iters=iters)

        smoothed = np.rint(smoothed).astype(np.int32)
        smoothed[self.field.zero_traces[self.i_min:self.i_max + 1,
                                        self.x_min:self.x_max + 1] == 1] = self.FILL_VALUE

        self.matrix = smoothed
        self.reset_storage('points')

    def interpolate(self, kernel=None, kernel_size=3, sigma=0.8, iters=1, **_):
        """ Interpolate horizon surface on the regions with missing traces.

        Under the hood, we fill missing traces with smoothed neighbor values.
        """
        interpolated = interpolate(self.matrix, kernel=kernel, kernel_size=kernel_size, preserve=False,
                                   sigma=sigma, fill_value=self.FILL_VALUE, iters=iters)

        interpolated = np.rint(interpolated).astype(np.int32)
        interpolated[self.field.zero_traces[self.i_min:self.i_max + 1,
                                        self.x_min:self.x_max + 1] == 1] = self.FILL_VALUE

        self.matrix = interpolated
        self.reset_storage('points')

    # Horizon distortions
    def make_carcass(self, frequencies=100, regular=True, margin=50, apply_smoothing=False, add_prefix=True, **kwargs):
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
        #pylint: disable=import-outside-toplevel
        frequencies = frequencies if isinstance(frequencies, (tuple, list)) else [frequencies]
        carcass = self.copy(add_prefix=add_prefix)
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

    def make_holes(self, inplace=False, n=10, scale=1.0, max_scale=.25,
                   max_angles_amount=4, max_sharpness=5.0, locations=None,
                   points_proportion=1e-5, points_shape=1,
                   noise_level=0, seed=None):
        """ Make holes in a of horizon. Optionally, make a copy before filtering. """
        #pylint: disable=self-cls-assignment
        filtering_matrix = self.make_random_holes_matrix(n=n, scale=scale, max_scale=max_scale,
                                                         max_angles_amount=max_angles_amount,
                                                         max_sharpness=max_sharpness, locations=locations,
                                                         points_proportion=points_proportion, points_shape=points_shape,
                                                         noise_level=noise_level, seed=seed)
        self = self if inplace is True else self.copy()
        self.filter(filtering_matrix)
        return self

    make_holes.__doc__ += '\n' + '\n'.join(make_random_holes_matrix.__doc__.split('\n')[1:])
