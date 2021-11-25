""" Seismic Crop Batch. """
import string
import random
from copy import copy
from warnings import warn

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter, hilbert

from ..batchflow import DatasetIndex, Batch, action, inbatch_parallel, SkipBatchException, apply_parallel

from .labels import Horizon
from .plotters import plot_image
from .synthetic.generator import generate_synthetic
from .utils import compute_attribute, to_list, AugmentedDict


AFFIX = '___'
SIZE_POSTFIX = 12
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX
CHARS = string.ascii_uppercase + string.digits


class SeismicCropBatch(Batch):
    """ Batch with ability to generate 3d-crops of various shapes.

    The first action in any pipeline with this class should be `make_locations` to transform batch index from
    individual cubes into crop-based indices. The transformation uses randomly generated postfix (see `:meth:.salt`)
    to obtain unique elements.
    """
    components = None
    apply_defaults = {
        'target': 'for',
        'post': '_assemble'
    }


    def _init_component(self, *args, **kwargs):
        """ Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices.
        """
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if isinstance(dst, str):
            dst = (dst,)
        for comp in dst:
            if not hasattr(self, comp):
                self.add_components(comp, np.array([np.nan] * len(self.index)))
        return self.indices

    @action
    def add_components(self, components, init=None):
        """ Add new components, checking that attributes of the same name are not present in dataset.

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data

        Raises
        ------
        ValueError
            If a component or an attribute with the given name already exists in batch or dataset.
        """
        for component in to_list(components):
            if hasattr(self.dataset, component):
                msg = f"Component with `{component}` name cannot be added to batch, "\
                      "since attribute with this name is already present in dataset."
                raise ValueError(msg)
        super().add_components(components=components, init=init)

    # Inner workings
    @staticmethod
    def salt(path):
        """ Adds random postfix of predefined length to string.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        path : str
            supplied string with random postfix.

        Notes
        -----
        Action `make_locations` makes a new instance of SeismicCropBatch with different (enlarged) index.
        Items in that index should point to cube location to cut crops from.
        Since we can't store multiple copies of the same string in one index (due to internal usage of dictionary),
        we need to augment those strings with random postfix, which can be removed later.
        """
        return path + AFFIX + ''.join(random.choice(CHARS) for _ in range(SIZE_POSTFIX))

    @staticmethod
    def has_salt(path):
        """ Check whether path is salted. """
        return path[::-1].find(AFFIX) == SIZE_POSTFIX

    @staticmethod
    def unsalt(path):
        """ Removes postfix that was made by `salt` method.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        str
            string without postfix.
        """
        if AFFIX in path:
            return path[:-SIZE_SALT]
        return path


    def get(self, item=None, component=None):
        """ Custom access for batch attributes.
        If `component` is present in dataset and is an instance of `AugmentedDict`,
        then index it with item and return it.
        Otherwise retrieve `component` from batch itself and optionally index it with `item` position in `self.indices`.
        """
        data = getattr(self.dataset, component, None)
        if isinstance(data, AugmentedDict):
            if isinstance(item, str) and self.has_salt(item):
                item = self.unsalt(item)
            return data[item]

        data = getattr(self, component) if isinstance(component, str) else component
        if item is not None:
            if isinstance(data, (np.ndarray, list)) and len(data) == len(self):
                pos = np.where(self.indices == item)[0][0]
                return data[pos]
            return super().get(item, component)

        return data


    # Core actions
    @action
    def make_locations(self, generator, batch_size=None, passdown=None):
        """ Use `generator` to create `batch_size` locations.
        Each location defines position in a cube and can be used to retrieve data/create masks at this place.

        Generator can be either Sampler or Grid to make locations in a random or deterministic fashion.
        `generator` must be a callable and return (batch_size, 9+) array, where the first nine columns should be:
        (field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
        `generator` must have `to_names` method to convert cube and label ids into actual strings.

        Field and label ids are transformed into names of actual fields and labels (horizons, faults, facies, etc).
        Then we create a completely new instance of `SeismicCropBatch`, where the new index is set to
        cube names with additional postfixes (see `:meth:.salt`), which is returned as the result of this action.

        After parsing contents of generated (batch_size, 9+) array we add following attributes:
            - `locations` with triplets of slices
            - `orientations` with crop orientation: 0 for iline direction, 1 for crossline direction
            - `shapes`
            - `label_names`
            - `generated` with originally generated data
        If `generator` creates more than 9 columns, they are not used, but stored in the  `generated` attribute.

        Parameters
        ----------
        generator : callable
            Sampler or Grid to retrieve locations. Must be a callable from positive integer.
        batch_size : int
            Number of locations to generate.
        passdown : str or sequence of str
            Components to pass down to a newly created batch.

        Returns
        -------
        SeismicCropBatch
            A completely new instance of Batch.
        """
        # pylint: disable=protected-access
        generated = generator(batch_size)

        # Convert IDs to names, that are used in dataset
        field_names, label_names = generator.to_names(generated[:, [0, 1]]).T

        # Locations: 3D slices in the cube coordinates
        locations = [[slice(i_start, i_stop), slice(x_start, x_stop), slice(h_start, h_stop)]
                      for i_start, x_start, h_start, i_stop,  x_stop,  h_stop in generated[:, 3:9]]

        # Additional info
        orientations = generated[:, 2]
        shapes = generated[:, [6, 7, 8]] - generated[:, [3, 4, 5]]

        # Create a new SeismicCropBatch instance
        new_index = [self.salt(ix) for ix in field_names]
        new_batch = type(self)(DatasetIndex.from_index(index=new_index))

        # Keep chosen components in the new batch
        if passdown:
            passdown = [passdown] if isinstance(passdown, str) else passdown
            for component in passdown:
                if hasattr(self, component):
                    new_batch.add_components(component, getattr(self, component))

        new_batch.add_components(('locations', 'generated', 'shapes', 'orientations', 'label_names'),
                                 (locations, generated, shapes, orientations, label_names))
        return new_batch

    @action
    def adaptive_reshape(self, src=None, dst=None):
        """ Transpose crops with crossline orientation into (x, i, h) order. """
        src = src if isinstance(src, (tuple, list)) else [src]
        if dst is None:
            dst = src
        dst = dst if isinstance(dst, (tuple, list)) else [dst]

        for src_, dst_ in zip(src, dst):
            result = []
            for ix in self.indices:
                item = self.get(ix, src_)
                if self.get(ix, 'orientations'):
                    item = item.transpose(1, 0, 2)
                result.append(item)
            setattr(self, dst_, np.stack(result))
        return self


    # Loading of cube data and its derivatives
    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def load_cubes(self, ix, dst, native_slicing=False, src_geometry='geometry', **kwargs):
        """ Load data from cube for stored `locations`.

        Parameters
        ----------
        dst : str
            Component of batch to put loaded crops in.
        slicing : str
            If 'custom', use `load_crop` method to make crops.
            if 'native', crop will be loaded as a slice of geometry. Preferred for 3D crops to speed up loading.
        src_geometry : str
            Field attribute with desired geometry.
        """
        field = self.get(ix, 'fields')
        location = self.get(ix, 'locations')
        return field.load_seismic(location=location, native_slicing=native_slicing, src=src_geometry, **kwargs)

    @action
    @inbatch_parallel(init='indices', src='indices', target='for', post='_assemble',
                      dst=('cubes', 'masks', 'faults'))
    def generate_synthetic(self, ix, dst=None, **kwargs):
        """ Generate synthetic seismic, corresponding horizons' and faults'
        masks and put it into batch-components.

        Parameters
        ----------
        dst : sequence
            Sequence of length=3 to put created components in.
        **kwargs : dict
            All arguments of `generate_synthetic`-method. See the docstring for more
            info.
        """
        _, _ = self, ix

        # if the requested shape is in fact 2d (for instance [1, 64, 128] or [64, 1, 128])
        # performs `generate_synthetic` in 2d and then adds the missing axis
        shape = np.array(kwargs['shape'])
        if 1 in shape[:2]:
            kwargs['shape'] = tuple(shape[shape != 1])
            axis_num = 0 if shape[0] == 1 else 1
        else:
            axis_num = None

        # generate synthetic crop, horizons and faults
        crop, horizons, faults = generate_synthetic(**kwargs)
        if axis_num is not None:
            crop, horizons, faults = [np.expand_dims(array, axis=axis_num) for array in (crop, horizons, faults)]

        # set locations if needed
        # locations = (slice(0, crop.shape[0]), slice(0, crop.shape[1]), slice(0, crop.shape[2]))
        return crop, horizons, faults

    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def normalize(self, ix, mode=None, itemwise=False, src=None, dst=None, q=(0.01, 0.99)):
        """ Normalize values in crop.

        Parameters
        ----------
        mode : callable or str
            If callable, then directly applied to data.
            If str, then :meth:`~SeismicGeometry.scaler` applied in one of the modes:
            - `minmax`: scaled to [0, 1] via minmax scaling.
            - `q` or `normalize`: divided by the maximum of absolute values
                                  of the 0.01 and 0.99 quantiles. Quantiles can
                                  be changed by `q` parameter.
            - `q_clip`: clipped to 0.01 and 0.99 quantiles and then divided
                        by the maximum of absolute values of the two. Quantiles can
                        be changed by `q` parameter.
        itemwise : bool
            The way to compute 'min', 'max' and quantiles. If False, stats will be computed
            for the whole cubes. Otherwise, for each data item separately.
        q : tuple
            Left and right quantiles to use.
        """
        data = self.get(ix, src)
        if callable(mode):
            normalized = mode(data)
            return normalized

        if itemwise:
            # Adjust data based on the current item only
            if mode == 'minmax':
                min_, max_ = data.min(), data.max()
                normalized = (data - min_) / (max_ - min_) if (max_ != min_) else np.zeros_like(data)
            else:
                left, right = np.quantile(data, q)
                if mode in ['q', 'normalize']:
                    normalized = 2 * (data - left) / (right - left) - 1 if right != left else np.zeros_like(data)
                elif mode == 'q_clip':
                    normalized =  np.clip(data, left, right) / max(abs(left), abs(right))
                else:
                    raise ValueError(f'Unknown mode: {mode}')
        else:
            field = self.get(ix, 'fields')
            normalized = field.geometry.normalize(data, mode=mode)
        return normalized


    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def compute_attribute(self, ix, dst, src='images', attribute='semblance', window=10, stride=1, device='cpu'):
        """ Compute geological attribute.

        Parameters
        ----------
        dst : str
            Destination batch component
        src : str, optional
            Source batch component, by default 'images'
        attribute : str, optional
            Attribute to compute, by default 'semblance'
        window : int or tuple, optional
            Window to compute attribute, by default 10 (for each axis)
        stride : int, optional
            Stride for windows, by default 1 (for each axis)
        device : str, optional
            Device to compute attribute, by default 'cpu'

        Returns
        -------
        SeismicCropBatch
            Batch with loaded masks in desired components.
        """
        image = self.get(ix, src)
        result = compute_attribute(image, window, device, attribute)
        return result


    # Loading of labels
    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def create_masks(self, ix, dst, indices='all', width=3, src_labels='labels'):
        """ Create masks from labels in stored `locations`.

        Parameters
        ----------
        dst : str
            Component of batch to put loaded masks in.
        indices : str, int or sequence of ints
            Which labels to use in mask creation.
            If 'all', then use all labels.
            If 'single' or `random`, then use one random label.
            If int or array-like, then element(s) are interpreted as indices of desired labels.
        width : int
            Width of the resulting label.
        src_labels : str
            Dataset attribute with labels dict.
        """
        field = self.get(ix, 'fields')
        location = self.get(ix, 'locations')
        return field.make_mask(location=location, width=width, indices=indices, src=src_labels)


    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def compute_label_attribute(self, ix, dst, src='amplitudes', atleast_3d=True, dtype=np.float32, **kwargs):
        """ Compute requested attribute along label surface. Target labels are defined by sampled locations.

        Parameters
        ----------
        src : str
            Keyword that defines label attribute to compute.
        atleast_3d : bool
            Whether add one more dimension to 2d result or not.
        dtype : valid dtype compatible with requested attribute
            A dtype that result must have.
        kwargs : misc
            Passed directly to one of attribute-evaluating methods.

        Notes
        -----
        Correspondence between the attribute and the method that computes it
        is defined by :attr:`~Horizon.ATTRIBUTE_TO_METHOD`.
        """
        field = self.get(ix, 'fields')
        location = self.get(ix, 'locations')
        label_index = self.get(ix, 'generated')[1]
        src = src.replace('*', str(label_index))

        src_labels = src[:src.find(':')]
        label = getattr(field, src_labels)[label_index]
        label_name = self.get(ix, 'label_names')
        if label.short_name != label_name:
            msg = f"Name `{label.short_name}` of the label loaded by index {label_index} "\
                  f"from {src_labels} does not match label name {label_name} from batch."\
                  f"This might have happened due to items order change in {src_labels} "\
                  f"in between sampler creation and `make_locations` call."
            raise ValueError(msg)

        result = field.load_attribute(src=src, location=location, atleast_3d=atleast_3d, dtype=dtype, **kwargs)

        return result


    # More methods to work with labels
    @action
    @inbatch_parallel(init='indices', post='_post_mask_rebatch', target='for',
                      src='masks', threshold=0.8, passdown=None, axis=-1)
    def mask_rebatch(self, ix, src='masks', threshold=0.8, passdown=None, axis=-1):
        """ Remove elements with masks area lesser than a threshold.

        Parameters
        ----------
        threshold : float
            Minimum percentage of covered area for a mask to be kept in the batch.
        passdown : sequence of str
            Components to filter in the batch.
        axis : int
            Axis to project masks to before computing mask area.
        """
        _ = threshold, passdown
        mask = self.get(ix, src)

        reduced = np.max(mask, axis=axis) > 0.0
        return np.sum(reduced) / np.prod(reduced.shape)

    def _post_mask_rebatch(self, areas, *args, src=None, passdown=None, threshold=None, **kwargs):
        #pylint: disable=protected-access, access-member-before-definition, attribute-defined-outside-init
        _ = args, kwargs
        new_index = [self.indices[i] for i, area in enumerate(areas) if area > threshold]
        if len(new_index) > 0:
            self.index = DatasetIndex.from_index(index=new_index)
        else:
            raise SkipBatchException

        passdown = passdown or []
        passdown.extend([src, 'locations', 'shapes', 'generated', 'orientations', 'label_names'])
        passdown = list(set(passdown))

        for compo in passdown:
            new_data = [getattr(self, compo)[i] for i, area in enumerate(areas) if area > threshold]
            setattr(self, compo, np.array(new_data))
        return self


    @action
    @inbatch_parallel(init='_init_component', post='_assemble', target='for')
    def filter_out(self, ix, src=None, dst=None, expr=None, low=None, high=None, length=None, p=1.0):
        """ Zero out mask for horizon extension task.

        Parameters
        ----------
        src : str
            Component of batch with mask
        dst : str
            Component of batch to put cut mask in.
        expr : callable, optional.
            Some vectorized function. Accepts points in cube, returns either float.
            If not None, low or high/length should also be supplied.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if not (src and dst):
            raise ValueError('Src and dst must be provided')

        mask = self.get(ix, src)
        coords = np.where(mask > 0)

        if np.random.binomial(1, 1 - p) or len(coords[0]) == 0 or expr is None:
            new_mask = mask
        else:
            new_mask = np.zeros_like(mask)
            coords = np.array(coords).astype(np.float).T
            cond = np.ones(shape=coords.shape[0]).astype(bool)
            coords /= np.reshape(mask.shape, newshape=(1, 3))
            if low is not None:
                cond &= np.greater_equal(expr(coords), low)
            if high is not None:
                cond &= np.less_equal(expr(coords), high)
            if length is not None:
                low = 0 if not low else low
                cond &= np.less_equal(expr(coords), low + length)
            coords *= np.reshape(mask.shape, newshape=(1, 3))
            coords = np.round(coords).astype(np.int32)[cond]
            new_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = mask[coords[:, 0],
                                                                      coords[:, 1],
                                                                      coords[:, 2]]
        return new_mask


    @apply_parallel
    def shift_masks(self, crop, n_segments=3, max_shift=4, min_len=5, max_len=10):
        """ Randomly shift parts of the crop up or down.

        Parameters
        ----------
        n_segments : int
            Number of segments to shift.
        max_shift : int
            Max size of shift along vertical axis.
        min_len : int
            Min size of shift along horizontal axis.
        max_len : int
            Max size of shift along horizontal axis.
        """
        crop = np.copy(crop)
        for _ in range(n_segments):
            # Point of starting the distortion, its length and size
            begin = np.random.randint(0, crop.shape[1])
            length = np.random.randint(min_len, max_len)
            shift = np.random.randint(-max_shift, max_shift)

            # Apply shift
            segment = crop[:, begin:min(begin + length, crop.shape[1]), :]
            shifted_segment = np.zeros_like(segment)
            if shift > 0:
                shifted_segment[:, :, shift:] = segment[:, :, :-shift]
            elif shift < 0:
                shifted_segment[:, :, :shift] = segment[:, :, -shift:]
            if shift != 0:
                crop[:, begin:min(begin + length, crop.shape[1]), :] = shifted_segment
        return crop

    @apply_parallel
    def bend_masks(self, crop, angle=10):
        """ Rotate part of the mask on a given angle.
        Must be used for crops in (xlines, heights, inlines) format.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees.
        """
        shape = crop.shape
        point_x = np.random.randint(0, shape[0])
        point_h = np.argmax(crop[point_x, :, :])

        if np.sum(crop[point_x, point_h, :]) == 0.0:
            return crop

        matrix = cv2.getRotationMatrix2D((point_h, point_x), angle, 1)
        rotated = cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

        combined = np.zeros_like(crop)
        if point_x >= shape[0]//2:
            combined[:point_x, :, :] = crop[:point_x, :, :]
            combined[point_x:, :, :] = rotated[point_x:, :, :]
        else:
            combined[point_x:, :, :] = crop[point_x:, :, :]
            combined[:point_x, :, :] = rotated[:point_x, :, :]
        return combined

    @apply_parallel
    def linearize_masks(self, crop, n=3, shift=0, kind='random', width=None):
        """ Sample `n` points from the original mask and create a new mask by interpolating them.

        Parameters
        ----------
        n : int
            Number of points to sample.
        shift : int
            Maximum amplitude of random shift along the heights axis.
        kind : {'random', 'linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'}
            Type of interpolation to use. If 'random', then chosen randomly for each crop.
        width : int
            Width of interpolated lines.
        """
        # Parse arguments
        if kind == 'random':
            kind = np.random.choice(['linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'])
        if width is None:
            width = np.sum(crop, axis=2)
            width = int(np.round(np.mean(width[width!=0])))

        # Choose the anchor points
        axis = 1 - np.argmin(crop.shape)
        *nz, _ = np.nonzero(crop)
        min_, max_ = nz[axis][0], nz[axis][-1]
        idx = [min_, max_]

        step = (max_ - min_) // n
        for i in range(0, max_-step, step):
            idx.append(np.random.randint(i, i + step))

        # Put anchors into new mask
        mask_ = np.zeros_like(crop)
        slc = (idx if axis == 0 else slice(None),
               idx if axis == 1 else slice(None),
               slice(None))
        mask_[slc] = crop[slc]
        *nz, y = np.nonzero(mask_)

        # Shift heights randomly
        x = nz[axis]
        y += np.random.randint(-shift, shift + 1, size=y.shape)

        # Sort and keep only unique values, based on `x` to remove width of original mask
        sort_indices = np.argsort(x)
        x, y = x[sort_indices], y[sort_indices]
        _, unique_indices = np.unique(x, return_index=True)
        x, y = x[unique_indices], y[unique_indices]

        # Interpolate points; put into mask
        interpolator = interp1d(x, y, kind=kind)
        indices = np.arange(min_, max_, dtype=np.int32)
        heights = interpolator(indices).astype(np.int32)

        slc = (indices if axis == 0 else indices * 0,
               indices if axis == 1 else indices * 0,
               np.clip(heights, 0, crop.shape[2]-1))
        mask_ = np.zeros_like(crop)
        mask_[slc] = 1

        # Make horizon wider
        structure = np.ones((1, width), dtype=np.uint8)
        shape = mask_.shape
        mask_ = mask_.reshape((mask_.shape[axis], mask_.shape[2]))
        mask_ = cv2.dilate(mask_, kernel=structure, iterations=1).reshape(shape)
        return mask_


    # Predictions
    @action
    @inbatch_parallel(init='indices', post=None, target='for')
    def update_accumulator(self, ix, src, accumulator):
        """ Update accumulator with data from crops.
        Allows to gradually accumulate predicitons in a single instance, instead of
        keeping all of them and assembling later.

        Parameters
        ----------
        src : str
            Component with crops.
        accumulator : Accumulator3D
            Container for cube aggregation.
        """
        crop = self.get(ix, src)
        location = self.get(ix, 'locations')
        if self.get(ix, 'orientations'):
            crop = crop.transpose(1, 0, 2)
        accumulator.update(crop, location)
        return self

    @action
    @inbatch_parallel(init='indices', target='for', post='_masks_to_horizons_post')
    def masks_to_horizons(self, ix, src_masks='masks', dst='predicted_labels',
                          threshold=0.5, mode='mean', minsize=0, mean_threshold=2.0,
                          adjacency=1, skip_merge=False, prefix='predict'):
        """ Convert predicted segmentation mask to a list of Horizon instances.

        Parameters
        ----------
        src_masks : str
            Component of batch that stores masks.
        dst : str/object
            Component of batch to store the resulting horizons.
        threshold, mode, minsize, mean_threshold, adjacency, prefix
            Passed directly to :meth:`Horizon.from_mask`.
        """
        _ = dst, mean_threshold, adjacency, skip_merge

        # Threshold the mask, transpose and rotate the mask if needed
        mask = self.get(ix, src_masks)
        if self.get(ix, 'orientations'):
            mask = np.transpose(mask, (1, 0, 2))

        field = self.get(ix, 'fields')
        origin = [self.get(ix, 'locations')[k].start for k in range(3)]
        horizons = Horizon.from_mask(mask, field=field, origin=origin, threshold=threshold,
                                     mode=mode, minsize=minsize, prefix=prefix)
        return horizons

    def _masks_to_horizons_post(self, horizons_lists, *args, dst=None, skip_merge=False,
                                mean_threshold=2.0, adjacency=1, **kwargs):
        """ Flatten list of lists of horizons, attempting to merge what can be merged. """
        _, _ = args, kwargs
        if dst is None:
            raise ValueError("dst should be initialized with empty list.")

        if skip_merge:
            setattr(self, dst, [hor for hor_list in horizons_lists for hor in hor_list])
            return self

        for horizons in horizons_lists:
            for horizon_candidate in horizons:
                for horizon_target in dst:
                    merge_code, _ = Horizon.verify_merge(horizon_target, horizon_candidate,
                                                         mean_threshold=mean_threshold,
                                                         adjacency=adjacency)
                    if merge_code == 3:
                        merged = Horizon.overlap_merge(horizon_target, horizon_candidate, inplace=True)
                    elif merge_code == 2:
                        merged = Horizon.adjacent_merge(horizon_target, horizon_candidate, inplace=True,
                                                        adjacency=adjacency, mean_threshold=mean_threshold)
                    else:
                        merged = False
                    if merged:
                        break
                else:
                    # If a horizon can't be merged to any of the previous ones, we append it as it is
                    dst.append(horizon_candidate)
        return self


    # More component actions
    @action
    def concat_components(self, src, dst, axis=-1):
        """ Concatenate a list of components and save results to `dst` component.

        Parameters
        ----------
        src : array-like
            List of components to concatenate of length more than one.
        dst : str
            Component of batch to put results in.
        axis : int
            The axis along which the arrays will be joined.
        """
        if len(src) == 1:
            warn("Since `src` contains only one component, concatenation not needed.")

        items = [self.get(None, attr) for attr in src]

        concat_axis_length = sum(item.shape[axis] for item in items)
        final_shape = [*items[0].shape]
        final_shape[axis] = concat_axis_length
        if axis < 0:
            axis = len(final_shape) + axis
        prealloc = np.empty(final_shape, dtype=np.float32)

        length_counter = 0
        slicing = [slice(None) for _ in range(axis + 1)]
        for item in items:
            length_shift = item.shape[axis]
            slicing[-1] = slice(length_counter, length_counter + length_shift)
            prealloc[slicing] = item
            length_counter += length_shift
        setattr(self, dst, prealloc)
        return self

    @action
    def transpose(self, src, order):
        """ Change order of axis. """
        src = [src] if isinstance(src, str) else src
        order = [i+1 for i in order] # Correct for batch items dimension
        for attr in src:
            setattr(self, attr, np.transpose(self.get(component=attr), (0, *order)))
        return self

    @apply_parallel
    def rotate_axes(self, crop):
        """ The last shall be the first and the first last.

        Notes
        -----
        Actions `make_locations`, `load_cubes`, `create_mask` make data in [iline, xline, height] format.
        Since most of the TensorFlow models percieve ilines as channels, it might be convinient
        to change format to [xlines, height, ilines] via this action.
        """
        crop_ = np.swapaxes(crop, 0, 1)
        crop_ = np.swapaxes(crop_, 1, 2)
        return crop_


    # Augmentations
    @apply_parallel
    def add_axis(self, crop):
        """ Add new axis.

        Notes
        -----
        Used in combination with `dice` and `ce` losses to tell model that input is
        3D entity, but 2D convolutions are used.
        """
        return crop[..., np.newaxis]

    @apply_parallel
    def additive_noise(self, crop, scale):
        """ Add random value to each entry of crop. Added values are centered at 0.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution.
        """
        rng = np.random.default_rng()
        noise = scale * rng.standard_normal(dtype=np.float32, size=crop.shape)
        return crop + noise

    @apply_parallel
    def multiplicative_noise(self, crop, scale):
        """ Multiply each entry of crop by random value, centered at 1.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution.
        """
        rng = np.random.default_rng()
        noise = 1 + scale * rng.standard_normal(dtype=np.float32, size=crop.shape)
        return crop * noise

    @apply_parallel
    def cutout_2d(self, crop, patch_shape, n_patches, fill_value=0):
        """ Change patches of data to zeros.

        Parameters
        ----------
        patch_shape : array-like
            Shape or patches along each axis.
        n_patches : number
            Number of patches to cut.
        fill_value : number
            Value to fill patches with.
        """
        rnd = np.random.RandomState(int(n_patches * 100)).uniform
        patch_shape = patch_shape.astype(int)

        copy_ = copy(crop)
        for _ in range(int(n_patches)):
            starts = [int(rnd(crop.shape[ax] - patch_shape[ax])) for ax in range(3)]
            stops = [starts[ax] + patch_shape[ax] for ax in range(3)]
            slices = [slice(start, stop) for start, stop in zip(starts, stops)]
            copy_[tuple(slices)] = fill_value
        return copy_

    @apply_parallel
    def rotate(self, crop, angle, fill_value=0):
        """ Rotate crop along the first two axes. Angles are defined as Tait-Bryan angles and the sequence of
        extrinsic rotations axes is (axis_2, axis_0, axis_1).

        Parameters
        ----------
        angle : float or tuple of floats
            Angles of rotation about each axes (axis_2, axis_0, axis_1). If float, angle of rotation
            about the last axis.
        fill_value : number
            Value to put at empty positions appeared after crop roration.
        """
        angle = angle if isinstance(angle, (tuple, list)) else (angle, 0, 0)
        crop = self._rotate(crop, angle[0], fill_value)
        if angle[1] != 0:
            crop = crop.transpose(1, 2, 0)
            crop = self._rotate(crop, angle[1], fill_value)
            crop = crop.transpose(2, 0, 1)
        if angle[2] != 0:
            crop = crop.transpose(2, 0, 1)
            crop = self._rotate(crop, angle[2], fill_value)
            crop = crop.transpose(1, 2, 0)
        return crop

    def _rotate(self, crop, angle, fill_value):
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), angle, 1)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0]), borderValue=fill_value).reshape(shape)

    @apply_parallel
    def flip(self, crop, axis=0, seed=0.1, threshold=0.5):
        """ Flip crop along the given axis.

        Parameters
        ----------
        axis : int
            Axis to flip along
        """
        rnd = np.random.RandomState(int(seed*100)).uniform
        if rnd() >= threshold:
            return cv2.flip(crop, axis).reshape(crop.shape)
        return crop

    @apply_parallel
    def scale_2d(self, crop, scale):
        """ Zoom in or zoom out along the first two axis.

        Parameters
        ----------
        scale : tuple or float
            Zooming factor for the first two axis.
        """
        scale = scale if isinstance(scale, (list, tuple)) else [scale] * 2
        crop = self._scale(crop, [scale[0], scale[1]])
        return crop

    @apply_parallel
    def scale(self, crop, scale):
        """ Zoom in or zoom out along each axis of crop.

        Parameters
        ----------
        scale : tuple or float
            Zooming factor for each axis.
        """
        scale = scale if isinstance(scale, (list, tuple)) else [scale] * 3
        crop = self._scale(crop, [scale[0], scale[1]])

        crop = crop.transpose(1, 2, 0)
        crop = self._scale(crop, [1, scale[-1]]).transpose(2, 0, 1)
        return crop

    def _scale(self, crop, scale):
        shape = crop.shape
        matrix = np.zeros((2, 3))
        matrix[:, :-1] = np.diag([scale[1], scale[0]])
        matrix[:, -1] = np.array([shape[1], shape[0]]) * (1 - np.array([scale[1], scale[0]])) / 2
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

    @apply_parallel
    def affine_transform(self, crop, alpha_affine=10):
        """ Perspective transform. Moves three points to other locations.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_affine : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_affine*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_affine >= min(shape)//16:
            alpha_affine = min(shape)//16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size]])

        pts2 = pts1 + rnd(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)


        matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    @apply_parallel
    def perspective_transform(self, crop, alpha_persp):
        """ Perspective transform. Moves four points to other four.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_persp : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_persp*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_persp >= min(shape) // 16:
            alpha_persp = min(shape) // 16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size],
                           [center_[0] - square_size, center_[1] + square_size]])

        pts2 = pts1 + rnd(-alpha_persp, alpha_persp, size=pts1.shape).astype(np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    @apply_parallel
    def elastic_transform(self, crop, alpha=40, sigma=4):
        """ Transform indexing grid of the first two axes.

        Parameters
        ----------
        alpha : float
            Maximum shift along each axis.
        sigma : float
            Smoothening factor.
        """
        rng = np.random.default_rng(seed=int(alpha*100))
        shape_size = crop.shape[:2]

        grid_scale = 4
        alpha //= grid_scale
        sigma //= grid_scale
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x.astype(np.float32) + rand_x)
        grid_y = (grid_y.astype(np.float32) + rand_y)

        distorted_img = cv2.remap(crop, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT_101,
                                  interpolation=cv2.INTER_LINEAR)
        return distorted_img.reshape(crop.shape)

    @apply_parallel
    def bandwidth_filter(self, crop, lowcut=None, highcut=None, fs=1, order=3):
        """ Keep only frequences between lowcut and highcut.

        Notes
        -----
        Use it before other augmentations, especially before ones that add lots of zeros.

        Parameters
        ----------
        lowcut : float
            Lower bound for frequences kept.
        highcut : float
            Upper bound for frequences kept.
        fs : float
            Sampling rate.
        order : int
            Filtering order.
        """
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = butter(order, lowcut / nyq, btype='low')
        else:
            b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, crop, axis=1)

    @apply_parallel
    def sign(self, crop):
        """ Element-wise indication of the sign of a number. """
        return np.sign(crop)

    @apply_parallel
    def analytic_transform(self, crop, axis=1, mode='phase'):
        """ Compute instantaneous phase or frequency via the Hilbert transform.

        Parameters
        ----------
        axis : int
            Axis of transformation. Intended to be used after `rotate_axes`, so default value
            is to make transform along depth dimension.
        mode : str
            If 'phase', compute instantaneous phase.
            If 'freq', compute instantaneous frequency.
        """
        analytic = hilbert(crop, axis=axis)
        phase = np.unwrap(np.angle(analytic))

        if mode == 'phase':
            return phase
        if 'freq' in mode:
            return np.diff(phase, axis=axis, prepend=0) / (2*np.pi)
        raise ValueError('Unknown `mode` parameter.')

    @apply_parallel
    def gaussian_filter(self, crop, axis=1, sigma=2, order=0):
        """ Apply a gaussian filter along specified axis. """
        return gaussian_filter1d(crop, sigma=sigma, axis=axis, order=order)

    @apply_parallel
    def central_crop(self, crop, shape):
        """ Central crop of defined shape. """
        crop_shape = np.array(crop.shape)
        shape = np.array(shape)
        if (shape > crop_shape).any():
            raise ValueError(f"shape can't be large then crop shape ({crop_shape}) but {shape} was given.")
        corner = crop_shape // 2 - shape // 2
        slices = tuple(slice(start, start+length) for start, length in zip(corner, shape))
        return crop[slices]

    @apply_parallel
    def translate(self, crop, shift=5, scale=0.0):
        """ Add and multiply values by uniformly sampled values. """
        shift = self.random.uniform(-shift, shift)
        scale = self.random.uniform(1-scale, 1+scale)
        return (crop + shift)*scale

    @action
    def adaptive_expand(self, src, dst=None, channels='first'):
        """ Add channels dimension to 4D components if needed. If component data has shape `(batch_size, 1, n_x, n_d)`,
        it will be keeped. If shape is `(batch_size, n_i, n_x, n_d)` and `n_i > 1`, channels axis
        at position `axis` will be created.
        """
        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst
        axis = 1 if channels in [0, 'first'] else -1
        for _src, _dst in zip(src, dst):
            crop = getattr(self, _src)
            if crop.ndim == 4 and crop.shape[1] != 1:
                crop = np.expand_dims(crop, axis=axis)
            setattr(self, _dst, crop)
        return self

    @action
    def adaptive_squeeze(self, src, dst=None, channels='first'):
        """ Remove channels dimension from 5D components if needed. If component data has shape
        `(batch_size, n_c, n_i, n_x, n_d)` for `channels='first'` or `(batch_size, n_i, n_x, n_d, n_c)`
        for `channels='last'` and `n_c > 1`, shape will be keeped. If `n_c == 1` , channels axis at position `axis`
        will be squeezed.
        """
        dst = dst or src
        src = [src] if isinstance(src, str) else src
        dst = [dst] if isinstance(dst, str) else dst
        axis = 1 if channels in [0, 'first'] else -1
        for _src, _dst in zip(src, dst):
            crop = getattr(self, _src)
            if crop.ndim == 5 and crop.shape[axis] == 1:
                crop = np.squeeze(crop, axis=axis)
            setattr(self, _dst, crop)
        return self

    def plot_components(self, *components, idx=0, slide=None, **kwargs):
        """ Plot components of batch.

        Parameters
        ----------
        components : str or sequence of str
            Components to get from batch and draw.
        idx : int or None
            If int, then index of desired image in list.
            If None, then no indexing is applied.
        slide : slice
            Indexing element for individual images.
        """
        # Get components data
        if idx is not None:
            data = [getattr(self, comp)[idx].squeeze() for comp in components]
        else:
            data = [getattr(self, comp).squeeze() for comp in components]

        if slide is not None:
            data = [item[slide] for item in data]

        # Get location
        l = self.locations[idx]
        field_name = self.unsalt(self.indices[idx])
        displayed_name = self.dataset[field_name].displayed_name

        if (l[0].stop - l[0].start) == 1:
            suptitle = f'INLINE {l[0].start}   CROSSLINES {l[1].start}:{l[1].stop}   DEPTH {l[2].start}:{l[2].stop}'
        elif (l[1].stop - l[1].start) == 1:
            suptitle = f'CROSSLINE {l[1].start}   INLINES {l[0].start}:{l[0].stop}   DEPTH {l[2].start}:{l[2].stop}'
        else:
            suptitle = f'DEPTH {l[2].start}  INLINES {l[0].start}:{l[0].stop}   CROSSLINES {l[1].start}:{l[1].stop}'
        suptitle = f'batch item {idx}                  {displayed_name}\n{suptitle}'

        # Plot parameters
        kwargs = {
            'figsize': (8 * len(components), 8),
            'suptitle_label': suptitle,
            'title': list(components),
            'xlabel': 'xlines',
            'ylabel': 'depth',
            'cmap': ['gray'] + ['viridis'] * len(components),
            'bad_values': (),
            **kwargs
        }
        return plot_image(data, **kwargs)


    def show(self, n=1, separate=True, components=None, **kwargs):
        """ Plot `n` random batch items. """
        available_components = components or ['images', 'masks', 'predictions']
        available_components = [compo for compo in available_components
                                if hasattr(self, compo)]

        n = min(n, len(self))

        for idx in self.random.choice(len(self), size=n, replace=False):
            self.plot_components(*available_components, idx=idx, separate=separate, **kwargs)
