""" A mixin with field visualizations. """
#pylint: disable=global-variable-undefined
import re
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from .viewer import FieldViewer
from ..utils import DelegatingList, to_list
from ..plotters import plot_image, MatplotlibPlotter, show_3d
from ..labels.horizon_attributes import AttributesMixin



COLOR_GENERATOR = iter(MatplotlibPlotter.MASK_COLORS)
NAME_TO_COLOR = {}



class VisualizationMixin:
    """ Methods for field visualization: textual, 2d along various axis, 2d interactive, 3d. """
    # Textual representation
    def __repr__(self):
        return f"""<Field `{self.displayed_name}` at {hex(id(self))}>"""

    REPR_MAX_LEN = 100
    REPR_MAX_ROWS = 5

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ' and labels:' if self.labels else ''
        msg = f'Field `{self.displayed_name}` with {processed_prefix}processed geometry{labels_prefix}\n'

        for label_src in self.loaded_labels:
            labels = getattr(self, label_src)
            names = [label.short_name for label in labels]

            labels_msg = ''
            line = f'    - {label_src}: ['
            while names:
                line += names.pop(0)

                if names:
                    line += ', '
                else:
                    labels_msg += line
                    break

                if len(line) > self.REPR_MAX_LEN:
                    labels_msg += line
                    line = '\n         ' + ' ' * len(label_src)

                if len(labels_msg) > self.REPR_MAX_LEN * self.REPR_MAX_ROWS:
                    break

            if names:
                labels_msg += f'\n         {" "*len(label_src)}and {len(names)} more item(s)'
            labels_msg += ']\n'
            msg += labels_msg
        return msg

    # 2D along axis
    def show_slide(self, loc, width=None, axis='i', zoom_slice=None,
                   src_geometry='geometry', src_labels='labels', indices='all', **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        width : int
            Horizon thickness. If None given, set to 1% of seismic slide height.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        """
        axis = self.geometry.parse_axis(axis)

        # Load seismic and mask
        seismic_slide = getattr(self, src_geometry).load_slide(loc=loc, axis=axis)

        src_labels = src_labels if isinstance(src_labels, (tuple, list)) else [src_labels]
        masks = []
        for src in src_labels:
            masks.append(self.make_mask(location=loc, axis=axis, src=src, width=width, indices=indices))
        mask = sum(masks)

        # src_labels = src_labels if isinstance(src_labels, (tuple, list)) else [src_labels]
        # masks = []
        # for src in src_labels:
        #     masks.extend(getattr(self, src).load_slide(loc=loc, axis=axis, width=width))
        # mask = sum(masks)

        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xmin, xmax, ymin, ymax = 0, seismic_slide.shape[0], seismic_slide.shape[1], 0

        if zoom_slice:
            seismic_slide = seismic_slide[zoom_slice]
            mask = mask[zoom_slice]
            xmin = zoom_slice[0].start or xmin
            xmax = zoom_slice[0].stop or xmax
            ymin = zoom_slice[1].stop or ymin
            ymax = zoom_slice[1].start or ymax

        # defaults for plotting if not supplied in kwargs
        header = self.geometry.axis_names[axis]
        total = self.geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = self.geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.geometry.index_headers[0]
            ylabel = self.geometry.index_headers[1]
            total = self.geometry.depth

        title = f'Field `{self.displayed_name}`\n {header} {loc} out of {total}'

        kwargs = {
            'title_label': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': False,
            'labeltop': False,
            'labelright': False,
            'curve_width': width,
            'grid': [False, True],
            'colorbar': [True, False],
            **kwargs
        }
        return plot_image(data=[seismic_slide, mask], **kwargs)


    # 2D depth slice
    def show_points(self, src='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.spatial_shape)
        denum = np.zeros(self.spatial_shape)

        for label in getattr(self, src):
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src)[0]).__name__
        kwargs = {
            'title_label': f'{labels_class}s on {self.displayed_name}',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'cmap': ['Reds', 'black'],
            'alpha': [1.0, 0.4],
            'colorbar': True,
            **kwargs
        }
        return plot_image([map_, self.zero_traces], **kwargs)


    # 2D top-view maps
    def show(self, attributes='snr', mode='imshow', title_pattern='{attributes} of {label_name}',
             bbox=False, savepath=None, return_figure=False, load_kwargs=None, **plot_kwargs):
        """ Show one or more field attributes on one figure.

        Parameters
        ----------
        attributes : str, np.ndarray, dict or sequence of them
            Attributes to display.
            If str, then use `:meth:.load_attribute` to load the data. For example, `geometry/snr`, `labels:0/depths`.
            If instead of label index contains `:*`, for example, `labels:*/amplitudes`, then run this method
            for each of the objects in `labels` attribute.
            If np.ndarray, then directly used as data to display.
            If dict, then should define either string or np.ndarray (and used the same as previous types),
            as well as other parameters for `:meth:.load_attribute`.
            If sequence of them, then either should be a list to display loaded entities one over the other,
            or nested list to define separate axis and overlaying for each of them.
            For more details, refer to `:func:plot_image`.
        mode : 'imshow' or 'hist'
            Mode to display images.
        title_pattern : str with key substrings to be replaced by corresponding variables values
            If {src_label} in pattern, replaced by name of labels source (e.g. 'horizons:0').
            If {label_name} in pattern, replaced by label name (e.g. 'predicted_#3.char').
            If {attributes} in pattern, replaced by list of attributes names (e.g. '['depths', 'amplitudes']').
            If multiple labels displayed on single subplot, pattern will be repeated in title for every one of them.
        bbox : bool
            Whether crop horizon by its bounding box or not.
        savepath : str, optional
            Path to save the figure. `**` is changed to a field base directory, `*` is changed to field base name.
        return_figure : bool
            Whether to return the figure.
        load_kwargs : dict
            Loading parameters common for every requested attribute.
        plot_kwargs : dict
            Additional parameters for plot creation.

        Examples
        --------
        Simplest possible plot of a geometry-related attribute:
        >>> field.show('mean_matrix')

        Display attribute of a fan over the geometry map:
        >>> field.show(['mean_matrix', 'fans:0/mask'])

        Display attributes on separate axis:
        >>> field.show(['mean_matrix', 'horizons:0/fourier', custom_data_array], separate=True)

        Use various parameters for each of the plots:
        >>> field.show([{'src': 'labels:0/fourier', 'window': 20, 'normalize': True},
                        {'src': 'labels:0/fourier', 'window': 40, 'n_components': 3}],
                       separate=True)

        Display amplitudes and gradients for each of the horizons in a field:
        >>> field.show(['horizons:*/amplitudes', 'horizons:*/gradient'], separate=True)

        Display several attributes on multiple axes with overlays and save it near the cube:
        >>> field.show(['geometry/std_matrix', 'horizons:3/amplitudes',
                        ['horizons:3/instant_phases', 'fans:3/mask'],
                        ['horizons:3/instant_phases', predicted_mask]],
                       savepath='~/IMAGES/complex.png')
        """
        # Wrap given attributes load parameters in a structure that allows applying functions to its nested items
        load_params = DelegatingList(attributes)

        # Prepare data loading params
        load_params = load_params.apply(self._make_load_params, common_params=load_kwargs)

        # Extract names of labels sources that require wildcard loading
        detect_wildcard = lambda params: params['src_labels'] if params['label_num'] == '*' else []
        labels_require_wildcard_loading = load_params.apply(detect_wildcard).flat

        # If any attributes require wildcard loading, run `show` for every label item
        if any(labels_require_wildcard_loading):
            figures = []

            reference_labels_source = labels_require_wildcard_loading[0]
            n_items = len(getattr(self, reference_labels_source))
            for label_num in range(n_items):
                #pylint: disable=cell-var-from-loop
                substitutor = lambda params: {**params, 'src': params['src'].replace('*', str(label_num))}
                label_attributes = load_params.apply(substitutor)

                fig = self.show(attributes=label_attributes, mode=mode, bbox=bbox, title_pattern=title_pattern,
                                savepath=savepath, return_figure=return_figure, load_kwargs=load_kwargs, **plot_kwargs)
                figures.append(fig)

            return figures if return_figure else None

        data_params = load_params.apply(self._load_data)

        # Prepare default plotting parameters
        plot_params = data_params.apply(self._make_plot_params, mode=mode).to_dict()
        plot_params['suptitle'] = f'Field `{self.displayed_name}`'

        if mode == 'imshow':
            plot_params['colorbar'] = True
            plot_params['xlabel'] = self.index_headers[0]
            plot_params['ylabel'] = self.index_headers[1]

        if title_pattern:
            plot_params['title'] = data_params.apply(self._make_title, shallow=True, title_pattern=title_pattern)

        if bbox:
            bboxes_list = data_params.apply(lambda params: params['bbox'])
            lims_list = [np.stack([bboxes]).transpose(1, 2, 0) for bboxes in bboxes_list]
            plot_params['xlim'] = [(lims[0, 0].min(), lims[0, 1].max()) for lims in lims_list]
            plot_params['ylim'] = [(lims[1, 1].max(), lims[1, 0].min()) for lims in lims_list]

        if savepath:
            first_label_name = data_params.reference_object['label_name']
            plot_params['savepath'] = self.make_path(savepath, name=first_label_name)

        # Plot image with given params and return resulting figure
        plot_params = {**plot_params, **plot_kwargs}
        figure = plot_image(mode=mode, **plot_params)
        plt.show()

        return figure if return_figure else None

    # Auxilary methods utilized by `show`
    ALIAS_TO_ATTRIBUTE = AttributesMixin.ALIAS_TO_ATTRIBUTE

    def _make_load_params(self, attribute, common_params):
        # Transform load parameters into dict if needed, extract string indicating data source to use
        if isinstance(attribute, str):
            params = {'src': attribute}
        elif isinstance(attribute, np.ndarray):
            params = {'src': 'user data', 'data': attribute}
        elif isinstance(attribute, dict):
            params = attribute
        else:
            raise TypeError(f'Attribute should be either str, dict or array! Got {type(attribute)} instead.')

        # Extract source labels names and attribute names, detect if any labels sources require wildcard loading,
        # i.e. loading of data for every label stored in requested attribute (e.g. 'horizons:*/depths')
        attribute_name, label_num, src_labels = (re.split(':([0-9, *]+)/', params['src'])[::-1] + ['', 'geometry'])[:3]
        params['attribute_name'] = self.ALIAS_TO_ATTRIBUTE.get(attribute_name, attribute_name)
        params['src_labels'] = src_labels
        params['label_num'] = label_num

        # Make data loading defaults
        default_params = {'dtype': np.float32}

        if params['attribute_name'] in ['fourier_decomposition', 'wavelet_decomposition']:
            default_params['n_components'] = 1

        if params['attribute_name'] in ['full_binary_matrix']:
            default_params['fill_value'] = 0

        # Merge defaults with provided parameters
        params = {**default_params, **(common_params or {}), **params}

        return params

    def _load_data(self, load_params):
        params = {'attribute_name': load_params.pop('attribute_name'),
                  'src_labels': load_params.pop('src_labels'),
                  'label_num': load_params.pop('label_num')}

        postprocess = load_params.pop('postprocess', lambda x: x)

        if 'data' not in load_params:
            data, label = self.load_attribute(_return_label=True, **load_params)
            params['label_name'] = label.displayed_name
            params['bbox'] = label.bbox[:2]
        else:
            data = load_params['data']
            params['label_name'] = self.displayed_name
            params['bbox'] = np.array([[0, max] for max in data.shape])

        params['data'] = postprocess(data.squeeze())

        return params

    CMAP_TO_ATTRIBUTE = {
        'Depths': ['full_matrix'],
        'Reds': ['spikes', 'quality_map', 'quality_grid'],
        'Metric': ['metric']
    }
    ATTRIBUTE_TO_CMAP = {attr: cmap for cmap, attributes in CMAP_TO_ATTRIBUTE.items()
                         for attr in attributes}

    def _make_plot_params(self, data_params, mode):
        params = {'data': data_params['data']}

        src_labels = data_params['src_labels']
        attribute_name = data_params['attribute_name']

        # Choose default cmap
        if attribute_name == 'full_binary_matrix' or mode in ['hist', 'histogramm']:
            global_name = f"{src_labels}/{attribute_name}"
            if global_name not in NAME_TO_COLOR:
                NAME_TO_COLOR[global_name] = next(COLOR_GENERATOR)
            cmap = NAME_TO_COLOR[global_name]
        else:
            cmap = self.ATTRIBUTE_TO_CMAP.get(attribute_name, 'Basic')

        params['cmap'] = cmap

        # Choose default alpha
        if attribute_name in ['full_binary_matrix']:
            alpha = 0.7
        else:
            alpha = 1.0

        params['alpha'] = alpha

        return params

    def _make_title(self, data_params, title_pattern):
        linkage = defaultdict(list)

        for params in to_list(data_params):
            src_label = params['src_labels']
            if params['label_num']:
                src_label += ':' + params['label_num']
            label_name = params['label_name']

            linkage[(src_label, label_name)].append(params['attribute_name'])

        title = ''

        for (src_label, label_name), attributes in linkage.items():
            title += '\n' * (title != '')
            part = title_pattern
            part = part.replace('{src_label}', src_label)
            part = part.replace('{label_name}', label_name)
            part = part.replace('{attributes}', ','.join(attributes))
            title += part

        return title

    # 2D interactive
    def viewer(self, figsize=(8, 8), **kwargs):
        """ Interactive field viewer. """
        return FieldViewer(field=self, figsize=figsize, **kwargs)


    # 3D interactive
    def show_3d(self, src='labels', aspect_ratio=None, zoom_slice=None,
                 n_points=100, threshold=100, n_sticks=100, n_nodes=10,
                 slides=None, margin=(0, 0, 20), colors=None, **kwargs):
        """ Interactive 3D plot for some elements of a field.
        Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `n_sticks` and `n_nodes` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube if needed

        Parameters
        ----------
        src : str, Horizon-instance or list
            Items to draw, by default, 'labels'. If item of list (or `src` itself) is str, then all items of
            that dataset attribute will be drawn.
        aspect_ratio : None, tuple of floats or Nones
            Aspect ratio for each axis. Each None in the resulting tuple will be replaced by item from
            `(geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)`.
        zoom_slice : tuple of slices or None
            Crop from cube to show. By default, the whole cube volume will be shown.
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger height differences in vertices.
        n_sticks : int
            Number of sticks for each fault.
        n_nodes : int
            Number of nodes for each stick.
        slides : list of tuples
            Each tuple is pair of location and axis to load slide from seismic cube.
        margin : tuple of ints
            Added margin for each axis, by default, (0, 0, 20).
        colors : dict or list
            Mapping of label class name to color defined as str, by default, all labels will be shown in green.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : number
            Size of the image.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        src = src if isinstance(src, (tuple, list)) else [src]
        coords = []
        simplices = []

        if zoom_slice is None:
            zoom_slice = [slice(0, s) for s in self.shape]
        else:
            zoom_slice = [
                slice(item.start or 0, item.stop or stop) for item, stop in zip(zoom_slice, self.shape)
            ]
        zoom_slice = tuple(zoom_slice)
        triangulation_kwargs = {
            'n_points': n_points,
            'threshold': threshold,
            'n_sticks': n_sticks,
            'n_nodes': n_nodes,
            'slices': zoom_slice
        }

        labels = [getattr(self, src_) if isinstance(src_, str) else [src_] for src_ in src]
        labels = sum(labels, [])

        if colors is None:
            colors = ['green' for label in labels]
        if isinstance(colors, dict):
            colors = [colors.get(type(label).__name__, colors.get('all', 'green')) for label in labels]

        simplices_colors = []
        for label, color in zip(labels, colors):
            x, y, z, simplices_ = label.make_triangulation(**triangulation_kwargs)
            if x is not None:
                simplices += [simplices_ + sum([len(item) for item in coords])]
                simplices_colors += [[color] * len(simplices_)]
                coords += [np.stack([x, y, z], axis=1)]

        simplices = np.concatenate(simplices, axis=0)
        coords = np.concatenate(coords, axis=0)
        simplices_colors = np.concatenate(simplices_colors)
        title = self.displayed_name

        default_aspect_ratio = (self.shape[0] / self.shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (self.index_headers[0], self.index_headers[1], 'DEPTH')

        images = []
        if slides is not None:
            for loc, axis in slides:
                image = self.geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom_slice[1:]]
                elif axis == 1:
                    image = image[zoom_slice[0], zoom_slice[-1]]
                else:
                    image = image[zoom_slice[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom_slice, simplices_colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)
