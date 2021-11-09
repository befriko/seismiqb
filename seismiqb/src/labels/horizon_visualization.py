""" Mixin for horizon visualization. """
from functools import partial
from textwrap import dedent

import numpy as np
from scipy.spatial import Delaunay

from ..plotters import plot_image, show_3d
from ..utils import filter_simplices



class VisualizationMixin:
    """ Methods for textual and visual representation of a horizon. """
    #pylint: disable=protected-access
    def __repr__(self):
        return f"""<Horizon `{self.name}` for `{self.field.displayed_name}` at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.field.displayed_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Depth range:       {self.h_min} to {self.h_max}
        Depth mean:        {self.h_mean:.6}
        Depth std:         {self.h_std:.6}

        Length:            {len(self)}
        Perimeter:         {self.perimeter}
        Coverage:          {self.coverage:3.5}
        Solidity:          {self.solidity:3.5}
        Num of holes:      {self.number_of_holes}
        """

        if self.is_carcass:
            msg += f"""
        Unique ilines:     {self.carcass_ilines}
        Unique xlines:     {self.carcass_xlines}
        """
        return dedent(msg)


    # 2D
    def find_self(self):
        """ Get reference to the instance in a field.
        If it was loaded/added correctly, then it should be one of `loaded_labels`.
        Otherwise, we add it in a fake attribute and remove later.
        """
        for src in self.field.loaded_labels:
            labels = getattr(self.field, src)

            if isinstance(labels, list):
                for idx, label in enumerate(labels):
                    if label is self:
                        return f'{src}:{idx}'

        # Instance is not attached to a field: add it temporarily (clean-up later)
        self.field._unknown_label = self
        self.field.loaded_labels.append('_unknown_label')
        return '_unknown_label'

    @staticmethod
    def _show_add_prefix(attribute, prefix=None):
        if isinstance(attribute, str):
            attribute = ('/'.join([prefix, attribute])).replace('//', '/')
        elif isinstance(attribute, dict):
            attribute['src'] = ('/'.join([prefix, attribute['src']])).replace('//', '/')
        return attribute


    def show(self, attributes='depths', mode='imshow', short_title=True, return_figure=False, **kwargs):
        """ Field visualization with custom naming scheme. """
        prefix = self.find_self()
        add_prefix = partial(self._show_add_prefix, prefix=prefix)
        attributes = self.field.apply_nested(add_prefix, attributes)

        kwargs = {
            'suptitle_label': f'`{self.name}` on field `{self.field.displayed_name}`',
            **kwargs
        }
        figure = self.field.show(attributes=attributes, mode=mode, short_title=short_title,
                        return_figure=return_figure, **kwargs)

        # Clean-up
        if self.field.loaded_labels[-1] == '_unknown_label':
            self.field._unknown_label = None
            self.field.loaded_labels.pop(-1)

        return figure if return_figure else None



    def show_slide(self, loc, width=None, axis='i', zoom_slice=None, **kwargs):
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
        # Make `locations` for slide loading
        axis = self.field.geometry.parse_axis(axis)

        # Load seismic and mask
        seismic_slide = self.field.geometry.load_slide(loc=loc, axis=axis)
        mask = self.load_slide(loc=loc, axis=axis, width=width)
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
        header = self.field.axis_names[axis]
        total = self.field.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = self.field.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.field.index_headers[0]
            ylabel = self.field.index_headers[1]
            total = self.field.depth

        title = f'Horizon `{self.name}` on cube `{self.field.displayed_name}`\n {header} {loc} out of {total}'

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

    # 3D
    def show_3d(self, n_points=100, threshold=100., z_ratio=1., zoom_slice=None, show_axes=True,
                width=1200, height=1200, margin=(0, 0, 100), savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : int
            Threshold to remove triangles with bigger height differences in vertices.
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom_slice : tuple of slices
            Crop from cube to show.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : int
            Size of the image.
        margin : int
            Added margin from below and above along height axis.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Horizon `{self.short_name}` on `{self.field.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')
        if zoom_slice is None:
            zoom_slice = [slice(0, i) for i in self.field.shape]
        zoom_slice[-1] = slice(self.h_min, self.h_max)

        x, y, z, simplices = self.make_triangulation(n_points, threshold, zoom_slice)

        show_3d(x, y, z, simplices, title, zoom_slice, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)

    def make_triangulation(self, n_points, threshold, slices, **kwargs):
        """ Create triangultaion of horizon.

        Parameters
        ----------
        n_points: int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        slices : tuple
            Region to process.

        Returns
        -------
        x, y, z, simplices
            `x`, `y` and `z` are np.ndarrays of triangle vertices, `simplices` is (N, 3) array where each row
            represent triangle. Elements of row are indices of points that are vertices of triangle.
        """
        _ = kwargs
        weights_matrix = self.full_matrix.astype(np.float32)

        grad_i = np.diff(weights_matrix, axis=0, prepend=0)
        grad_x = np.diff(weights_matrix, axis=1, prepend=0)
        weights_matrix = (grad_i + grad_x) / 2
        weights_matrix[np.abs(weights_matrix) > 100] = np.nan

        idx = np.stack(np.nonzero(self.full_matrix > 0), axis=0)
        mask_1 = (idx <= np.array([slices[0].stop, slices[1].stop]).reshape(2, 1)).all(axis=0)
        mask_2 = (idx >= np.array([slices[0].start, slices[1].start]).reshape(2, 1)).all(axis=0)
        mask = np.logical_and(mask_1, mask_2)
        idx = idx[:, mask]

        probs = np.abs(weights_matrix[idx[0], idx[1]].flatten())
        probs[np.isnan(probs)] = np.nanmax(probs)
        indices = np.random.choice(len(probs), size=n_points, p=probs / probs.sum())

        # Convert to meshgrid
        ilines = self.points[mask, 0][indices]
        xlines = self.points[mask, 1][indices]
        ilines, xlines = np.meshgrid(ilines, xlines)
        ilines = ilines.flatten()
        xlines = xlines.flatten()

        # Remove from grid points with no horizon in it
        heights = self.full_matrix[ilines, xlines]
        mask = (heights != self.FILL_VALUE)
        x = ilines[mask]
        y = xlines[mask]
        z = heights[mask]

        # Triangulate points and remove some of the triangles
        tri = Delaunay(np.vstack([x, y]).T)
        simplices = filter_simplices(simplices=tri.simplices, points=tri.points,
                                     matrix=self.full_matrix, threshold=threshold)
        return x, y, z, simplices
