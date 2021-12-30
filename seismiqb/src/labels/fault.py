""" Fault class and processing methods. """

import os
import glob
import warnings

import numpy as np
import pandas as pd

from numba import prange, njit

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import measurements

from ...batchflow.notifier import Notifier

from .horizon import Horizon
from .fault_triangulation import make_triangulation, triangle_rasterization
from .fault_postprocessing import faults_sizes
from ..plotters import show_3d
from ..geometry import SeismicGeometry
from ..utils import concat_sorted, split_array



class Fault(Horizon):
    """ Contains points of fault.

    Initialized from `storage` and `geometry`.

    Storage can be one of:
        - csv-like file in CHARISMA, REDUCED_CHARISMA or FAULT_STICKS format.
        - ndarray of (N, 3) shape.
        - hdf5 file as a binary mask for cube.
    """
    #pylint: disable=attribute-defined-outside-init
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def __init__(self, *args, nodes=None, **kwargs):
        self.nodes = None
        self.sticks = None
        super().__init__(*args, **kwargs)
        if nodes is not None:
            self.from_points(nodes, dst='nodes', reset=None, **kwargs)

    def from_file(self, path, transform=True, verify=True, direction=None, **kwargs):
        """ Init from path to either CHARISMA, REDUCED_CHARISMA or FAULT_STICKS csv-like file
        or from numpy formats.
        """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path)
        ext = os.path.splitext(path)[1][1:]

        if ext == 'npz':
            points, nodes, sticks = self.load_npz(path)
            self.format = 'file-npz'
        elif ext == 'npy':
            points, nodes, sticks = self.load_npy(path)
            self.format = 'file-npy'
        else:
            points, nodes, sticks = self.load_fault_sticks(path, transform, verify, **kwargs)
            self.format = 'file-csv'

        self.from_points(points, verify=False, **kwargs)
        if nodes is not None:
            self.from_points(nodes, dst='nodes', verify=False, reset=None, **kwargs)
        if sticks is not None:
            self.sticks = sticks

        if direction is None:
            if len(self.points) > 0:
                mean_depth = int(self.points[:, 2].mean())
                depth_slice = self.points[self.points[:, 2] == mean_depth]
                self.direction = 0 if depth_slice[:, 0].ptp() > depth_slice[:, 1].ptp() else 1
            else:
                self.direction = 0
        elif isinstance(direction, int):
            self.direction = direction
        elif isinstance(direction[self.field.short_name], int):
            self.direction = direction[self.field.short_name]
        else:
            self.direction = direction[self.field.short_name][self.name]

    def load_npz(self, path):
        """ Load fault points, nodes and sticks from npz file. """
        npzfile = np.load(path, allow_pickle=False)
        points, nodes, stick_labels = npzfile['points'], npzfile['nodes'], npzfile['stick_labels']
        if len(stick_labels) != len(nodes):
            raise ValueError('nodes and stick_labels must be of the same length.')
        if len(nodes) == 0:
            nodes = None
            sticks = None
        else:
            sticks = np.array(split_array(nodes, stick_labels), dtype=object)
        return points, nodes, sticks

    def load_npy(self, path):
        """ Load fault points from npy file. """
        points = np.load(path, allow_pickle=False)
        nodes = None
        sticks = None
        return points, nodes, sticks

    def load_fault_sticks(self, path, transform=True, verify=True, fix=False, **kwargs):
        """ Get point cloud array from file values. """
        df = self.read_file(path)
        if df is None:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.array([])

        if 'cdp_x' in df.columns:
            df = self.recover_lines_from_cdp(df)

        points = df[self.REDUCED_CHARISMA_SPEC].values
        if transform:
            points = self.field_reference.geometry.lines_to_cubic(points)
            df[self.REDUCED_CHARISMA_SPEC] = points

        if verify:
            idx = np.where((points[:, 0] >= 0) &
                        (points[:, 1] >= 0) &
                        (points[:, 2] >= 0) &
                        (points[:, 0] < self.field_reference.shape[0]) &
                        (points[:, 1] < self.field_reference.shape[1]) &
                        (points[:, 2] < self.field_reference.shape[2]))[0]
        df = df.iloc[idx]

        sticks = self.read_sticks(df, fix)
        if len(sticks) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.array([])
        points = self.interpolate_3d(sticks, **kwargs)
        nodes = np.concatenate(sticks.values)
        return points, nodes, sticks.values

    def read_file(self, path):
        """ Read data frame with sticks. """
        with open(path, encoding='utf-8') as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])

        if line_len == 0:
            return None

        if line_len == 3:
            names = self.REDUCED_CHARISMA_SPEC
        elif line_len == 8:
            names = self.FAULT_STICKS
        elif line_len >= 9:
            names = self.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def read_sticks(self, df, fix=False):
        """ Transform initial fault dataframe to array of sticks. """
        if 'number' in df.columns: # fault file has stick index
            col = 'number'
        elif df.iline.iloc[0] == df.iline.iloc[1]: # there is stick points with the same iline
            col = 'iline'
        elif df.xline.iloc[0] == df.xline.iloc[1]: # there is stick points with the same xline
            col = 'xline'
        else:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')
        df = df.sort_values('height')
        sticks = df.groupby(col).apply(lambda x: x[Horizon.COLUMNS].values).reset_index(drop=True)
        if fix:
            # Remove sticks with horizontal parts.
            mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
            if not mask.all():
                warnings.warn(f'{self.name}: Fault has horizontal parts of sticks.')
            sticks = sticks.loc[mask]
            # Remove sticks with one node.
            mask = sticks.apply(len) > 1
            if not mask.all():
                warnings.warn(f'{self.name}: Fault has one-point sticks.')
            sticks = sticks.loc[mask]
            # Filter faults with one stick.
            if len(sticks) == 1:
                warnings.warn(f'{self.name}: Fault has an only one stick')
                sticks = pd.Series()

        #Order sticks with respect of fault direction. Is necessary to perform following triangulation.
        if len(sticks) > 0:
            pca = PCA(1)
            coords = pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values]))
            indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
            return sticks.iloc[indices]
        return sticks

    def interpolate_3d(self, sticks, width=1, **kwargs):
        """ Interpolate fault sticks as a surface. """
        triangles = make_triangulation(sticks)
        points = []
        for triangle in triangles:
            res = triangle_rasterization(triangle, width)
            points += [res]
        return np.concatenate(points, axis=0)

    def add_to_mask(self, mask, locations=None, sparse=False, **kwargs):
        """ Add fault to background. """
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                            [locations[1].start, locations[1].stop],
                            [locations[2].start, locations[2].stop]],
                            dtype=np.int32)
        points = self.points

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
            return mask

        if sparse and self.nodes is not None:
            slides_indices = np.unique(self.nodes[:, self.direction])
            indices = np.isin(points[:, self.direction], slides_indices)
            points = points[indices]
            mask_pos = np.isin(
                np.arange(mask.shape[self.direction]),
                slides_indices - locations[self.direction].start
            )
            if mask_pos.any():
                if self.direction == 0:
                    mask[mask_pos, :] = np.clip(mask[mask_pos, :], 0, 1)
                else:
                    mask[:, mask_pos] = np.clip(mask[:, mask_pos], 0, 1)

        insert_fault_into_mask(mask, points, mask_bbox)
        return mask

    @classmethod
    def check_format(cls, path, verbose=False):
        """ Find errors in fault file.

        Parameters
        ----------
        path : str
            path to file or glob expression
        verbose : bool
            response if file is succesfully readed.
        """
        for filename in glob.glob(path):
            try:
                df = cls.read_file(filename)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': fault file must be splitted.')
                elif len(cls.read_sticks(df)) == 1:
                    print(filename, ': fault has an only one stick')
                elif any(cls.read_sticks(df).apply(len) == 1):
                    print(filename, ': fault has one point stick')
                elif verbose:
                    print(filename, ': OK')

    @classmethod
    def split_file(cls, path, dst):
        """ Split file with multiple faults into separate files. """
        if dst and not os.path.isdir(dst):
            os.makedirs(dst)
        df = pd.read_csv(path, sep=r'\s+', names=cls.FAULT_STICKS)
        df.groupby('name').apply(cls.fault_to_csv, dst=dst)

    def merge(self, other, **kwargs):
        """ Merge two Fault instances"""
        points = concat_sorted(self.points, other.points)
        if self.nodes is not None:
            nodes = concat_sorted(self.nodes, other.nodes)
        else:
            nodes = None
        return Fault(points, nodes=nodes, field=self.field, **kwargs)

    @classmethod
    def fault_to_csv(cls, df, dst):
        """ Save the fault to csv. """
        df.to_csv(os.path.join(dst, df.name), sep=' ', header=False, index=False)

    def dump_points(self, path):
        """ Dump points. """
        path = self.field.make_path(path, name=self.short_name, makedirs=False)

        if os.path.exists(path):
            raise ValueError(f'{path} already exists.')

        points = self.points
        nodes = self.nodes if self.nodes is not None else np.zeros((0, 3), dtype=np.int32)
        sticks = self.sticks if self.sticks is not None else []
        stick_labels = sum([[i] * len(item) for i, item in enumerate(sticks)], [])

        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        np.savez(path, points=points, nodes=nodes, stick_labels=stick_labels) # TODO: what about allow_pickle?

    def split_faults(self, **kwargs):
        """ Split file with faults points into separate connected faults.

        Parameters
        ----------
        **kwargs
            Arguments for `split_faults` function.
        """
        array = np.zeros(self.field.shape)
        array[self.points[:, 0], self.points[:, 1], self.points[:, 2]] = 1
        return self.from_mask(array, cube_shape=self.field.shape, field=self.field, **kwargs)

    def show_3d(self, n_sticks=100, n_nodes=10, z_ratio=1., zoom_slice=None, show_axes=True,
                width=1200, height=1200, margin=20, savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        n_sticks : int
            Number of sticks for each fault.
        n_nodes : int
            Number of nodes for each stick.
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom_slice : tuple of slices or None.
            Crop from cube to show. If None, the whole cube volume will be shown.
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
        title = f'Fault `{self.name}` on `{self.field.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')
        if zoom_slice is None:
            zoom_slice = [slice(0, i) for i in self.field.shape]
        zoom_slice[-1] = slice(self.h_min, self.h_max)
        margin = [margin] * 3 if isinstance(margin, int) else margin
        x, y, z, simplices = self.make_triangulation(n_sticks, n_nodes, zoom_slice)

        show_3d(x, y, z, simplices, title, zoom_slice, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)

    def make_triangulation(self, n_sticks, n_nodes, slices, **kwargs):
        """ Create triangultaion of fault.

        Parameters
        ----------
        n_sticks : int
            Number of sticks to create.
        n_nodes : int
            Number of nodes for each stick.
        slices : tuple
            Region to process.

        Returns
        -------
        x, y, z, simplices
            `x`, `y` and `z` are numpy.ndarrays of triangle vertices, `simplices` is (N, 3) array where each row
            represent triangle. Elements of row are indices of points that are vertices of triangle.
        """
        points = self.points.copy()
        for i in range(3):
            points = points[points[:, i] <= slices[i].stop]
            points = points[points[:, i] >= slices[i].start]
        if len(points) <= 3:
            return None, None, None, None
        sticks = get_sticks(points, n_sticks, n_nodes)
        simplices = make_triangulation(sticks, True)
        coords = np.concatenate(sticks)
        return coords[:, 0], coords[:, 1], coords[:, 2], simplices

    @classmethod
    def from_mask(cls, array, field=None, chunk_size=None, threshold=None, overlap=1, pbar=False,
                  cube_shape=None, fmt='mask'):
        """ Label faults in an array.

        Parameters
        ----------
        array : numpy.ndarray or SeismicGeometry
            binary mask of faults or array of coordinates.
        field : Field or None
            Where the fault is.
        chunk_size : int
            size of chunks to apply `measurements.label`.
        threshold : float or None
            threshold to drop small faults.
        overlap : int
            size of overlap to join faults from different chunks.
        pbar : bool
            progress bar
        cube_shape : tuple
            shape of cube. If fmt='mask', can be infered from array.
        fmt : str
            if 'mask', array is a binary mask of faults. If 'points', array consists of coordinates of fault points.

        Returns
        -------
        numpy.ndarray
            array of shape (n_faults, ) where each item is array of fault points of shape (N_i, 3).
        """
        # TODO: make chunks along xlines
        if isinstance(array, SeismicGeometry):
            array = array.file['cube_i']
        chunk_size = chunk_size or len(array)
        if chunk_size == len(array):
            overlap = 0

        if cube_shape is None and fmt == 'points':
            raise ValueError("If fmt='points', cube_shape must be specified")

        cube_shape = cube_shape or array.shape

        if fmt == 'mask':
            chunks = [(start, array[start:start+chunk_size]) for start in range(0, cube_shape[0], chunk_size-overlap)]
            total = len(chunks)
        else:
            def _chunks():
                for start in range(0, cube_shape[0], chunk_size-overlap):
                    chunk = np.zeros((chunk_size, *cube_shape[1:]))
                    points = array[array[:, 0] < start+chunk_size]
                    points = points[points[:, 0] >= start]
                    chunk[points[:, 0]-start, points[:, 1], points[:, 2]] = 1
                    yield (start, chunk)
            chunks = _chunks()
            total = len(range(0, cube_shape[0], chunk_size-overlap))

        prev_overlapped_labels = None
        labels = np.zeros((0, 4), dtype='int32')
        n_objects = 0

        for start, item in Notifier(pbar, total=total)(chunks):
            chunk_labels, new_objects = measurements.label(item, structure=np.ones((3, 3, 3))) # labels for new chunk
            new_labels = np.where(chunk_labels)
            new_labels = np.stack([*new_labels, chunk_labels[new_labels] + n_objects], axis = -1)

            overlapped_labels = new_labels[new_labels[:, 0] < overlap, 3]
            if prev_overlapped_labels is not None:
                # while there are the same objects with different labels repeat procedure
                while (overlapped_labels != prev_overlapped_labels).any():
                    # find overlapping objects and change labels in new chunk
                    transform = {k: v for k, v in zip(overlapped_labels, prev_overlapped_labels) if k != v}

                    for k, v in transform.items():
                        new_labels[new_labels[:, 3] == k, 3] = v
                    overlapped_labels = new_labels[new_labels[:, 0] < overlap, 3]
                    transform = {k: v for k, v in zip(prev_overlapped_labels, overlapped_labels) if k != v}

                    # find overlapping objects and change labels in processed part of cube
                    for k, v in transform.items():
                        labels[labels[:, 3] == k, 3] = v
                    prev_overlapped_labels = labels[labels[:, 0] >= start - overlap + 1, 3]

            if start != 0:
                new_labels = new_labels[new_labels[:, 0] >= overlap]

            new_labels[:, 0] += start
            labels = np.concatenate([labels, new_labels])
            prev_overlapped_labels = labels[labels[:, 0] >= start + item.shape[0] - overlap, 3]
            n_objects += new_objects

        labels = labels[np.argsort(labels[:, 3])]
        labels = np.array(split_array(labels[:, :-1], labels[:, 3]), dtype=object)
        sizes = faults_sizes(labels)
        labels = sorted(zip(sizes, labels), key=lambda x: x[0], reverse=True)
        if threshold:
            labels = [item for item in labels if item[0] >= threshold]
        if field is not None:
            labels = [Fault(item[1].astype('int32'), name=f'fault_{i}', field=field)
                      for i, item in Notifier(pbar)(enumerate(labels))]
        return labels

def get_sticks(points, n_sticks, n_nodes):
    """ Get sticks from fault which is represented as a cloud of points.

    Parameters
    ----------
    points : np.ndarray
        Fault points.
    n_sticks : int
        Number of sticks to create.
    n_nodes : int
        Number of nodes for each stick.

    Returns
    -------
    numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    """
    pca = PCA(1)
    pca.fit(points)
    axis = 0 if np.abs(pca.components_[0][0]) > np.abs(pca.components_[0][1]) else 1

    column = points[:, 0] if axis == 0 else points[:, 1]
    step = max((column.max() - column.min()) // (n_sticks + 1), 1)

    points = points[np.argsort(points[:, axis])]
    projections = np.split(points, np.unique(points[:, axis], return_index=True)[1][1:])[::step]

    res = []

    for p in projections:
        points_ = thicken_line(p).astype(int)
        loc = p[0, axis]
        if len(points_) > 3:
            nodes = approximate_points(points_[:, [1-axis, 2]], n_nodes)
            nodes_ = np.zeros((len(nodes), 3))
            nodes_[:, [1-axis, 2]] = nodes
            nodes_[:, axis] = loc
            res += [nodes_]
    return res

def thicken_line(points):
    """ Make thick line. """
    points = points[np.argsort(points[:, -1])]
    splitted = split_array(points, points[:, -1])
    return np.stack([np.mean(item, axis=0) for item in splitted], axis=0)

def approximate_points(points, n_points):
    """ Approximate points by stick. """
    pca = PCA(1)
    array = pca.fit_transform(points)

    step = (array.max() - array.min()) / (n_points - 1)
    initial = np.arange(array.min(), array.max() + step / 2, step)
    indices = np.unique(nearest_neighbors(initial.reshape(-1, 1), array.reshape(-1, 1), 1))
    return points[indices]

def nearest_neighbors(values, all_values, n_neighbors=10):
    """ Find nearest neighbours for each `value` items in `all_values`. """
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(all_values)
    return nn.kneighbors(values)[1].flatten()

@njit(parallel=True)
def insert_fault_into_mask(mask, points, mask_bbox):
    """ Add new points into binary mask. """
    #pylint: disable=not-an-iterable
    for i in prange(len(points)):
        point = points[i]
        if (point[0] >= mask_bbox[0][0]) and (point[0] < mask_bbox[0][1]):
            if (point[1] >= mask_bbox[1][0]) and (point[1] < mask_bbox[1][1]):
                if (point[2] >= mask_bbox[2][0]) and (point[2] < mask_bbox[2][1]):
                    mask[point[0] - mask_bbox[0][0], point[1] - mask_bbox[1][0], point[2] - mask_bbox[2][0]] = 1
