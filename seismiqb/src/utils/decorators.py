""" Collection of decorators. """
from functools import wraps
import numpy as np



def transformable(method):
    """ Transform the output matrix of a function to optionally:
        - put the matrix on a background with spatial shape of a cube
        - change values at absent points to a desired value
        - set dtype of a matrix
        - normalize values
        - reduce dimensionality via PCA transform
        - view data as atleast 3d array.
    By default, does nothing.

    Parameters
    ----------
    on_full : bool
        Whether to put the matrix on full cube spatial-shaped background.
    fill_value : number, optional
        If provided, then used at points where the horizon is absent.
    dtype : numpy dtype, optional
        If provided, then dtype of a matrix is changed, and fill_value at absent points is changed accordingly.
    normalize : bool, str, optional
        If `min-max` or True, then use min-max scaling.
        If `mean-std`, then use mean-std scaling.
        If False, don't scale matrix.
    n_components : number, optional
        If integer, then number of components to keep after PCA transformation.
        If float in (0, 1) range, then amount of variance to be explained after PCA transformation.
    atleast_3d : bool
        Whether to return the view of a resulting array as at least 3-dimensional entity.
    """
    @wraps(method)
    def wrapper(instance, *args, on_full=False, fill_value=None, dtype=None,
                normalize=False, enlarge=False, enlarge_width=10, n_components=None, atleast_3d=False, **kwargs):
        result = method(instance, *args, **kwargs)

        if dtype and hasattr(instance, 'matrix_set_dtype'):
            result = instance.matrix_set_dtype(result, dtype=dtype)
        if on_full and hasattr(instance, 'matrix_put_on_full'):
            result = instance.matrix_put_on_full(result)
        if normalize and hasattr(instance, 'matrix_normalize'):
            result = instance.matrix_normalize(result, normalize)
        if fill_value is not None and hasattr(instance, 'matrix_fill_to_num'):
            result = instance.matrix_fill_to_num(result, value=fill_value)
        if enlarge and hasattr(instance, 'matrix_enlarge'):
            result = instance.matrix_enlarge(result, width=enlarge_width)
        if atleast_3d:
            result = np.atleast_3d(result)
        if n_components and hasattr(instance, 'pca_transform'):
            if result.ndim != 3:
                raise ValueError(f'PCA transformation can be applied only to 3D arrays, got `{result.ndim}`')
            result = instance.pca_transform(result, n_components=n_components)

        return result
    return wrapper
