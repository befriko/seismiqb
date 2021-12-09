""" Tests helper functions."""
import nbformat

def find_traceback_in_outputs(cell_info):
    """ Find cell output with a traceback and extract the traceback. """
    outputs = cell_info.get('outputs', [])
    traceback_msg = ""

    for output in outputs:
        traceback = output.get('traceback', [])

        if traceback:
            for line in traceback:
                traceback_msg += line
            break

    return traceback_msg

def extract_traceback(path_ipynb, cell_num=None):
    """ Extracts error traceback from tests notebooks.

    Parameters
    ----------
    path_ipynb: str
        Path to a notebook from which extract error traceback.
    cell_num: int or None
        A number of an error cell.
        If None, than we will find cell_num iterating over notebook.
        If notebook doesn't contain markdown cells than `cell_num` equals to `exec_info`
        from `seismiqb.batchflow.utils_notebook.run_notebook` in error case.
    """
    traceback_msg = "TRACEBACK: \n"
    out_notebook = nbformat.read(path_ipynb, as_version=4)

    if cell_num is not None:
        # Get a traceback from cell directly
        cell_info = out_notebook['cells'][cell_num]
        traceback_msg += find_traceback_in_outputs(cell_info=cell_info)

    else:
        # Find a cell with a traceback
        for cell_info in out_notebook['cells']:
            traceback_msg += find_traceback_in_outputs(cell_info=cell_info)

    return traceback_msg
