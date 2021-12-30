""" Tests helper functions."""
import nbformat

def find_traceback_in_outputs(cell_info):
    """ Find cell output with a traceback and extract the traceback. """
    outputs = cell_info.get('outputs', [])
    traceback_msg = ""
    has_error_traceback = False

    for output in outputs:
        traceback = output.get('traceback', [])

        if traceback:
            has_error_traceback = True

            for line in traceback:
                traceback_msg += line
            break

    return has_error_traceback, traceback_msg

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
    failed = False
    out_notebook = nbformat.read(path_ipynb, as_version=4)

    if cell_num is not None:
        # Get a traceback from cell directly
        cell_info = out_notebook['cells'][cell_num]

        has_error_traceback, current_traceback_msg = find_traceback_in_outputs(cell_info=cell_info)

        traceback_msg += current_traceback_msg
        failed = failed or has_error_traceback

    else:
        # Find a cell with a traceback
        for cell_info in out_notebook['cells']:
            has_error_traceback, current_traceback_msg = find_traceback_in_outputs(cell_info=cell_info)

            traceback_msg += current_traceback_msg
            failed = failed or has_error_traceback

    return failed, traceback_msg
