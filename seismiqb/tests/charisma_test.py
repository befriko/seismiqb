""" Script for running the controller notebook for CharismaMixin tests.

The behaviour of the test is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
OUTPUT_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, etc.).

And you can manage test running with parameters:

USE_TMP_OUTPUT_DIR: bool
    Whether to use pytest tmpdir as a workspace.
    If True, then all files are saved in temporary directories.
    If False, then all files are saved in local directories.
REMOVE_OUTDATED_FILES: bool
    Whether to remove outdated files which relate to previous executions.
REMOVE_EXTRA_FILES : bool
    Whether to remove extra files after execution.
    Extra files are temporary files and execution savings that relate to successful tests.
SHOW_MESSAGE : bool
    Whether to show a detailed tests execution message.
SHOW_TEST_ERROR_INFO : bool
    Whether to show error traceback in outputs.
    Notice that it only works with SHOW_MESSAGE = True.

You can also manage notebook execution kwargs which relates to cube and horizon for the test:

CUBE_SHAPE : sequence of three integers
    Shape of a synthetic cube.
SEED: int or None
    Seed used for creation of random generator (check out `np.random.default_rng`).
"""
import glob
import os
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook


def test_charisma(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True
):
    """ Run CharismaMixin tests notebook.

    This test runs ./notebooks/charisma_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube (Field), saves it and checks
    matrices savings and loadings in CHARISMA data format.
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    # Workspace preparation
    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir('notebooks')
        _ = OUTPUT_DIR.mkdir('charisma_tests_files')

        out_path_ipynb = OUTPUT_DIR.join(f"charisma_test_out_{DATESTAMP}.ipynb")

    else:
        # Remove outdated executed controller notebook (It is saved near to the original one)
        if REMOVE_OUTDATED_FILES:
            previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/charisma_test_out_*.ipynb'))

            for file in previous_output_files:
                os.remove(file)

        # Create main paths links
        if OUTPUT_DIR is None:
            OUTPUT_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks', 'charisma_tests_files')

        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/charisma_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/charisma_test.ipynb'),
        nb_kwargs={
            # Workspace constants
            'DATESTAMP': DATESTAMP,
            'OUTPUT_DIR': OUTPUT_DIR,

            # Execution parameters
            'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
            'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
            'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,

             # Data creation parameters
            'CUBE_SHAPE': (100, 100, 100),
            'SEED': 10
        },
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    # Tests exit
    failed, traceback_msg = extract_traceback(path_ipynb=out_path_ipynb)
    failed = failed or (exec_info is not True)

    if not failed:
        msg = ['Tests for CharismaMixin were executed successfully.\n']

    else:
        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg = traceback_msg

        msg.append('\nCharismaMixin tests execution failed.')

    msg = ''.join(msg)

    with capsys.disabled():
        # Output message
        if SHOW_MESSAGE:
            print(msg)

        # End of the running message
        if (msg.find('fail') != -1) or failed:
            assert False, 'CharismaMixin tests failed.\n'
