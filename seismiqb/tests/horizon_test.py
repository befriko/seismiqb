""" Script for running the controller notebook for Horizon tests.

The behaviour of the test is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
OUTPUT_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, horizons etc.).

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

SYNTHETIC_MODE : bool
    Whether to create a synthetic data (cube and horizon) or use existed, provided by paths.
CUBE_PATH : str or None
    Path to an existed seismic cube.
    Notice that it is only used with SYNTHETIC_MODE = False.
HORIZON_PATH : str or None
    Path to an existed seismic horizon.
    Notice that it is only used with SYNTHETIC_MODE = False.
CUBE_SHAPE : sequence of three integers
    Shape of a synthetic cube.
GRID_SHAPE: sequence of two integers
    Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
SEED: int or None
    Seed used for creation of random generator (check out `np.random.default_rng`).

Visualizations in saved execution notebooks are controlled with:

FIGSIZE : sequence of two integers
    Figures width and height in inches.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.
"""
from glob import glob
import os
from shutil import rmtree
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook


def test_horizon(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True
):
    """ Run Horizon test notebook.

    This test runs ./notebooks/horizon_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube with horizon, saves them
    and runs Horizon tests notebooks (base, manipulations, attributes).
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    # Workspace preparation
    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir("notebooks").mkdir("horizon_test_files")
        _ = OUTPUT_DIR.mkdir('tmp')

        out_path_ipynb = OUTPUT_DIR.join(f"horizon_test_out_{DATESTAMP}.ipynb")

    else:
        # Remove outdated executed controller notebook (It is saved near to the original one)
        if REMOVE_OUTDATED_FILES:
            previous_output_files = glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_out_*.ipynb'))

            for file in previous_output_files:
                os.remove(file)

        # Create main paths links
        if OUTPUT_DIR is None:
            OUTPUT_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_files/')

        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/horizon_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test.ipynb'),
        nb_kwargs={
            # Workspace constants
            'DATESTAMP': DATESTAMP,
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'OUTPUT_DIR': OUTPUT_DIR,

            # Execution parameters
            'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
            'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
            'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,

            # Synthetic creation parameters
            'SYNTHETIC_MODE': True,
            'CUBE_PATH': None,
            'HORIZON_PATH': None,
            'CUBE_SHAPE': (500, 500, 200),
            'GRID_SHAPE': (10, 10),
            'SEED': 42,

            # Visualization parameters
            'FIGSIZE': (12, 7),
            'SHOW_FIGURES': False
        },
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    # Tests exit
    if exec_info is True:
        # Open message
        message_path = glob(os.path.join(OUTPUT_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()

    else:
        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg = extract_traceback(path_ipynb=out_path_ipynb)

        msg.append('\nHorizon tests execution failed.')

    msg = ''.join(msg)

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            print(msg)

        # End of the running message
        if exec_info is True and msg.find('fail')==-1:
            # Clear directory with extra files
            if not USE_TMP_OUTPUT_DIR and REMOVE_EXTRA_FILES:
                try:
                    rmtree(OUTPUT_DIR)
                except OSError as e:
                    print(f"Can't delete the directory {OUTPUT_DIR} : {e.strerror}")

        else:
            assert False, 'Horizon tests failed.\n'
