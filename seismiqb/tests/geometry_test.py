""" Script for running the controller notebook for SeismicGeometry tests.

The behaviour of the test is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
LOGS_DIR : str
    Path to the directory with test logs (timings, message).
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
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
"""
from glob import glob
import json
import os
import pprint
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook


def test_geometry(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True
):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
    LOGS_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')

    # Workspace preparation
    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir("notebooks").mkdir("geometry_test_files")
        _ = OUTPUT_DIR.mkdir("notebooks")
        _ = OUTPUT_DIR.mkdir("tmp")

        out_path_ipynb = OUTPUT_DIR.join(f"geometry_test_out_{DATESTAMP}.ipynb")

    else:
        # Remove outdated executed controller notebook (It is saved near to the original one)
        if REMOVE_OUTDATED_FILES:
            previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_out_*.ipynb'))

            for file in previous_output_files:
                os.remove(file)

        # Create main paths links
        if OUTPUT_DIR is None:
            OUTPUT_DIR = LOGS_DIR

        out_path_ipynb = os.path.join(OUTPUT_DIR, f'geometry_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test.ipynb'),
        nb_kwargs={
            # Workspace constants
            'DATESTAMP': DATESTAMP,
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'LOGS_DIR': LOGS_DIR,
            'OUTPUT_DIR': OUTPUT_DIR,

            # Execution parameters
            'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
            'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
            'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO
        },
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    # Tests exit
    failed, traceback_msg = extract_traceback(path_ipynb=out_path_ipynb)
    failed = failed or (exec_info is not True)

    if not failed:
        # Open message
        message_path = glob(os.path.join(OUTPUT_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()

        # Open timings
        timings_path = glob(os.path.join(OUTPUT_DIR, 'timings_*.json'))[-1]

        with open(timings_path, "r", encoding="utf-8") as infile:
            timings = json.load(infile)

    else:
        timings= {'state': 'FAIL'}

        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg = traceback_msg

        msg += '\nSeismicGeometry tests execution failed.'

    # Provide output message to the terminal
    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            print(''.join(msg))

        pp = pprint.PrettyPrinter()
        pp.pprint(timings)
        print('\n')

        # End of the running message
        if timings['state'] == 'OK' and not failed:
            print('Tests for SeismicGeometry were executed successfully.\n')
        else:
            assert False, 'SeismicGeometry tests failed.\n'
