""" Script for running notebook with SeismicGeometry tests."""
import glob
import json
import os
import pprint
from datetime import date

from ..batchflow.utils_notebook import run_notebook

# Constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
DROP_EXTRA_FILES = True
SHOW_TEST_ERROR_INFO = True
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')
SHOW_MESSAGE = True
GITHUB_MODE = True

def test_geometry(capsys, tmpdir):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks").mkdir("geometry_test_files")
        out_path_ipynb = SAVING_DIR.join(f"geometry_test_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = TEST_DIR
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/geometry_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test.ipynb'),
        nb_kwargs={
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'TEST_DIR': TEST_DIR,
            'DATESTAMP': DATESTAMP,
            'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,
            'SAVING_DIR': SAVING_DIR,
            'GITHUB_MODE': GITHUB_MODE
        },
        insert_pos=1,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if exec_info is True:
        # Open message
        message_path = glob.glob(os.path.join(SAVING_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()

        # Open timings
        timings_path = glob.glob(os.path.join(SAVING_DIR, 'timings_*.json'))[-1]

        with open(timings_path, "r", encoding="utf-8") as infile:
            timings = json.load(infile)

    else:
        msg = ['SeismicGeometry tests execution failed.\n']
        timings= {'state': 'FAIL'}

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            for line in msg:
                print(line)

        pp = pprint.PrettyPrinter()
        pp.pprint(timings)
        print('\n')

        # End of the running message
        if timings['state']=='OK':
            print('Tests for SeismicGeometry were executed successfully.\n')
        else:
            assert False, 'SeismicGeometry tests failed.\n'
