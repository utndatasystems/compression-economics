from src.utils import *

import pytest

@pytest.mark.parametrize(
    "input_path, output_path, expected",
    [   (   "/home/hpc/v164be/v164be10/src/compression-economics/text_results_gt.txt",
            "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            False,),
        (   "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            True,),],)
def test_check_mismatch(input_path, output_path, expected):
    assert check_mismatch(input_path, output_path=output_path) is expected

