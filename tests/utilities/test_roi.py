import pytest
import pandas as pd
import mindscope_qc.utilities.roi as roi
@pytest.mark.onprem
def test_load_legacy_rois():
    expected_data_filepath = '//allen/programs/mindscope/production/learning/prod0/specimen_1187454009/ophys_session_1197920532/ophys_experiment_1198067556/processed/ophys_cell_segmentation_run_1198081388/objectlist.txt'
    expected_result = pd.read_csv(expected_data_filepath, sep=',',header=0)
    expected_result = expected_result[expected_result.loc[:," eXcluded"] == 0]
    result = roi.load_legacy_rois(1198067556)
    pd.testing.assert_frame_equal(result,expected_result)