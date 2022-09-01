
from pathlib import PosixPath
import pytest
import pandas as pd
import mindscope_qc.data_access.from_lims as from_lims

@pytest.mark.onprem
def test_get_mouse_ids_from_id():
    expected = pd.DataFrame(data={'donor_id': [1189413031], 'labtracks_id': ['634402'], 'specimen_id': [1189413042]})
    mouse_ids = from_lims._get_mouse_ids_from_id("external_specimen_name", 634402)

    pd.testing.assert_frame_equal(expected, mouse_ids)

@pytest.mark.onprem
def test_get_mouse_ids_from_id__return_type_dataframe():
    mouse_ids = from_lims._get_mouse_ids_from_id("external_specimen_name", 634402)
    assert type(mouse_ids) == pd.DataFrame

@pytest.mark.onprem
def test_get_mouse_ids_from_id__string_id_works():
    expected = pd.DataFrame(data={'donor_id': [1189413031], 'labtracks_id': ['634402'], 'specimen_id': [1189413042]})
    mouse_ids = from_lims._get_mouse_ids_from_id("external_specimen_name", "634402")

    pd.testing.assert_frame_equal(expected, mouse_ids)

@pytest.mark.onprem
def test_get_mouse_ids_from_id__wrong_id_type_raises_error():
    with pytest.raises(AssertionError, match=r".*input value WRONG_ID_TYPE is not in MOUSE_IDS_DICT keys..*"):
        mouse_ids = from_lims._get_mouse_ids_from_id("WRONG_ID_TYPE", "634402")

@pytest.mark.onprem
def test_get_mouse_ids_from_id__unknown_id_number_returns_empty_dataframe():
    mouse_ids = from_lims._get_mouse_ids_from_id("external_specimen_name", "000000000")
    assert mouse_ids.empty


@pytest.mark.onprem
def test_get_motion_preview_filepath():
    expected = PosixPath('//allen/programs/mindscope/production/learning/prod0/specimen_1187454009/ophys_session_1197920532/ophys_experiment_1198067556/processed/1198067556_suite2p_motion_preview.webm')
    filepath = from_lims._get_motion_preview_filepath(1198067556)
    assert filepath == expected

@pytest.mark.onprem
def test_get_motion_preview_filepath__raises_error_with_incorrect_input():
    with pytest.raises(AssertionError, match=r".*Incorrect id type. Entered Id type is.*correct id type is ophys_experiment_id.*"):
        filepath = from_lims._get_motion_preview_filepath(000000000)

