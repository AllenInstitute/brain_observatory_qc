
from pathlib import PosixPath
import pytest
import pandas as pd
import brain_observatory_qc.data_access.from_lims as from_lims

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


@pytest.mark.onprem
def test_get_cortical_z_stacks_works():
    cortical_z_stacks_fp = from_lims.get_zstack_cortical_filepath(1197160226) # A working specimen as of 11/04/2022
    cortical_z_stacks_fp = list([str(cortical_z_stacks_fp)])

    #got this list manually
    true_z_stack_paths = [
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_AL_cortical_00001.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_AM_cortical_00001.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_LM_cortical_00001.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_V1_cortical_00001.tif', 
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_v1_week2_00001.tif', 
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_al_week2_00001.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_am_week2_00001.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_lm_week2_00002.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_cortical_AL_week3_00001.tif', 
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_cortical_AM_week3_00001.tif', 
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_cortical_LM_week3_00002.tif',
        '//allen/programs/mindscope/production/learning/prod0/specimen_1197160226/637848_cortical_V1_week3_00001.tif'
        ]

    assert sorted(cortical_z_stacks_fp) == sorted(true_z_stack_paths), "Cortical Z stack path lists are not equal"

    