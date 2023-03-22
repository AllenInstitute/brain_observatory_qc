import brain_observatory_qc.data_access.from_lims as from_lims
import pandas as pd


def get_lims_psql_dict_cursor():
    return from_lims._get_psql_dict_cursor()


def generic_lims_query(query: str) -> pd.DataFrame:
    result = from_lims._generic_lims_query(query=query)
    return result


def get_mouse_ids_from_id(id_type: str, id_number: int):
    mouse_ids = from_lims._get_mouse_ids_from_id(id_type=id_type, id_number=id_number)
    return mouse_ids


def get_id_type(id_number: int) -> str:
    id_type = from_lims._get_id_type(id_number=id_number)
    return id_type


def general_id_type_query(id_type: str, id_number: int):
    result = from_lims._general_id_type_query(id_type=id_type, id_number=id_number)
    return result


# #################################Get Filepaths


def get_motion_preview_filepath(ophys_experiment_id):
    filepath = from_lims._get_motion_preview_filepath(ophys_experiment_id=ophys_experiment_id)
    return filepath
