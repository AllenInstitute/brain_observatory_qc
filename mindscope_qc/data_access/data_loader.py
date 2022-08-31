import mindscope_qc.data_access.from_lims as from_lims

def get_mouse_ids_from_id(id_type: str, id_number: int):

    mouse_ids = from_lims._get_mouse_ids_from_id(id_type=id_type, id_number=id_number)
    return mouse_ids