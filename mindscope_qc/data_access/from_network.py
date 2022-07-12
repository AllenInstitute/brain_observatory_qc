import os
import pandas as pd

import mindscope_qc.data_access.utilities as utils
import mindscope_qc.data_access.from_lims as from_lims

def get_sync_path(lims_data):
    ophys_session_dir = get_ophys_session_dir(lims_data)

    sync_file = [file for file in os.listdir(ophys_session_dir) if 'sync' in file]
    if len(sync_file) > 0:
        sync_file = sync_file[0]
    else:
        json_path = [file for file in os.listdir(ophys_session_dir) if '_platform.json' in file][0]
        with open(os.path.join(ophys_session_dir, json_path)) as pointer_json:
            json_data = json.load(pointer_json)
            sync_file = json_data['sync_file']
    sync_path = os.path.join(ophys_session_dir, sync_file)
    return sync_path
