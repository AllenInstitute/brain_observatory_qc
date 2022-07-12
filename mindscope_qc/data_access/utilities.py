import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime



# warning
gen_depr_str = 'this function is deprecated and will be removed in a future version, ' \
               + 'please use {}.{} instead'

# CONVENIENCE FUNCTIONS TO GET VARIOUS INFORMATION #




class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate


def get_ssim(img0, img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1, gaussian_weights=True)
    return ssim_pair


def get_lims_data(lims_id):
    ld = LimsDatabase(int(lims_id))
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data



def get_roi_group(lims_data):
    experiment_id = int(lims_data.experiment_id.values[0])
    ophys_session_dir = get_ophys_session_dir(lims_data)
    import json
    json_file = [file for file in os.listdir(ophys_session_dir) if ('SPLITTING' in file) and ('input.json' in file)]
    json_path = os.path.join(ophys_session_dir, json_file[0])
    with open(json_path, 'r') as w:
        jin = json.load(w)
    # figure out which roi_group the current experiment belongs to
    # plane_data = pd.DataFrame()
    for i, roi_group in enumerate(range(len(jin['plane_groups']))):
        group = jin['plane_groups'][roi_group]['ophys_experiments']
        for j, plane in enumerate(range(len(group))):
            expt_id = int(group[plane]['experiment_id'])
            if expt_id == experiment_id:
                expt_roi_group = i
    return expt_roi_group


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def correct_filepath(filepath):
    """using the pathlib python module, takes in a filepath from an
    arbitrary operating system and returns a filepath that should work
    for the users operating system

    Parameters
    ----------
    filepath : string
        given filepath

    Returns
    -------
    string
        filepath adjusted for users operating system
    """
    filepath = filepath.replace('/allen', '//allen')
    corrected_path = Path(filepath)
    return corrected_path


def correct_dataframe_filepath(dataframe, column_string):
    """applies the correct_filepath function to a given dataframe
    column, replacing the filepath in that column in place


    Parameters
    ----------
    dataframe : table
        pandas dataframe with the column
    column_string : string
        the name of the column that contains the filepath to be
        replaced

    Returns
    -------
    dataframe
        returns the input dataframe with the filepath in the given
        column 'corrected' for the users operating system, in place
    """
    dataframe[column_string] = dataframe[column_string].apply(lambda x: correct_filepath(x))
    return dataframe


def dateformat(exp_date):
    """
    reformat date of acquisition for accurate sorting by date
    """
    
    date = int(datetime.strptime(exp_date, '%Y-%m-%d  %H:%M:%S.%f').strftime('%Y%m%d'))
    return date

