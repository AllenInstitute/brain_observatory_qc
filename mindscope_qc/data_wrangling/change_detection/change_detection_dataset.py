import pandas as pd

import mindscope_qc.data_wrangling.change_detection.utils as utils
import mindscope_qc.data_wrangling.change_detection.pkl_translator as pkl_translator
import mindscope_qc.data_wrangling.change_detection.annotate as annotate
import mindscope_qc.data_wrangling.change_detection.extended_trials_dataframe as ext_trials_df

import mindscope_qc.data_access.from_lims as from_lims


class ChangeDetectionDataset(object):
    def __init__(self, data_file=None, behavior_session_id=None,
                 catch_hack: bool = False):
        """Class for loading and manipulating change detection data, 
        using legacy "foraging" style data structures

        Parameters
        ----------
        data_file : str, optional
            path to data file, by default None
        behavior_session_id : int, optional
            behavior session id, by default None
        catch_hack : bool, optional
            whether to use the catch hack, by default False
        """
        
        self.behavior_session_id = behavior_session_id
        self.catch_hack = catch_hack
        self.raw_data = None
        self.core_data = None
        self.extended_trials_df = None

        if data_file:
            self.data_file = data_file
        elif behavior_session_id:
            func = from_lims.get_stimulus_pkl_filepath_for_behavior_session
            self.data_file = func(behavior_session_id)
        
        self._read_raw_data()
        self._annotate_licks()
        self._create_extended_dataframe()

    
    def _read_raw_data(self):
        self.raw_data = pd.read_pickle(self.data_file)
        if self.catch_hack:
            self.core_data = utils.catch_trials_pkl_hack(self.raw_data)
        self.core_data = pkl_translator.data_to_change_detection_core(self.raw_data)

        
    def _annotate_licks(self):
        self.core_data["licks"] = annotate.annotate_licks(self.core_data)


    def _create_extended_dataframe(self):
        self.extended_trials_df = ext_trials_df.create_extended_dataframe(self.core_data)
    

    # GETTER METHODS
    def get_clean_data(self):
        return self.clean_data

    # get core data
    def get_core_data(self):
        return self.core_data

    # get raw data
    def get_raw_data(self):
        return self.raw_data

    # get data file
    def get_data_file(self):
        return self.data_file

    # get extended trials dataframe
    def get_extended_trials_df(self):
        return self.extended_trials_df