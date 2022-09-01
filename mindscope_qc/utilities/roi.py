# standard
import pandas as pd
import numpy as np

# allen 
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


import mindscope_qc.data_access.data_loader as data_loader

def roi_count_from_cell_table(expt_list: list) -> pd.DataFrame:
    """Return count of ROIs for each experiment from cell table, merges
        with expt table

        Parameters
        ----------
        expt_list : list
            list of ophys experiment ids

        Returns
        -------
        rc_table : pandas.DataFrame
            table with roi counts for each experiment
        """

    # use cache and get tables
    cache = VisualBehaviorOphysProjectCache.from_lims()
    expt_table_all = cache.get_ophys_experiment_table(passed_only=False)
    cells_table_all = cache.get_ophys_cells_table()
    
    # filtered cell and expt tables
    cells_table = cells_table_all.query("ophys_experiment_id in @expt_list")
    expt_table = expt_table_all.query("ophys_experiment_id in @expt_list")

    #calculate roi counts
    roi_counts = cells_table.ophys_experiment_id.value_counts()
    rc_table = (expt_table.merge(roi_counts, left_on='ophys_experiment_id',
                right_index=True, how='left')
                .rename(columns={'ophys_experiment_id': 'roi_count'}))

    return rc_table

def load_legacy_rois(expt_id: int, excluded_rois: bool = True) -> np.ndarray:
    """Load legacy rois from objectlist.txt
    
    Parameters
    ----------
    expt_id : int
        ophys experiment id
    excluded_rois : bool
        if True, load excluded rois, if False, load included rois

    Returns
    -------
    rois : np.ndarray
        array of roi coordinates
    """

    pf = data_loader.get_motion_preview_filepath(expt_id).parent

    # find folder with segmention in pf
    seg_folders = [f for f in pf.iterdir() if 'segmentation' in f.name]
    seg_folders

    # get most recent folder in segmentation_folder
    seg_folder = max(seg_folders, key=lambda x: x.stat().st_mtime)
    roi_file = seg_folder / 'objectlist.txt'
    print(roi_file)
    legacy_rois = pd.read_csv(roi_file, sep=',',header=0)

    if excluded_rois:
        return legacy_rois[legacy_rois.loc[:," eXcluded"] == 0]
    else:
        return legacy_rois
    
def add_legacy_roi_count_col(expt_table: pd.DataFrame) -> pd.DataFrame:
    """Add column with legacy roi count to expt_table

    Parameters
    ----------
    expt_table : pandas.DataFrame
        table with ophys experiment ids

    Returns
    -------
    expt_table : pandas.DataFrame
        table with legacy roi count column
    """
    expt_table["legacy_roi_count"] = expt_table.apply(lambda x: 
                                     load_legacy_rois(x.name).shape[0],axis=1)
    return expt_table