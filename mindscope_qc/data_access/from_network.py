import os
import sys
from tkinter import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import mindscope_qc.data_access.utilities as utils
import mindscope_qc.data_access.from_lims as lims


### ________________LIMS__STORAGE__DIRECTORIES_____________________ # noqa: E266

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


def get_experiment_storage_directory(ophys_experiment_id: int) -> str:
    """gets the experiment level storage directory filepath for a 
    specific ophys experiment

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath string to the experiment storage directory folder
    """
    directories_df = lims.get_storage_directories_for_id("ophys_experiment_id", ophys_experiment_id)
    experiment_path = directories_df["experiment_storage_directory"][0]
    return experiment_path


def get_session_storage_directory(ophys_session_id: int) -> str:
    """gets the session level storage directory filepath for a 
    specific ophys session

    Parameters
    ----------
    ophys_session_id : int
        unique identifier for an ophys session

    Returns
    -------
    str
        filepath string to the session storage directory folder
    """
    directories_df = lims.get_storage_directories_for_id("ophys_session_id", ophys_session_id)
    session_path = directories_df["session_storage_directory"][0]
    return session_path


def get_container_storage_directory(ophys_container_id: int) -> str:
    """gets the container level storage directory filepath for a 
    specific ophys container
    Parameters
    ----------
    ophys_container_id : int
        unique identifier for an ophys container

    Returns
    -------
    str
        filepath string to the container storage directory folder
    """
    directories_df = lims.get_storage_directories_for_id("ophys_container_id", ophys_container_id)
    container_path = directories_df["container_storage_directory"][0]
    return container_path


def get_crosstalk_storage_directory(ophys_session_id: int) -> str:
    """
    gets the network storage directory for the crosstalk folder
    Usually the crosstalk folder is within the session folder. 

    Parameters
    ----------
    ophys_session_id : int
       unique id for an ophys session

    Returns
    -------
    str
        filepath string to the "crosstalk" folder
    """
    session_directory = get_session_storage_directory(ophys_session_id)
    crosstalk_directory = os.path.join(session_directory, "crosstalk")
    return crosstalk_directory


def get_eye_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    eye_tracking_directory = os.path.join(session_directory, "eye_tracking")
    return eye_tracking_directory


def get_face_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    face_tracking_directory = os.path.join(session_directory, "face_tracking")
    return face_tracking_directory


def get_side_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    side_tracking_directory = os.path.join(session_directory, "side_tracking")
    return side_tracking_directory


def get_demix_storage_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    demix_directory = os.path.join(experiment_directory, "demix")
    return demix_directory


def get_demix_plots_storage_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    demix_plots_directory = os.path.join(experiment_directory, "demix", "demix_plots")
    return demix_plots_directory


def get_neuropil_subtraction_plots_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    np_subtraction_directory = os.path.join(experiment_directory, "demix", "demix_plots")
    return np_subtraction_directory


def get_experiment_processed_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    processed_directory = os.path.join(experiment_directory, "processed")
    return processed_directory


def get_current_cell_segmentation_run_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    segmentation_run_id = lims.get_current_segmentation_run_id(ophys_experiment_id)
    cell_segmentation_run_directory = os.path.join(experiment_directory, "processed", "ophys_cell_segmentation_run_{}".format(segmentation_run_id))
    return cell_segmentation_run_directory


def get_suite2p_registration_summary_png_filepath(ophys_experiment_id: int) -> str:
    """filepath to png of image of boundary of all detected objects
    to see if and how objects overlap.

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    processed_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = '{}__suite2p_registration_summary.png'.format(ophys_experiment_id)
    image_filepath = os.path.join(processed_directory, filename)
    return image_filepath


def load_suite2p_registration_summary_image(ophys_experiment_id: int) -> str:
    image_filepath = get_suite2p_registration_summary_png_filepath(ophys_experiment_id)
    image = mpimg.imread(image_filepath)
    return image


def get_maxInt_boundary_PNG_filepath(ophys_experiment_id: int) -> str:
    """filepath to png of image of boundary of all detected objects
    to see if and how objects overlap.

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    image_filepath = os.path.join(cell_seg_directory, "maxInt_boundary.png")
    return image_filepath


def load_maxInt_boundary_PNG_filepath(ophys_experiment_id: int) -> str:
    image_filepath = get_maxInt_boundary_PNG_filepath(ophys_experiment_id)
    image = mpimg.imread(image_filepath)
    return image


def get_maxInt_masks_TIF_filepath(ophys_experiment_id: int) -> str:
    """gets the filepath for the maxInt_masks.TIF file:
    Segmented ROI file


    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    image_filepath = os.path.join(cell_seg_directory, "maxInt_masks.TIF")
    return image_filepath


def get_maxInt_LOmasks_TIF_filepath(ophys_experiment_id: int) -> str:
    """gets the filepath for the maxInt_LOmasks.TIF file
    multi-layer image contains the masks of detected active cells
    where overlapping pixels of any object with any other object are
    excluded for cleaner trace calculation.

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    image_filepath = os.path.join(cell_seg_directory, "maxInt_masks.TIF")
    return image_filepath


def get_maxInt_a13a_PNG_filepath(ophys_experiment_id: int) -> str:
    """gets the filepath for the maxInt_a13a.png file
    Projection of fine-tuned aligned time sequence.

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    image_filepath = os.path.join(cell_seg_directory, "maxInt_a13a.png")
    return image_filepath


def get_avgInt_a1X_PNG_filepath(ophys_experiment_id: int) -> str:
    """gets the filepath for the avgInt_a1X.png file
    FOV background

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    image_filepath = os.path.join(cell_seg_directory, "avgInt_a1X.png")
    return image_filepath


def get_enhimgseq_TIF_filepath(ophys_experiment_id: int) -> str:
    """gets the filepath for the enhimgseq.TIF file
    produced enhanced image sequence where all segmented objects
    can be found and can be used for validation and visualization of
    detected active cell objects

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    cell_seg_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    image_filepath = os.path.join(cell_seg_directory, "enhimgseq.TIF")
    return image_filepath


