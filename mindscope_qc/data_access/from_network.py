import os
import numpy as np
import matplotlib.image as mpimg
from os.path import exists as file_exists
import mindscope_qc.data_access.from_lims as lims


######################################################
#             UTILITIY FUNCTIONS
######################################################


def get_filepath(storage_directory: str, filename: str) -> str:
    filepath = os.path.join(storage_directory, filename)
    if file_exists(filepath) is False:
        no_file_error = "Error: {} does not exist".format(filepath)
        print(no_file_error)
    else:
        return filepath


def load_image(image_filepath: str) -> np.ndarray:
    image = mpimg.imread(image_filepath)
    return image


######################################################
#             STORAGE DIRECTORIES
######################################################


def get_specimen_storage_directory(specimen_id: int) -> str:
    directories_df = lims.get_storage_directories_for_id("specimen_id", specimen_id)
    specimen_path = directories_df["specimen_storage_directory"][0]
    return specimen_path


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
    processed_directory = get_experiment_processed_directory(ophys_experiment_id)
    current_segmentation_run_id = lims.get_current_segmentation_run_id(ophys_experiment_id)
    current_cell_segmentation_run_directory = os.path.join(processed_directory, "ophys_cell_segmentation_run_{}".format(current_segmentation_run_id))
    return current_cell_segmentation_run_directory


######################################################
#             SPECIMEN LEVEL FILES
######################################################

def get_post_surgical_photodoc_PNG_filepath(specimen_id: int) -> str:
    storage_directory = get_specimen_storage_directory(specimen_id)
    filename = '1_{}-0000.png'.format(specimen_id)
    image_filepath = get_filepath(storage_directory, filename)
    return image_filepath


def load_post_surgical_photodoc_image(specimen_id: int) -> np.ndarray:
    image_filepath = get_post_surgical_photodoc_PNG_filepath(specimen_id)
    image = load_image(image_filepath)
    return image


######################################################
#             SESSION LEVEL FILES
######################################################


def get_stimulus_PKL_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}.pkl'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_platform_JSON_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}_platform.json'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_reticle_TIF_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}_reticle.tif'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_reticle_image(ophys_session_id: int) -> np.ndarray:
    image_filepath = get_reticle_TIF_filepath(ophys_session_id)
    image = load_image(image_filepath)
    return image


def get_vasculature_TIF_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}_vasculature.tif'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_vasculature_image(ophys_session_id: int) -> np.ndarray:
    image_filepath = get_vasculature_TIF_filepath(ophys_session_id)
    image = load_image(image_filepath)
    return image


def get_vasculature_downsampled_TIF_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}_vasculature_downsampled.tif'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_vasculature_downsampled_image(ophys_session_id: int) -> np.ndarray:
    image_filepath = get_vasculature_downsampled_TIF_filepath(ophys_session_id)
    image = load_image(image_filepath)
    return image


def get_cortical_zstack_TIFF_filepath(ophys_session_id: int) -> str:
    storage_directory = get_session_storage_directory(ophys_session_id)
    filename = '{}_cortical_z_stack.tiff'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


######################################################
#             EYE TRACKING FILES
######################################################


def get_ellipse_H5_filepath(ophys_session_id: int) -> str:
    storage_directory = get_eye_tracking_storage_directory(ophys_session_id)
    filename = '{}_ellipse.h5'.format(ophys_session_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


######################################################
#             EXPERIMENT LEVEL FILES
######################################################

def get_experiment_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_surface_TIF_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}_surface.tif'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_surface_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_surface_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


def get_depth_TIF_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}_depth.tif'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_depth_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_depth_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


def get_dff_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}_dff.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath

def get_event_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}_event.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_local_zstack_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_storage_directory(ophys_experiment_id)
    filename = '{}_z_stack_local.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_neuropil_traces_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = 'neuropil_traces.h5'
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_roi_traces_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = 'roi_traces.h5'
    filepath = get_filepath(storage_directory, filename)
    return filepath


######################################################
#             MOTION CORRECTION FILES
######################################################


def get_suite2p_rigid_motion_transform_CSV_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = '{}_suite2p_rigid_motion_transform.csv'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_suite2p_maximum_projection_PNG_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = '{}_suite2p_maximum_projection.png'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_suite2p_maximum_projection_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_suite2p_maximum_projection_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


def get_suite2p_average_projection_PNG_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = '{}_suite2p_average_projection.png'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_suite2p_average_projection_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_suite2p_average_projection_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


def get_suite2p_motion_output_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = '{}_suite2p_motion_output.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_suite2p_registration_summary_PNG_filepath(ophys_experiment_id: int) -> str:
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
    storage_directory = get_experiment_processed_directory(ophys_experiment_id)
    filename = '{}__suite2p_registration_summary.png'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_suite2p_registration_summary_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_suite2p_registration_summary_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


######################################################
#             DEMIX FILES
######################################################


def get_demix_traces_H5_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_demix_storage_directory(ophys_experiment_id)
    filename = '{}_demixed_traces.h5'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_input_roi_demixing_JSON_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_demix_storage_directory(ophys_experiment_id)
    filename = '{}_input_roi_demixing.json'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


def get_output_roi_demixing_JSON_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_demix_storage_directory(ophys_experiment_id)
    filename = '{}_output_roi_demixing.json'.format(ophys_experiment_id)
    filepath = get_filepath(storage_directory, filename)
    return filepath


######################################################
#             CELL SEGMENTATION FILES
######################################################


def get_avgInt_TIF_filepath(ophys_experiment_id: int) -> str:
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "avgInt.tif"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_avgInt_image_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_avgInt_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "maxInt_boundary.png"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_maxInt_boundary_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_maxInt_boundary_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "maxInt_masks.TIF"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_maxInt_masks_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_maxInt_masks_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "maxInt_masks.TIF"
    filepath =get_filepath(storage_directory, filename)
    return filepath


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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename =  "maxInt_a13a.png"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_maxInt_a13a_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_maxInt_a13a_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "avgInt_a1X.png"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_avgInt_a1X_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_avgInt_a1X_PNG_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image 


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
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id) 
    filename = "enhimgseq.TIF"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_enhimgseq_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_enhimgseq_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image


def get_maxInt_masks2_TIF_filepath(ophys_experiment_id: int) -> str:
    """

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    str
        filepath
    """
    storage_directory = get_current_cell_segmentation_run_directory(ophys_experiment_id)
    filename = "maxInt_masks2.TIF"
    filepath = get_filepath(storage_directory, filename)
    return filepath


def load_maxInt_masks2_image(ophys_experiment_id: int) -> np.ndarray:
    image_filepath = get_maxInt_masks2_TIF_filepath(ophys_experiment_id)
    image = load_image(image_filepath)
    return image
