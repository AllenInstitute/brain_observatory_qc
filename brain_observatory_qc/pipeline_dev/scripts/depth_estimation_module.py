import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pystackreg import StackReg
import os
import h5py
import scipy
import skimage
import cv2
import imageio
import tifffile
from PIL import Image
from pathlib import Path

from visual_behavior.data_access import from_lims
# Migrate dependency from visual_behavior to brain_observatory_qc when all necessary functions are implemented in the brain_observatory_qc
from visual_behavior.data_access import from_lims_utilities
from visual_behavior import database as db

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    from skimage.feature import register_translation as phase_cross_correlation

if os.name == 'nt':
    global_base_dir = Path(
        r'\\allen\programs\mindscope\workgroups\learning\ophys\zdrift'.replace('\\', '/'))
else:
    global_base_dir = Path('/allen/programs/mindscope/workgroups/learning/ophys/zdrift')

###############################################################################
# General tools
###############################################################################


def image_normalization_uint8(image, im_thresh=0):
    """Normalize 2D image and convert to uint8
    Prevent saturation.

    Args:
        image (np.ndarray): input image (2D)
                            Just works with 3D data as well.
        im_thresh (float, optional): threshold when calculating pixel intensity percentile.
                            0 by default
    Return:
        norm_image (np.ndarray)
    """
    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    uint8_image = ((norm_image + 0.05) * np.iinfo(np.uint8).max * 0.9).astype(np.uint8)
    return uint8_image


def image_normalization_uint16(image, im_thresh=0):
    """Normalize 2D image and convert to uint16
    Prevent saturation.

    Args:
        image (np.ndarray): input image (2D)
                            Just works with 3D data as well.
        im_thresh (float, optional): threshold when calculating pixel intensity percentile.
                            0 by default
    Return:
        norm_image (np.ndarray)
    """
    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    uint16_image = ((norm_image + 0.05) * np.iinfo(np.uint16).max * 0.9).astype(np.uint16)
    return uint16_image


def moving_average(data1d, window=10):
    ''' Moving average
    Input:
        data1d (1d array): data to smooth
        window (int): window for smoothing.
    Output:
        avg_data (1d array): smoothed data
    '''

    assert len(data1d.shape) == 1
    avg_data = np.zeros(len(data1d))
    for i in range(len(data1d)):
        min_idx = np.max([i - window // 2, 0])
        max_idx = np.min([i + window // 2, len(data1d)])
        avg_data[i] = np.mean(data1d[min_idx:max_idx])
    return avg_data


def get_frame_start_end(movie_len, segment_len):
    """Get frame start and end of each segment,
    given the length of movie and that of a segment.

    Parameters
    ----------
    movie_len : int
        number of frames of the whole movie
    segment_len : int
        number of frames per segment

    Returns
    -------
    list
        [frame start, frame end] for each segment
    """
    dividers = range(0, movie_len, segment_len)
    frame_start_end = []
    for i in range(len(dividers) - 2):
        frame_start_end.append([dividers[i], dividers[i + 1]])
    if movie_len // segment_len == int(np.round(movie_len / segment_len)):
        # the last segment is appended
        frame_start_end.append([dividers[-2], movie_len])
    else:
        # the last segment as independent segment
        frame_start_end.append([dividers[-2], dividers[-1]])
        frame_start_end.append([dividers[-1], movie_len])
    return frame_start_end


def im_blend(image, overlay, alpha):
    """Blend two images to show match or discrepancy

    Parameters
    ----------
    image : np.ndimage(2d)
        base image
    overlay : np.ndimage(2d)
        image to overlay on top of the base image
    alpha : float ([0,1])
        alpha value for blending

    Returns
    -------
    np.ndimage (2d)
        blended image
    """
    assert len(image.shape) == 2
    assert len(overlay.shape) == 2
    assert image.shape == overlay.shape
    img_uint8 = image_normalization_uint8(image)
    img_rgb = (np.dstack((img_uint8, np.zeros_like(
        img_uint8), img_uint8))) * (1 - alpha)
    overlay_uint8 = image_normalization_uint8(overlay)
    overlay_rgb = np.dstack(
        (np.zeros_like(img_uint8), overlay_uint8, overlay_uint8)) * alpha
    blended = img_rgb + overlay_rgb
    return blended

###############################################################################
# General LIMS tools
###############################################################################


def get_motion_correction_crop_xy_range(oeid):
    """Get x-y ranges to crop motion-correction frame rolling

    Parameters
    ----------
    oeid : int
        ophys experiment ID

    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    # TODO: validate in case where max < 0 or min > 0 (if there exists an example)
    motion_df = pd.read_csv(from_lims.get_motion_xy_offset_filepath(oeid))
    max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    range_y = [-min_y, -max_y]
    range_x = [-min_x, -max_x]
    return range_y, range_x


def check_correct_data(ophys_experiment_id, correct_image_size=(512, 512)):
    """ Check the dimension of experiment images

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    correct_Ly : int, optional
        dimension in y, by default 512
    correct_Lx : int, optional
        dimension in x, by default 512

    Returns
    -------
    bool
        if the data dimension is correct
    """
    correct_Ly = correct_image_size[0]
    correct_Lx = correct_image_size[1]
    Ly, Lx = get_image_size(ophys_experiment_id)
    if (Ly == correct_Ly) and (Lx == correct_Lx):
        return True
    else:
        return False


def get_image_size(ophys_experiment_id):
    """Get image size

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    tuple (2 entries)
        dimension of the image
    """
    try:
        raw_h5_fn = get_raw_h5_filepath(ophys_experiment_id)
    except Exception:
        Ly, Lx = (0, 0)
        return Ly, Lx
    if os.path.isfile(raw_h5_fn):
        with h5py.File(raw_h5_fn, 'r') as h:
            if 'data' in h.keys():
                Ly, Lx = (h['data'].shape[1:])
            else:
                Ly, Lx = (0, 0)
    else:
        Ly, Lx = (0, 0)
    return Ly, Lx


def get_correct_frame_rate(ophys_experiment_id):
    """ Getting the correct frame rate, rather than fixed 11.0 from the metadata

    Args:
        ophys_experiment_id (int): ophys experiment ID

    Returns:
        float: frame rate
    """
    # TODO: change from_lims_utilities from vba to that in brain_observatory_qc.
    # brain_observatory_qc currently does not seem to have tools for getting timestamps.
    lims_data = from_lims_utilities.utils.get_lims_data(
        ophys_experiment_id)  # not yet implemented in brain_observatory_qc
    timestamps = from_lims_utilities.utils.get_timestamps(
        lims_data)  # not yet implemented in brain_observatory_qc
    frame_rate = 1 / np.mean(np.diff(timestamps.ophys_frames.timestamps))
    return frame_rate, timestamps


def find_first_experiment_id_from_ophys_container_id(ophys_container_id, valid_only=True, correct_image_size=(512, 512)):
    """Find the first experiment ID from the ophys container ID.
    Assume experiment IDs are assigned chronologically.

    Parameters
    ----------
    ophys_container_id : int
        ophys container ID
    valid_only : bool, optional
        whether to find only the valid experiment, by default True
        For now, only looking at the frame size (either 1024x512 or 512x512)
            but in the future, various flags should be considered
    correct_image_size : tuple, optional
        Tuple for correct image size, (y,x), by default (512,512)

    Returns
    -------
    int
        the first experiment ID
    """
    # Get all ophys_experiment_ids from the ophys container id
    oeid_df = from_lims.get_ophys_experiment_ids_for_ophys_container_id(
        ophys_container_id)

    # There are random errors where first_oeid is not assigned.

    # # Pick the first one after sorting the ids (assume experiment ids are generated in ascending number)
    # oeids = np.sort(oeid_df.ophys_experiment_id.values)
    # if valid_only:
    #     for oeid in oeids:
    #         if check_correct_data(oeid):
    #             first_oeid = oeid
    #             break
    # else:
    #     first_oeid = oeids[0]

    # Attach correctness to the DataFrame first, gather all experiments with correct data, and sort them.

    if valid_only:
        correct_data = []
        for oeid in oeid_df.ophys_experiment_id.values:
            correct_data.append(check_correct_data(
                oeid, correct_image_size=correct_image_size))
        oeid_df['correct_data'] = correct_data
        oeids = np.sort(
            oeid_df[oeid_df.correct_data].ophys_experiment_id.values)
    else:
        oeids = np.sort(oeid_df.ophys_experiment_id.values)
    first_oeid = oeids[0]

    return first_oeid


def find_first_experiment_id_from_ophys_experiment_id(ophys_experiment_id, valid_only=True, correct_image_size=(512, 512)):
    """Find the first experiment ID from the ophys experiment ID.
    Assume experiment IDs are assigned chronologically.

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    valid_only : bool, optional
        whether to find only the valid experiment, by default True
        For now, only looking at the frame size (either 1024x512 or 512x512)
            but in the future, various flags should be considered
    correct_image_size : tuple, optional
        Tuple for correct image size, (y,x), by default (512,512)

    Returns
    -------
    int
        the first experiment ID
    """
    # Get ophys container id from ophys_experiment id
    ophys_container_id = from_lims.get_ophys_container_id_for_ophys_experiment_id(
        ophys_experiment_id)
    first_oeid = find_first_experiment_id_from_ophys_container_id(
        ophys_container_id, valid_only=valid_only, correct_image_size=correct_image_size)
    return first_oeid


def check_if_ophys_experiment_id_in_ophys_container_id(ophys_experiment_id, ophys_container_id):
    """ Check if ophys experiment ID is in the ophys container ID

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    ophys_container_id : int
        ophys container ID

    Returns
    -------
    bool
        True if ophys experiment ID is in ophys container ID
    """
    # Get ophys container id from ophys_experiment id
    ophys_container_id_from_ophys_experiment_id = from_lims.get_ophys_container_id_for_ophys_experiment_id(
        ophys_experiment_id)
    return ophys_container_id_from_ophys_experiment_id == ophys_container_id

##########################################
# Query tools
##########################################


def get_ophys_session_ids_for_mouse_id(mouse_id):
    """Get ophys session IDs for mouse ID

    Parameters
    ----------
    mouse_id : int
        mouse ID

    Returns
    -------
    DataFrame
        ophys session Ids for the mouse ID
    """
    q_str = f"SELECT os.id \
    FROM ophys_sessions os \
    JOIN specimens sp ON os.specimen_id = sp.id \
    WHERE sp.external_specimen_name = '{mouse_id}'"
    lims_results = db.lims_query(q_str)
    return lims_results


def get_region_depth_ophys_experiment_id(ophys_experiment_id):
    """Get targeted region and depths for an ophys experiment ID

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    DataFrame
        containing targeted region and depth
    """
    q_str = f'SELECT oe.id, st.acronym, oe.calculated_depth, imd.depth \
    FROM ophys_experiments oe \
    JOIN structures st ON st.id = oe.targeted_structure_id \
    JOIN imaging_depths imd ON imd.id = oe.imaging_depth_id \
    WHERE oe.id = {ophys_experiment_id}'
    lims_results = db.lims_query(q_str)
    return lims_results.acronym[0], lims_results.depth[0]


def get_oeid_region_depth_for_ophys_session_id(ophys_session_id):
    """Get targeted region and depths for the ophys experiment IDs within an ophys session ID

    Parameters
    ----------
    ophys_session_id : int
        ophys session ID

    Returns
    -------
    DataFrame
        containing ophys experiment IDs and their corresponding targeted region and depth
    """
    oeid_df = from_lims.get_ophys_experiment_ids_for_ophys_session_id(
        ophys_session_id)
    target_region = []
    depths = []
    for i in range(len(oeid_df)):
        (target, depth) = get_region_depth_ophys_experiment_id(
            oeid_df.ophys_experiment_id.values[i])
        target_region.append(target)
        depths.append(depth)
    oeid_df['target_region'] = target_region
    oeid_df['depth'] = depths
    return oeid_df

##########################################
# Get-path tools
##########################################


def get_local_zstack_path(ophys_experiment_id):
    """Get local z-stack path
    Using visual behavior

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    path
        path to the local z-stack
    """
    zstack_fp = from_lims.get_general_info_for_ophys_experiment_id(ophys_experiment_id).experiment_storage_directory[0] \
        / f'{ophys_experiment_id}_z_stack_local.h5'
    if not os.path.isfile(zstack_fp):
        zstack_fp = from_lims.get_general_info_for_ophys_experiment_id(ophys_experiment_id).experiment_storage_directory[0] \
            / f'{ophys_experiment_id}_zstack_local_dewarping.h5'
    if not os.path.isfile(zstack_fp):
        raise Exception(f'Error: zstack file path might be wrong ({zstack_fp})')
    return zstack_fp


def get_storage_directory_path_for_ophys_experiment_id(ophys_experiment_id):
    """Get storage directory path for ophys_experiment_id

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    path
        path to the storage directory
    """
    # sometimes dff is not generated
    # dir_path = from_lims.get_dff_traces_filepath(ophys_experiment_id).parent
    osid = from_lims.get_ophys_session_id_for_ophys_experiment_id(
        ophys_experiment_id)
    dir_path = from_lims.get_session_h5_filepath(
        osid).parent / f'ophys_experiment_{ophys_experiment_id}'
    return dir_path


def get_raw_h5_filepath(ophys_experiment_id):
    return get_storage_directory_path_for_ophys_experiment_id(ophys_experiment_id) / f'{ophys_experiment_id}.h5'

########################################
# Z-stack tools
########################################


def get_registered_zstack(ophys_experiment_id, ophys_experiment_dir, number_of_z_planes=None, number_of_repeats=20):
    """Get within and between plane registered z-stack from an experiment.
    If it was already processed, get it from the saved file.
    If not, process z-stack registration and save the results.

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    ophys_experiment_dir : path
        directory to get or save the results

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    if number_of_z_planes is None:
        equipment_name = from_lims.get_general_info_for_ophys_experiment_id(ophys_experiment_id).equipment_name[0]
        number_of_z_planes = 81 if 'MESO' in equipment_name else 80
    reg_zstack_fn = f'{ophys_experiment_id}_zstack_reg.h5'
    reg_zstack_fp = ophys_experiment_dir / reg_zstack_fn
    if os.path.isfile(reg_zstack_fp):
        with h5py.File(reg_zstack_fp, 'r') as h:
            reg_zstack = h['data'][:]
    else:
        # Make directories if not exist
        if not os.path.isdir(ophys_experiment_dir):
            os.makedirs(ophys_experiment_dir)
        # Register the first z-stack
        reg_zstack = register_z_stack(ophys_experiment_id, number_of_z_planes=number_of_z_planes, number_of_repeats=number_of_repeats)
        # Save the registered first z-stack
        with h5py.File(reg_zstack_fp, 'w') as h:
            h.create_dataset('data', data=reg_zstack)
    return reg_zstack


def register_z_stack(ophys_experiment_id, number_of_z_planes=None, number_of_repeats=20):
    """Get registered z-stack, both within and between planes
    Can work for both taken by step protocol and loop protocol
    TODO: check if it also works for loop protocol, after fixing the rolling effect
    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    number_of_z_planes : int, optional
        numer of planes, by default 81
    number_of_repeats : int, optional
        number of repeats per plane, by default 20

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    if number_of_z_planes is None:
        equipment_name = from_lims.get_general_info_for_ophys_experiment_id(ophys_experiment_id).equipment_name[0]
        number_of_z_planes = 81 if 'MESO' in equipment_name else 80
    local_zstack_path = get_local_zstack_path(ophys_experiment_id)
    h = h5py.File(local_zstack_path, 'r')
    local_z_stack = h['data'][:]
    total_num_frames = local_z_stack.shape[0]
    assert total_num_frames == number_of_z_planes * number_of_repeats

    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = local_z_stack[range(
            plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane = average_reg_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)
    zstack_reg = reg_between_planes(np.array(mean_local_zstack_reg))
    return zstack_reg


def average_reg_plane(images):
    """Get mean FOV of a plane after registration.
    Use phase correlation

    Parameters
    ----------
    images : np.ndarray (3D)
        frames from a plane

    Returns
    -------
    np.ndarray (2D)
        mean FOV of a plane after registration.
    """
    mean_img = np.mean(images, axis=0)
    reg = np.zeros_like(images)
    for i in range(images.shape[0]):
        shift, _, _ = phase_cross_correlation(
            mean_img, images[i, :, :])
        reg[i, :, :] = scipy.ndimage.shift(images[i, :, :], shift)
    return np.mean(reg, axis=0)


def reg_between_planes(stack_imgs):
    """Register between planes. Each plane with single 2D image
    Use phase correlation.
    Use median filtered images to calculate shift between neighboring planes.
    Resulting image is not filtered.

    Parameters
    ----------
    stack_imgs : np.ndarray (3D)
        images of a stack. Typically z-stack with each plane registered and averaged.

    Returns
    -------
    np.ndarray (3D)
        Stack after plane-to-plane registration.
    """
    num_planes = stack_imgs.shape[0]
    reg_stack_imgs = np.zeros_like(stack_imgs)
    reg_stack_imgs[0, :, :] = stack_imgs[0, :, :]
    medfilt_stack_imgs = med_filt_z_stack(stack_imgs)
    reg_medfilt_imgs = np.zeros_like(stack_imgs)
    reg_medfilt_imgs[0, :, :] = medfilt_stack_imgs[0, :, :]
    for i in range(1, num_planes):
        shift, _, _ = phase_cross_correlation(
            reg_medfilt_imgs[i - 1, :, :], medfilt_stack_imgs[i, :, :])
        reg_medfilt_imgs[i, :, :] = scipy.ndimage.shift(
            medfilt_stack_imgs[i, :, :], shift)
        reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
            stack_imgs[i, :, :], shift)
    return reg_stack_imgs


def med_filt_z_stack(zstack, kernel_size=5):
    """Get z-stack with each plane median-filtered

    Parameters
    ----------
    zstack : np.ndarray
        z-stack to apply median filtering
    kernel_size : int, optional
        kernel size for median filtering, by default 5
        It seems only certain odd numbers work, e.g., 3, 5, 11, ...

    Returns
    -------
    np.ndarray
        median-filtered z-stack
    """
    filtered_z_stack = []
    for image in zstack:
        filtered_z_stack.append(cv2.medianBlur(
            image.astype(np.uint16), kernel_size))
    return np.array(filtered_z_stack)


def save_stack_to_video(stack, save_fn_path, frame_rate=5, vmin=None, vmax=None):
    """Save image stack to video

    Parameters
    ----------
    stack : np.ndarray
        image stacks to save to video
    save_fn_path : Path
        path to filename
    frame_rate : int, optional
        frame rate, by default 5
    vmin : _type_, optional
        minimum value to crop intensity, by default None
    vmax : _type_, optional
        maximum value to crop intensity, by default None
    """
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.amax(stack)
    ex_video = ((stack - vmin) / (vmax - vmin) * np.iinfo(np.uint8).max).astype(np.uint8)
    out = cv2.VideoWriter(str(save_fn_path), cv2.VideoWriter_fourcc(*"divx"), frame_rate, ex_video.shape[1:], False)
    for img in ex_video:
        out.write(img)
    out.release()

#############################################
# Main z-drift and FOV matching codes
#############################################


def get_container_zdrift_df(ocid: int, ref_oeid: int = None, oeid_to_run: np.ndarray = None, correct_image_size=(512, 512),
                            num_planes_zstack=81, zstack_interval_in_micron=0.75, segment_minute: int = 10,
                            use_clahe=True, use_valid_pix_pc=True, use_valid_pix_sr=True,
                            save_dir: Path = None, save_df=True, rerun=False, save_exp=True, save_exp_dir: Path = None):
    """Get DataFrame about zdrift throughout the entire container.
    Get the result if it was saved, unless rerun=True.
    Save the result if save_df=True.

    TODO: Get num_planes_zstack and zstack_interval_in_micron from z-stack metadata.

    Parameters
    ----------
    ocid : int
        Ophys container ID
    ref_oeid : int, optional
        Reference ophys experiment ID, by default None
        If None, then the first valid experiment in the container
    correct_image_size : tuple, optional
        Tuple for correct image size, (y,x), by default (512,512)
    num_planes_zstack : int, optional
        Number of planes in the reference z-stack, by default 81
    zstack_interval_in_micron : float, optional
        Reference z-stack plane interval in micron, by default 0.75
    segment_minute : int, optional
        Number of minutes to segment the experiment video, by default 10
    use_clahe : bool, optional
        if to use CLAHE for registration, by default True
    use_valid_pix_pc : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 1st step - phase correlation registration, by default True
    use_valid_pix_sr : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 2nd step - StackReg registration, by default True
    save_dir : Path (from pathlib), optional
        Directory path to save the DataFrame, or to retrieve from, by default None

    Returns
    -------
    DataFrame
        DataFrame about the container-wise z-drift
    """
    if oeid_to_run is None:
        oeid_all = from_lims.get_ophys_experiment_ids_for_ophys_container_id(
            ocid).ophys_experiment_id.values
        oeid_to_run = np.asarray(
            [oeid for oeid in oeid_all if check_correct_data(oeid)])
    else:
        oeid_to_run = np.asarray(oeid_to_run)  # In case if it is not an array
    if ref_oeid is None:
        ref_oeid = min(oeid_to_run)  # Assume oeid is sorted.
    save_fn = f'{ocid}_zdrift_ref_{ref_oeid}.pkl'
    if save_dir is None:
        save_dir = global_base_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if save_exp_dir is None:
        save_exp_dir = save_dir
    # retrieve dataframe from a file
    run_flag = 0
    if os.path.isfile(save_dir / save_fn) and (not rerun):
        zdrift_df = pd.read_pickle(save_dir / save_fn)
        if (np.sort(zdrift_df.ophys_experiment_id.values) == np.sort(oeid_to_run)).all():
            run_flag = 1
        # TODO: save to different versions with different set of oeid_to_run
        # What's the use case?
    if run_flag == 0:
        # Initialize columns
        matched_plane_indice = []
        zdrift = []
        corrcoef = []
        peak_cc = []
        translation_shift_list = []
        rigid_tmat_list = []
        use_clahe_list = []
        use_valid_pix_pc_list = []
        use_valid_pix_sr_list = []
        center_plane_ind = np.round(num_planes_zstack / 2).astype(int)
        for oeid in oeid_to_run:
            _mpi, _cc, *_, rigid_tmat, translation_shift, ops = \
                get_experiment_zdrift(oeid, ref_oeid, segment_minute=segment_minute, correct_image_size=correct_image_size,
                                      use_clahe=use_clahe, use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr,
                                      save_dir=save_exp_dir, save_data=save_exp, rerun=rerun)
            translation_shift_list.append(translation_shift)
            rigid_tmat_list.append(rigid_tmat)
            matched_plane_indice.append(_mpi)
            zdrift.append(
                [(mpi - center_plane_ind) * zstack_interval_in_micron for mpi in _mpi])
            corrcoef.append(_cc)
            peak_cc.append(np.amax(_cc, axis=1))
            use_clahe_list.append(ops['use_clahe'])
            use_valid_pix_pc_list.append(ops['use_valid_pix_pc'])
            use_valid_pix_sr_list.append(ops['use_valid_pix_sr'])
        zdrift_df = pd.DataFrame({'ophys_experiment_id': oeid_to_run,
                                  'zdrift': zdrift,
                                  'peak_cc': peak_cc,
                                  'corrcoef': corrcoef,
                                  'matched_plane_indice': matched_plane_indice,
                                  'translation_shift': translation_shift_list,
                                  'rigid_tmat': rigid_tmat_list,
                                  'ref_experiment_id': ref_oeid,
                                  'use_clahe': use_clahe_list,
                                  'use_valid_pix_pc': use_valid_pix_pc_list,
                                  'use_valid_pix_sr': use_valid_pix_sr_list})
        if save_df:
            zdrift_df.to_pickle(save_dir / save_fn)

    return zdrift_df


def get_experiment_zdrift(oeid, ref_oeid=None, segment_minute: int = 10, correct_image_size=(512, 512),
                          use_clahe=True, use_valid_pix_pc=True, use_valid_pix_sr=True,
                          save_dir: Path = None, save_data=True, rerun=False):
    """Get zdrift within an ophys experiment
    Register the mean FOV image to the z-stack using 2-step registration.
    Then, apply the same transformation to segmented FOVs and
    calculated matched planes in each segment.

    Parameters
    ----------
    oeid : int
        Ophys Experiment ID
    ref_oeid : int, optional
        Reference experiment ID, by default None
        If None, then the first valid experiment in the container
    segment_minute : int, optional
        Number of minutes to segment the experiment video, by default 10
    correct_image_size : tuple, optional
        Tuple for correct image size, (y,x), by default (512,512)
    use_clahe : bool, optional
        if to use CLAHE for registration, by default True
    use_valid_pix_pc : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 1st step - phase correlation registration, by default True
    use_valid_pix_sr : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 2nd step - StackReg registration, by default True
    save_dir : Path (from pathlib), optional
        Directory path to save the DataFrame, or to retrieve from, by default None

    Returns
    -------
    np.ndarray (1d: int)
        matched plane indice from all the segments
    np.ndarray (2d: float)
        correlation coeffiecient for each segment across the reference z-stack
    np.ndarray (3d: float)
        Segment FOVs registered to the z-stack
    int
        Index of the matched plane for the mean FOV
    np.ndarray (1d: float)
        correlation coefficient for the mean FOV across the reference z-stack
    np.ndarray (2d: float)
        Mean FOV registered to the reference z-stack
    np.ndarray (3d: float)
        Reference z-stack cropped using motion output of the experiment
    np.ndarray (2d: float)
        Transformation matrix for StackReg RIGID_BODY
    np.ndarray (1d: int or float)
        An array of translation shift
    dict
        Options for calculating z-drift
    """
    valid_threshold = 0.01  # TODO: find the best valid threshold

    # Basic info
    if save_dir is None:
        save_dir = global_base_dir
    ophys_container_id = from_lims.get_ophys_container_id_for_ophys_experiment_id(
        oeid)
    range_y, range_x = get_motion_correction_crop_xy_range(
        oeid)  # to remove rolling effect from motion correction
    if ref_oeid is None:
        ref_oeid = find_first_experiment_id_from_ophys_experiment_id(
            oeid, correct_image_size=correct_image_size)
    ref_dir = save_dir / \
        f'container_{ophys_container_id}' / f'experiment_{ref_oeid}'
    exp_dir = save_dir / \
        f'container_{ophys_container_id}' / f'experiment_{oeid}'

    # Get the saved result if there is
    exp_fn = f'{oeid}_zdrift_ref_{ref_oeid}.h5'
    if os.path.isfile(exp_dir / exp_fn) and (not rerun):
        with h5py.File(exp_dir / exp_fn, 'r') as h:
            matched_plane_indice = h['matched_plane_indice'][:]
            corrcoef = h['corrcoef'][:]
            segment_reg_imgs = h['segment_fov_registered'][:]
            mpi_mean_fov = h['matched_plane_index_mean_fov'][0]
            cc_mean_fov = h['corrcoef_index_mean_fov'][:]
            regimg_mean_fov = h['mean_fov_registered'][:]
            ref_zstack_crop = h['ref_zstack_crop'][:]
            rigid_tmat = h['rigid_tmat'][:]
            translation_shift = h['translation_shift'][:]
            use_clahe = h['ops/use_clahe'][0]
            use_valid_pix_pc = h['ops/use_valid_pix_pc'][0]
            use_valid_pix_sr = h['ops/use_valid_pix_sr'][0]
            ops = {'use_clahe': use_clahe,
                   'use_valid_pix_pc': use_valid_pix_pc,
                   'use_valid_pix_sr': use_valid_pix_sr}

    else:
        # Get reference z-stack and crop
        ref_zstack = get_registered_zstack(ref_oeid, ref_dir)
        ref_zstack_crop = ref_zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

        # Get mean FOV and crop
        mean_img = np.asarray(Image.open(
            from_lims.get_average_intensity_projection_filepath(oeid)))  # in uint8
        mean_img_crop = mean_img[range_y[0]:range_y[1], range_x[0]:range_x[1]]

        # Run registration from mean FOV to reference z-stack
        mpi_mean_fov, cc_mean_fov, regimg_mean_fov, *_, rigid_tmat, translation_shift = estimate_plane_from_ref_zstack(
            mean_img_crop, ref_zstack_crop, use_clahe=use_clahe, use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr)

        # Get segmented FOVs and crop
        segment_mean_images = get_segment_mean_images(
            oeid, save_dir=exp_dir, segment_minute=segment_minute)
        segment_mean_images_crop = segment_mean_images[:,
                                                       range_y[0]:range_y[1], range_x[0]:range_x[1]]

        # Apply transformations from the mean FOV registration
        segment_reg_imgs = twostep_fov_reg_zdrift(
            segment_mean_images_crop, translation_shift, rigid_tmat)

        # Calculate correlation coefficients and estimate matched plane indice
        matched_plane_indice = []
        corrcoef_list = []
        for i in range(segment_reg_imgs.shape[0]):
            fov = segment_reg_imgs[i]
            corrcoef = np.zeros_like(np.zeros(ref_zstack_crop.shape[0]))
            for pi in range(len(corrcoef)):
                valid_y, valid_x = np.where(fov > valid_threshold)
                corrcoef[pi] = np.corrcoef(ref_zstack_crop[pi, valid_y, valid_x].flatten(
                ), fov[valid_y, valid_x].flatten())[0, 1]
            corrcoef = moving_average(np.array(corrcoef), 5)
            matched_plane_index = np.argmax(corrcoef)
            corrcoef_list.append(corrcoef)
            matched_plane_indice.append(matched_plane_index)
        corrcoef = np.asarray(corrcoef_list)
        matched_plane_indice = np.asarray(matched_plane_indice).astype(int)

        ops = {'use_clahe': use_clahe,
               'use_valid_pix_pc': use_valid_pix_pc,
               'use_valid_pix_sr': use_valid_pix_sr}

        if save_data:
            # Save the result
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
            with h5py.File(exp_dir / exp_fn, 'w') as h:
                h.create_dataset('matched_plane_indice',
                                 data=matched_plane_indice)
                h.create_dataset('corrcoef', data=corrcoef)
                h.create_dataset('segment_fov_registered',
                                 data=segment_reg_imgs)
                h.create_dataset('matched_plane_index_mean_fov',
                                 shape=(1,), data=mpi_mean_fov)
                h.create_dataset('corrcoef_index_mean_fov', data=cc_mean_fov)
                h.create_dataset('mean_fov_registered', data=regimg_mean_fov)
                h.create_dataset('ref_zstack_crop', data=ref_zstack_crop)
                h.create_dataset('rigid_tmat', data=rigid_tmat)
                h.create_dataset('translation_shift', data=translation_shift)
                h.create_dataset('ops/use_clahe', shape=(1,), data=use_clahe)
                h.create_dataset('ops/use_valid_pix_pc',
                                 shape=(1,), data=use_valid_pix_pc)
                h.create_dataset('ops/use_valid_pix_sr',
                                 shape=(1,), data=use_valid_pix_sr)

    return matched_plane_indice, corrcoef, segment_reg_imgs, \
        mpi_mean_fov, cc_mean_fov, regimg_mean_fov, ref_zstack_crop, \
        rigid_tmat, translation_shift, ops


def estimate_plane_from_ref_zstack(fov_crop: np.array, ref_zstack_crop: np.array,
                                   use_clahe=True, use_valid_pix_pc=True, use_valid_pix_sr=True):
    """Estimate matched plane of the FOV from the first z-stack.
    Assume both inputs are cropped using the motion correction result (get_motion_correction_crop_xy_range)
    Use phase correlation, and then StackReg.RIGID_BODY.
    Apply CLAHE and uint16 conversion before RIGID_BODY registration.
    Apply the StackReg result to original fov (after phase correlation registration).

    Parameters
    ----------
    fov_crop : np.array
        mean FOV
    ref_zstack_crop : np.array
        local z-stack from the reference session corresponding to the FOV
    use_clahe: bool, optional
        If to adjust contrast using CLAHE for phase correlation registration, by default True
    use_valid_pix_pc : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 1st step - phase correlation registration, by default True
    use_valid_pix_sr : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 2nd step - StackReg registration, by default True

    Returns
    -------
    matched_plane_index: int
        Index of the matched plane in the first z-stack
    corrcoef_second: np.array
        1d array of correlation coefficient after smoothing corrcoef_first
    corrcoef_first: np.array
        1d array of correlation coefficient after RIGID_BODY registration, in each plane
    corrcoef_pre: np.array
        1d array of correlation coefficient after phase correlation, in each plane
    registered_fov_sr: np.ndarray
        2d FOV after RIGID_BODY registration. Match the dimension to the fov parameter
    registered_fov_pc: np.ndarray
        2d FOV after phase correlation registration.
    """
    assert len(fov_crop.shape) == 2
    assert len(ref_zstack_crop.shape) == 3
    assert fov_crop.shape == ref_zstack_crop.shape[1:]

    # Register using phase correlation (translation only)
    fov_reg_stack_pre, corrcoef_pre, shift_list = \
        fov_stack_register_phase_correlation(
            fov_crop, ref_zstack_crop, use_clahe=use_clahe, use_valid_pix=use_valid_pix_pc)
    corrcoef_pre = moving_average(corrcoef_pre)

    # TODO: Find a better measure to test first pass registration
    # Select the best registered FOV and translation shift
    matched_pi_pre = np.argmax(corrcoef_pre)
    fov_pre = fov_reg_stack_pre[matched_pi_pre, :, :]
    translation_shift = shift_list[matched_pi_pre]

    # Crop the best fov and zstack using valid indices
    # 0 is fine because it is translation
    valid_y, valid_x = np.where(fov_pre > 0)
    fov_pre = fov_pre[min(valid_y):max(valid_y), min(valid_x):max(valid_x)]
    z_stack_for_rigid = ref_zstack_crop[:, min(valid_y):max(valid_y), min(
        valid_x):max(valid_x)]  # this works because it is translation

    # Rigid-body registration (allowing rotation)
    # After applying CLAHE and uint16 image normalization
    matched_plane_index, corrcoef_second, rigid_tmat = fov_stack_register_rigid(
        fov_pre, z_stack_for_rigid, use_valid_pix=use_valid_pix_sr)

    # Get the registered fovs matched to the input fov dimension
    registered_fov_pc = scipy.ndimage.shift(fov_crop, translation_shift)
    sr = StackReg(StackReg.RIGID_BODY)
    registered_fov = sr.transform(registered_fov_pc, tmat=rigid_tmat)

    return matched_plane_index, corrcoef_second, registered_fov, corrcoef_pre, registered_fov_pc, rigid_tmat, translation_shift


def fov_stack_register_phase_correlation(fov, stack, use_clahe=True, use_valid_pix=True):
    """ Reigster FOV to each plane in the stack

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image
    stack : np.ndarray (3d)
        stack images
    use_clahe: bool, optional
        If to adjust contrast using CLAHE for registration, by default True
    use_valid_pix : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient, by default True

    Returns
    -------
    np.ndarray (3d)
        stack of FOV registered to each plane in the input stack
    np.array (1d)
        correlation coefficient between the registered fov and the stack in each plane
    list
        list of translation shifts
    """
    assert len(fov.shape) == 2
    assert len(stack.shape) == 3
    assert fov.shape == stack.shape[1:]

    # median filtering the z-stack as default
    # if use_medfilt:
    stack_pre = med_filt_z_stack(stack)
    # else:
    #     stack_pre = stack.copy()

    if use_clahe:
        fov_for_reg = image_normalization_uint16(skimage.exposure.equalize_adapthist(
            fov.astype(np.uint16)))  # normalization to make it uint16
        stack_for_reg = np.zeros_like(stack_pre)
        for pi in range(stack.shape[0]):
            stack_for_reg[pi, :, :] = image_normalization_uint16(
                skimage.exposure.equalize_adapthist(stack_pre[pi, :, :].astype(np.uint16)))
    else:
        fov_for_reg = fov.copy()
        stack_for_reg = stack_pre.copy()

    fov_reg_stack = np.zeros_like(stack)
    corrcoef = np.zeros(stack.shape[0])
    shift_list = []
    for pi in range(stack.shape[0]):
        shift, _, _ = phase_cross_correlation(
            stack_for_reg[pi, :, :], fov_for_reg)
        fov_reg = scipy.ndimage.shift(fov, shift)
        fov_reg_stack[pi, :, :] = fov_reg
        if use_valid_pix:
            valid_y, valid_x = np.where(fov_reg > 0.01)
            corrcoef[pi] = np.corrcoef(stack[pi, valid_y, valid_x].flatten(
            ), fov_reg[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef[pi] = np.corrcoef(
                stack[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
        shift_list.append(shift)
    return fov_reg_stack, corrcoef, shift_list


def fov_stack_register_rigid(fov, stack, use_valid_pix=True):
    """Find the best-matched plane index from the stack using rigid-body registration (StackReg)
    First register the FOV to each plane in the stack
    First pass: Pick the best-matched plane using pixel-wise correlation
    Second pass: Using the transformed FOV to the first pass best-matched plane,
        sweep through all planes and refine the best-matched plane using correlation.
        This fixes minor mis-match due to registration error.

    Parameters
    ----------
    fov : np.ndarray (2D)
        input FOV
    stack : np.ndarray (3D)
        stack images
    use_valid_pix : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient, by default True

    Returns
    -------
    np.ndarray (1D)
        matched plane index

    """
    assert len(fov.shape) == 2
    assert len(stack.shape) == 3
    assert fov.shape == stack.shape[1:]

    # TODO: figure out what is the best threshold. This value should be larger than one because of the results after registration
    valid_pix_threshold = 0.01

    # median filtering the z-stack as default
    # if use_medfilt:
    stack_pre = med_filt_z_stack(stack)
    # else:
    #     stack_pre = stack.copy()

    sr = StackReg(StackReg.RIGID_BODY)

    # apply CLAHE and normalize to make it uint16 (for StackReg)
    fov_clahe = image_normalization_uint16(skimage.exposure.equalize_adapthist(
        fov.astype(np.uint16)))  # normalization to make it uint16
    stack_clahe = np.zeros_like(stack_pre)
    for pi in range(stack_pre.shape[0]):
        stack_clahe[pi, :, :] = image_normalization_uint16(
            skimage.exposure.equalize_adapthist(stack_pre[pi, :, :].astype(np.uint16)))

    # Initialize
    corrcoef_first = np.zeros(stack_clahe.shape[0])
    fov_reg_stack = np.zeros_like(stack)
    tmat_list = []
    for pi in range(len(corrcoef_first)):
        tmat = sr.register(stack_clahe[pi, :, :], fov_clahe)
        # Apply the transformation matrix to the FOV registered using phase correlation
        fov_reg = sr.transform(fov, tmat=tmat)
        if use_valid_pix:
            valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
            corrcoef_first[pi] = np.corrcoef(
                stack[pi, valid_y, valid_x].flatten(), fov_reg[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef_first[pi] = np.corrcoef(
                stack[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
        fov_reg_stack[pi, :, :] = fov_reg
        tmat_list.append(tmat)

    corrcoef_first = moving_average(corrcoef_first, 5)
    max_pi_first = np.argmax(corrcoef_first)

    # Use the best registered FOV to sweep through the first z-stack again
    fov_final = fov_reg_stack[max_pi_first, :, :]
    valid_y, valid_x = np.where(fov_final > valid_pix_threshold)

    # Calculate correlation between each z-stack to the best RIGID_BODY registered FOV
    corrcoef_second = np.zeros_like(corrcoef_first)
    for pi in range(len(corrcoef_first)):
        if use_valid_pix:
            corrcoef_second[pi] = np.corrcoef(stack[pi, valid_y, valid_x].flatten(
            ), fov_final[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef_second[pi] = np.corrcoef(
                stack[pi, :, :].flatten(), fov_final.flatten())[0, 1]
    corrcoef_second = moving_average(np.array(corrcoef_second), 5)
    matched_plane_index = np.argmax(corrcoef_second)
    tmat = tmat_list[matched_plane_index]

    return matched_plane_index, corrcoef_second, tmat


def get_segment_mean_images(ophys_experiment_id, save_dir=None, save_images=True, segment_minute=10):
    """Get segmented mean images of an experiment
    And save the result, in case the directory was specified.

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    save_dir : Path, optional
        ophys experiment directory to load/save.
        If None, then do not attemp loading previously saved data (it takes 2-3 min)
    segment_minute : int, optional
        length of a segment, in min, by default 10

    Returns
    -------
    np.ndarray (3d)
        mean FOVs from each segment of the video
    """

    if save_dir is None:
        save_dir = global_base_dir
    got_flag = 0
    segment_fov_fp = save_dir / f'{ophys_experiment_id}_segment_fov.h5'
    if os.path.isfile(segment_fov_fp):
        with h5py.File(segment_fov_fp, 'r') as h:
            mean_images = h['data'][:]
            got_flag = 1

    if got_flag == 0:
        frame_rate, timestamps = get_correct_frame_rate(ophys_experiment_id)
        movie_fp = from_lims.get_motion_corrected_movie_filepath(
            ophys_experiment_id)
        h = h5py.File(movie_fp, 'r')
        movie_len = h['data'].shape[0]
        segment_len = int(np.round(segment_minute * frame_rate * 60))

        # get frame start and end (for range) indices
        frame_start_end = get_frame_start_end(movie_len, segment_len)

        mean_images = np.zeros((len(frame_start_end), *h['data'].shape[1:]))
        for i in range(len(frame_start_end)):
            mean_images[i, :, :] = np.mean(
                h['data'][frame_start_end[i][0]:frame_start_end[i][1], :, :], axis=0)
        if save_images:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            with h5py.File(segment_fov_fp, 'w') as h:
                h.create_dataset('data', data=mean_images)
    return mean_images


def twostep_fov_reg_zdrift(segment_mean_images, translation_shift, rigid_tmat):
    """FOV registration using translation (1st step) and rigid body (2nd step).
    Use the result from previous registration (estimate_plane_from_ref_zstack)

    Parameters
    ----------
    segment_mean_images : np.ndarray(2d or 3d)
        mean images to apply 2-step registration
    translation_shift : np.ndarray(1d, 2 entries)
        parameters for translation shift (using phase correlation)
    rigid_tmat : np.ndarray (2d, 3 x 3)
        StackReg transformation matrix

    Returns
    -------
    np.ndarray(2d or 3d)
        resulting image(s)
    """
    sr = StackReg(StackReg.RIGID_BODY)
    if len(segment_mean_images.shape) == 2:
        segment_mean_images = np.expand_dims(segment_mean_images, 0)
    assert len(segment_mean_images.shape) == 3
    segment_reg_imgs = np.zeros_like(segment_mean_images)
    for i in range(segment_mean_images.shape[0]):
        fov = segment_mean_images[i]
        registered_fov_pc = scipy.ndimage.shift(fov, translation_shift)
        segment_reg_imgs[i] = sr.transform(registered_fov_pc, tmat=rigid_tmat)
    if segment_reg_imgs.shape[0] == 1:
        segment_reg_imgs = segment_reg_imgs[0]
    return segment_reg_imgs


########################################
# Plot
########################################

def plot_container_zdrift_using_df(ocid, zdrift_df, save_dir=None, save_figure=True, fig_size=(6, 5),
                                   cc_threshold=0.5, y_min=-20, y_max=20):
    if save_dir is None:
        save_dir = global_base_dir
    png_fn = f'container_{ocid}_zdrift.png'
    pdf_fn = f'container_{ocid}_zdrift.pdf'

    zdrift = zdrift_df.zdrift.values
    peak_cc = zdrift_df.peak_cc.values
    max_num_segments = max([len(z) for z in zdrift])
    num_exp = len(zdrift)
    has_low_cc = 0
    fig, ax = plt.subplots(figsize=fig_size)
    for i in range(num_exp):
        depth = np.asarray(zdrift[i])
        x = np.linspace(i, i + len(depth) / max_num_segments, len(depth) + 1)[:-1]
        ax.plot(x, depth, 'k-', zorder=-1)
        colors = peak_cc[i]
        h1 = ax.scatter(x, depth, s=20, c=colors,
                        cmap='binary', vmin=cc_threshold, vmax=1)

        under_threshold_ind = np.where(colors < cc_threshold)[0]
        if len(under_threshold_ind) > 0:
            has_low_cc = 1
            h2 = ax.scatter(x[under_threshold_ind], depth[under_threshold_ind], s=20,
                            c=colors[under_threshold_ind], cmap='Reds_r', vmin=0, vmax=cc_threshold)
    if has_low_cc == 0:
        ylim = ax.get_ybound()
        xlim = ax.get_xbound()
        h2 = ax.scatter(xlim[0] - 1, ylim[0] - 1, c=0,
                        cmap='Reds_r', vmin=0, vmax=cc_threshold)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    cax1 = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0 + (ax.get_position().height) * cc_threshold,
                        0.02,
                        ax.get_position().height * (1 - cc_threshold)])
    bar1 = plt.colorbar(h1, cax=cax1)
    cax2 = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height * cc_threshold])
    plt.colorbar(h2, cax=cax2)
    bar1.set_label('Correlation coefficient')

    ax.set_xlabel('Session #')
    ax.set_ylabel(r'Relative depths ($\mu$m)')
    ax.set_title(f'Container {ocid}')
    if (y_min is not None) and (y_max is not None):
        ylim = ax.get_ybound()
        if (ylim[0] < y_min) or (ylim[1]) > y_max:
            ax.yaxis.label.set_color('red')
            ax.spines['left'].set_color('red')
            ax.tick_params(axis='y', colors='red')
        else:
            ax.set_ylim(y_min, y_max)
    # if (not os.path.isfile(save_dir / png_fn)) or (not os.path.isfile(save_dir / pdf_fn)):
    if save_figure:
        fig.savefig(save_dir / png_fn, bbox_inches='tight')
        fig.savefig(save_dir / pdf_fn, bbox_inches='tight')
    return fig, ax


def plot_experiment_zdrift(oeid, ref_oeid=None, save_dir=None, save_figure=True, fig_size=(18, 10)):
    if ref_oeid is None:
        ref_oeid = find_first_experiment_id_from_ophys_experiment_id(oeid)
    matched_plane_indice, corrcoef, segment_reg_imgs, \
        mpi_mean_fov, cc_mean_fov, regimg_mean_fov, ref_zstack_crop, \
        rigid_tmat, translation_shift, ops = get_experiment_zdrift(
            oeid, ref_oeid=ref_oeid)

    fig, ax = plt.subplots(2, 3, figsize=fig_size)
    ax[0, 0].imshow(ref_zstack_crop[mpi_mean_fov])
    ax[0, 0].set_title('Matched z-stack plane')
    ax[0, 1].imshow(regimg_mean_fov)
    ax[0, 1].set_title('Mean FOV')
    ax[0, 2].imshow(image_normalization_uint8(
        im_blend(ref_zstack_crop[mpi_mean_fov], regimg_mean_fov, 0.5)))
    ax[0, 2].set_title('Merged')
    ax[1, 0].axhline(mpi_mean_fov, 0, 1, linestyle='--', color='r', zorder=-1)
    ax[1, 0].plot(range(len(matched_plane_indice)), matched_plane_indice, 'k-')
    h = ax[1, 0].scatter(range(len(matched_plane_indice)), matched_plane_indice, s=100, c=np.max(corrcoef, axis=1), cmap='binary',
                         vmin=0, vmax=1)
    cax = fig.add_axes([ax[1, 0].get_position().x1 + 0.005,
                        ax[1, 0].get_position().y0,
                        0.01,
                        ax[1, 0].get_position().height])
    bar = plt.colorbar(h, cax=cax)
    bar.set_label('Correlation coefficient')
    ax[1, 0].set_xlabel('Segment')
    ax[1, 0].set_ylabel('Matched plane index')
    ax[1, 0].set_yticks(range(np.amin(matched_plane_indice),
                        np.amax(matched_plane_indice) + 1))

    ax[1, 1].axis('off')

    for i in range(len(matched_plane_indice)):
        # ax[1,2].plot(corrcoef[i,:], color=[0, 0, 1-(i+1)/(len(matched_plane_indice)+1)], label=f'segment {i}')
        ax[1, 2].plot(corrcoef[i, :], c=cm.winter(
            i / len(matched_plane_indice)), label=f'segment {i}')

    ax[1, 2].legend()
    ax[1, 2].set_ylabel('Correlation coefficient')
    ax[1, 2].set_xlabel('z-stack plane index')

    fig.suptitle(f'Exp {oeid} z-drift from {ref_oeid}')

    return fig

#####################################
# For QC
#####################################


def plot_corrcoef_container(ophys_container_id, container_dir, save_dir=None, container_info=None, ref_oeid=None):
    container_title = f'Container {ophys_container_id}: '
    if container_info is not None:
        if 'mouse_id' in container_info.keys():
            mouse_id = container_info.mouse_id
            container_title = f'{container_title} Mouse ID {mouse_id}'
        if 'targeted_structure' in container_info.keys():
            targeted_structure = container_info.targeted_structure
            container_title = f'{container_title} {targeted_structure}'
        if 'upper_or_lower' in container_info.keys():
            targeted_layer = container_info.upper_or_lower
            container_title = f'{container_title} {targeted_layer}'
    if ref_oeid is None:
        ref_oeid = find_first_experiment_id_from_ophys_container_id(
            ophys_container_id, valid_only=True)
    container_title = f'{container_title} (reference epxeriment = {ref_oeid})'

    oeids = from_lims.get_ophys_experiment_ids_for_ophys_container_id(
        ophys_container_id).ophys_experiment_id.values
    num_oeid = len(oeids)
    nx = int(6)
    ny = int(np.ceil(num_oeid / nx))
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 3, ny * 3))
    for ax_i in range(ny * nx):
        yi = ax_i // nx
        xi = ax_i % nx
        ax_temp = ax[yi, xi]
        if ax_i < num_oeid:
            oeid = oeids[ax_i]
            if check_correct_data(oeid):
                exp_dir = container_dir / f'experiment_{oeid}'
                if oeid == ref_oeid:
                    match_fn = f'{oeid}_plane_estimated_from_session.h5'
                else:
                    match_fn = f'{oeid}_plane_estimated_from_{ref_oeid}.h5'
                h = h5py.File(exp_dir / match_fn, 'r')
                matched_plane_indice = h['matched_plane_indice'][:].astype(int)
                corrcoef = h['corrcoef'][:]
                num_segments = len(matched_plane_indice)
                num_planes = len(corrcoef[0])
                for i in range(num_segments):
                    x = range(-matched_plane_indice[i], - matched_plane_indice[i] + num_planes)
                    ax_temp.plot(x, corrcoef[i], c=cm.winter(
                        i / num_segments), label=f'segment_{i}')
                ax_temp.legend(fontsize=7)
                if yi == ny - 1:
                    ax_temp.set_xlabel(
                        'Plane index\n(Relative to the matched plane)', fontsize=10)
                if xi == 0:
                    ax_temp.set_ylabel('Correlation coefficient', fontsize=10)
                ax_temp.set_title(f'Experiment {oeid}', fontsize=10)
                ax_temp.tick_params(axis='both', which='major', labelsize=8)
            else:
                ax_temp.set_title(
                    f'DATA ERROR: Experiment {oeid}', fontsize=10)
                ax_temp.tick_params(axis='both', which='major', labelsize=8)
        else:
            ax_temp.axis('off')
    fig.suptitle(container_title, fontsize=12)
    fig.tight_layout()

    if save_dir is not None:
        save_fn_base = f'container_{ophys_container_id}_FOV_matching_correlation_plot'
        fig.savefig(save_dir / f'{save_fn_base}.png',
                    bbox_inches='tight', dpi=300)
        fig.savefig(save_dir / f'{save_fn_base}.pdf',
                    bbox_inches='tight', dpi=300, transparent=True)


def get_raw_segment_mean_images(ophys_experiment_id, save_dir=None, segment_minute=10):
    """Get segmented mean images of an experiment
    And save the result, in case the directory was specified.

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    save_dir : Path, optional
        ohpys experiment directory to load/save.
        If None, then do not attemp loading previously saved data (it takes 2-3 min)
    segment_minute : int, optional
        length of a segment, in min, by default 10

    Returns
    -------
    np.ndarray (3d)
        mean FOVs from each segment of the video
    """
    got_flag = 0
    if save_dir is not None:
        segment_fov_fp = save_dir / 'raw_segment_fov.h5'
        if os.path.isfile(segment_fov_fp):
            with h5py.File(segment_fov_fp, 'r') as h:
                mean_images = h['data'][:]
                got_flag = 1

    if got_flag == 0:
        frame_rate, timestamps = get_correct_frame_rate(ophys_experiment_id)
        raw_h5_fp = get_raw_h5_filepath(ophys_experiment_id)
        h = h5py.File(raw_h5_fp, 'r')
        movie_len = h['data'].shape[0]
        segment_len = int(np.round(segment_minute * frame_rate * 60))

        # get frame start and end (for range) indices
        frame_start_end = get_frame_start_end(movie_len, segment_len)

        mean_images = np.zeros((len(frame_start_end), *h['data'].shape[1:]))
        for i in range(len(frame_start_end)):
            mean_images[i, :, :] = np.mean(
                h['data'][frame_start_end[i][0]:frame_start_end[i][1], :, :], axis=0)
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            with h5py.File(segment_fov_fp, 'w') as h:
                h.create_dataset('data', data=mean_images)
    return mean_images


def register_between_mean_fovs(mean_fovs):
    reg_fovs = np.zeros_like(mean_fovs)
    reg_fovs[0] = mean_fovs[0]
    shift_prev = np.array([0, 0]).astype(np.float64)
    for i in range(1, mean_fovs.shape[0]):
        shift_temp, _, _ = phase_cross_correlation(
            mean_fovs[i - 1, :, :], mean_fovs[i, :, :])
        shift_prev += shift_temp
        reg_fovs[i, :, :] = scipy.ndimage.shift(mean_fovs[i, :, :], shift_prev)
    return reg_fovs


def get_raw_segment_mean_images_registered(ophys_experiment_id, save_dir=None, segment_minute=10):
    got_flag = 0
    if save_dir is not None:
        raw_reg_segment_fov_fp = save_dir / 'raw_segment_fov_registered.h5'
        if os.path.isfile(raw_reg_segment_fov_fp):
            with h5py.File(raw_reg_segment_fov_fp, 'r') as h:
                mean_images_reg = h['data'][:]
                got_flag = 1
    if got_flag == 0:
        mean_images = get_raw_segment_mean_images(
            ophys_experiment_id, save_dir=save_dir, segment_minute=segment_minute)
        mean_images_reg = register_between_mean_fovs(mean_images)
        if save_dir is not None:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            with h5py.File(raw_reg_segment_fov_fp, 'w') as h:
                h.create_dataset('data', data=mean_images_reg)
    return mean_images_reg


def save_segment_fov_motion_crop(ophys_experiment_id, save_dir, segment_minute=10):
    save_dir_bases = ['tif', 'gif']
    for dir_base in save_dir_bases:
        if not os.path.isdir(save_dir / dir_base):
            os.makedirs(save_dir / dir_base)
    if os.path.isfile(save_dir / 'gif' / f'{ophys_experiment_id}_segment_fov.gif'):
        print(f'{ophys_experiment_id} already saved.')
    else:
        range_y, range_x = get_motion_correction_crop_xy_range(
            ophys_experiment_id)
        segment_fovs = get_segment_mean_images(
            ophys_experiment_id, segment_minute=segment_minute)
        segment_fovs_crop = segment_fovs[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

        norm_uint8_segment_fovs = image_normalization_uint8(
            np.array(segment_fovs_crop), im_thresh=0)  # for gif and mp4
        norm_uint16_segment_fovs = image_normalization_uint16(
            np.array(segment_fovs_crop), im_thresh=0)  # for tiff stack
        tifffile.imwrite(
            save_dir / 'tif' / f'{ophys_experiment_id}_segment_fov.tif', norm_uint16_segment_fovs)
        imageio.mimsave(
            save_dir / 'gif' / f'{ophys_experiment_id}_segment_fov.gif', norm_uint8_segment_fovs, fps=2)


def save_segment_fov(ophys_experiment_id, save_dir, segment_minute=10):
    tif_image = save_dir / 'tif' / f'{ophys_experiment_id}_segment_fov.tif'
    gif = save_dir / 'gif' / f'{ophys_experiment_id}_segment_fov.gif'
    save_dir_bases = ['tif', 'gif']
    for dir_base in save_dir_bases:
        if not os.path.isdir(save_dir / dir_base):
            os.makedirs(save_dir / dir_base)
    if os.path.isfile(gif):
        print(f'{ophys_experiment_id} already saved.')
    else:
        segment_fovs = get_segment_mean_images(
            ophys_experiment_id, save_dir=save_dir, segment_minute=segment_minute)
        norm_uint8_segment_fovs = image_normalization_uint8(
            np.array(segment_fovs), im_thresh=0)  # for gif and mp4
        norm_uint16_segment_fovs = image_normalization_uint16(
            np.array(segment_fovs), im_thresh=0)  # for tiff stack
        tifffile.imwrite(
            tif_image, norm_uint16_segment_fovs)
        imageio.mimsave(
            gif, norm_uint8_segment_fovs, fps=2)
    return tif_image, gif


def save_raw_segment_fov(ophys_experiment_id, save_dir):
    raw_segment_fovs_registered = get_raw_segment_mean_images_registered(
        ophys_experiment_id, save_dir)
    norm_uint8_segment_fovs = image_normalization_uint8(
        np.array(raw_segment_fovs_registered), im_thresh=0)  # for gif and mp4
    norm_uint16_segment_fovs = image_normalization_uint16(
        np.array(raw_segment_fovs_registered), im_thresh=0)  # for tiff stack
    with h5py.File(save_dir / 'raw_segment_fov.h5', 'w') as h:
        h.create_dataset('data', data=np.array(raw_segment_fovs_registered))
    tif_image = save_dir / 'raw_segment_fov.tif'
    gif = save_dir / 'raw_segment_fov.gif'
    tifffile.imwrite(tif_image, norm_uint16_segment_fovs)
    save_stack_to_video(norm_uint8_segment_fovs, save_dir / 'raw_segment_fov.mp4', frame_rate=2)
    imageio.mimsave(gif,
                    norm_uint8_segment_fovs, fps=2)
    return tif_image, gif