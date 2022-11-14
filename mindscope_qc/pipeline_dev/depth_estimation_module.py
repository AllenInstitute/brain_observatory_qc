import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pystackreg import StackReg
import os, h5py, scipy, skimage, cv2
from pathlib import Path
from visual_behavior.data_access import from_lims as vba_from_lims 
from visual_behavior.data_access import from_lims_utilities as vba_from_lims_utilities
from visual_behavior import database as db
from mindscope_qc.data_access import from_lims as mqc_from_lims

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
    zstack_fp = vba_from_lims.get_general_info_for_ophys_experiment_id(ophys_experiment_id).experiment_storage_directory[0] \
        / f'{ophys_experiment_id}_z_stack_local.h5'
    return zstack_fp

def get_local_zstack_path_mqc(ophys_experiment_id):
    """Get local z-stack path using mindscop_qc

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    path
        path to the local z-stack
    """
    storage_path = get_storage_directory_path_for_ophys_experiment_id(ophys_experiment_id)
    local_zstack_path = storage_path / f'{ophys_experiment_id}_z_stack_local.h5'
    return local_zstack_path
    
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
    try:
        dir_path = mqc_from_lims.get_dff_traces_filepath(ophys_experiment_id).parent
    except:
        osid = mqc_from_lims.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
        dir_path = mqc_from_lims.get_session_h5_filepath(osid).parent / f'ophys_experiment_{ophys_experiment_id}'
    return dir_path

def get_ophys_session_ids_for_mouse_id(mouse_id):
    """Get ophyse session IDs for mouse ID

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
    oeid_df = vba_from_lims.get_ophys_experiment_ids_for_ophys_session_id(ophys_session_id)
    target_region = []
    depths = []
    for i in range(len(oeid_df)):
        (target, depth) = get_region_depth_ophys_experiment_id(oeid_df.ophys_experiment_id.values[i])
        target_region.append(target)
        depths.append(depth)
    oeid_df['target_region'] = target_region
    oeid_df['depth'] = depths
    return oeid_df

def get_segment_mean_images(ophys_experiment_id, segment_minute = 10):
    """Get segmented mean images of an experiment

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    segment_minute : int, optional
        length of a segment, in min, by default 10

    Returns
    -------
    np.ndarray (3d)
        mean FOVs from each segment of the video
    """
    frame_rate, timestamps = get_correct_frame_rate(ophys_experiment_id)
    movie_fp = mqc_from_lims.get_motion_corrected_movie_filepath(ophys_experiment_id)
    h = h5py.File(movie_fp, 'r')
    movie_len = h['data'].shape[0]
    segment_len = int(np.round(segment_minute * frame_rate * 60))

    # get frame start and end (for range) indices
    frame_start_end = get_frame_start_end(movie_len, segment_len)
    
    mean_images = np.zeros((len(frame_start_end), *h['data'].shape[1:]))
    for i in range(len(frame_start_end)):
        mean_images[i,:,:] = np.mean(h['data'][frame_start_end[i][0]:frame_start_end[i][1],:,:], axis=0)
    return mean_images
    
def get_correct_frame_rate(ophys_experiment_id):
    """ Getting the correct frame rate, rather than fixed 11.0 from the metadata

    Args:
        ophys_experiment_id (int): ophys experiment ID

    Returns:
        float: frame rate
    """
    # TODO: change from_lims_utilities from vba to that in mindscope_qc.
    # mindscope_qc currently does not seem to have tools for getting timestamps.
    lims_data = vba_from_lims_utilities.utils.get_lims_data(ophys_experiment_id)
    timestamps = vba_from_lims_utilities.utils.get_timestamps(lims_data)
    frame_rate = 1/np.mean(np.diff(timestamps.ophys_frames.timestamps))
    return frame_rate, timestamps

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
    dividers = range(0,movie_len,segment_len)
    frame_start_end = []
    for i in range(len(dividers)-2):
        frame_start_end.append([dividers[i], dividers[i+1]])
    if movie_len//segment_len == int(np.round(movie_len/segment_len)):
        # the last segment is appended
        frame_start_end.append([dividers[-2], movie_len])
    else:
        # the last segment as independent segment
        frame_start_end.append([dividers[-2], dividers[-1]])
        frame_start_end.append([dividers[-1], movie_len])
    return frame_start_end

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
    mean_img = np.mean(images,axis=0)
    reg = np.zeros_like(images)
    for i in range(images.shape[0]):
        shift,_,_ = skimage.registration.phase_cross_correlation(mean_img, images[i,:,:], normalization=None)
        reg[i,:,:] = scipy.ndimage.shift(images[i,:,:], shift)
    return np.mean(reg,axis=0)

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
    reg_stack_imgs[0,:,:] = stack_imgs[0,:,:]
    medfilt_stack_imgs = med_filt_z_stack(stack_imgs)
    reg_medfilt_imgs = np.zeros_like(stack_imgs)
    reg_medfilt_imgs[0,:,:] = medfilt_stack_imgs[0,:,:]
    for i in range(1,num_planes):
        shift,_,_ = skimage.registration.phase_cross_correlation(reg_medfilt_imgs[i-1,:,:], medfilt_stack_imgs[i,:,:], normalization=None)
        reg_medfilt_imgs[i,:,:] = scipy.ndimage.shift(medfilt_stack_imgs[i,:,:], shift)
        reg_stack_imgs[i,:,:] = scipy.ndimage.shift(stack_imgs[i,:,:], shift)
    return reg_stack_imgs

def find_first_experiment_id_from_ophys_experiment_id(ophys_experiment_id):
    """Find the first experiment ID from the ophys experiment ID.
    Assume experiment IDs are assigned chronologically.

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID

    Returns
    -------
    int
        the first experiment ID
    """
    # Get ophys container id from ophys_experiment id
    ophys_container_id = mqc_from_lims.get_ophys_container_id_for_ophys_experiment_id(ophys_experiment_id)
    # Get all ophys_experiment_ids from the ophys container id
    oeid_df = mqc_from_lims.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)
    # Pick the first one after sorting the ids (assume experiment ids are generated in ascending number)
    first_oeid = np.sort(oeid_df.ophys_experiment_id.values)[0]
    return first_oeid

def get_registered_z_stack_step(ophys_experiment_id, number_of_z_planes=81, number_of_repeats=20):
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
    local_zstack_path = get_local_zstack_path_mqc(ophys_experiment_id)
    h = h5py.File(local_zstack_path, 'r')
    local_z_stack = h['data'][:]
    total_num_frames = local_z_stack.shape[0]
    assert total_num_frames == number_of_z_planes * number_of_repeats

    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = local_z_stack[range(plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane = average_reg_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)
    zstack_reg = reg_between_planes(np.array(mean_local_zstack_reg))
    return zstack_reg

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
        filtered_z_stack.append(cv2.medianBlur(image.astype(np.uint16), kernel_size))
    return np.array(filtered_z_stack)

# def gaus_filt_z_stack(z_stack):
#     filtered_z_stack = []
#     for image in z_stack:
#         filtered_z_stack.append(scipy.ndimage.gaussian_filter(image, 2))
#     return np.array(filtered_z_stack)

def moving_average(data1d, window=10):
    ''' Moving average
    Input:
        data1d (1d array): data to smooth
        window (int): window for smoothing. 
    Output:
        avg_data (1d array): smoothed data
    '''

    assert len(data1d.shape)==1
    avg_data = np.zeros(len(data1d))
    for i in range(len(data1d)):
        min_idx = np.max([i-window//2,0])
        max_idx = np.min([i+window//2,len(data1d)])
        avg_data[i] = np.mean(data1d[min_idx:max_idx])
    return avg_data

def image_normalization_uint16(image):
    """Normalize 2D image and convert to uint16
    Prevent saturation.

    Args:
        image (np.ndarray): input image (2D)
    Return:
        norm_image (np.ndarray)
    """
    clip_image = np.clip(image, np.percentile(image,0.2), np.percentile(image,99.8))
    norm_image = (clip_image - np.amin(clip_image)) / (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    uint16_image = ((norm_image+0.05)*np.iinfo(np.uint16).max*0.9).astype(np.uint16)
    return uint16_image

def get_valid_xy_range(fov):
    """ Get valid xy range from the fov
    For now, select the center of the FOV (2/3 of each dimension)
    TODO: Get better mask using suite2p registration result

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image

    Returns
    -------
    list, list
        list of y range and x range to use
    """
    len_y, len_x = fov.shape
    range_y = [len_y//6, -len_y//6]
    range_x = [len_x//6, -len_x//6]
    return range_y, range_x

def fov_stack_register_phase_correlation(fov, stack):
    """ Reigster FOV to each plane in the stack

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image
    stack : np.ndarray (3d)
        stack images

    Returns
    -------
    np.ndarray (3d)
        stack of FOV registered to each plane in the input stack
    np.array (1d)
        correlation coefficient between the registered fov and the stack in each plane
    list
        list of translation shifts
    """
    fov_reg_stack = np.zeros_like(stack)
    corrcoef = np.zeros(stack.shape[0])
    shift_list = []
    for pi in range(stack.shape[0]):
        shift,_,_ = skimage.registration.phase_cross_correlation(stack[pi,:,:], fov, normalization=None)
        fov_reg = scipy.ndimage.shift(fov, shift)
        fov_reg_stack[pi,:,:] = fov_reg
        valid_y, valid_x = np.where(fov_reg>0)
        corrcoef[pi] = np.corrcoef(stack[pi,valid_y,valid_x].flatten(), fov_reg[valid_y,valid_x].flatten())[0,1]
        shift_list.append(shift)
    return fov_reg_stack, corrcoef, shift_list

def select_best_registered_fov(fov_reg_stack, zstack):
    """ Select the best registered FOV to the z-stack,
    and return both only with valid pixels (based on the best registered FOV)

    Parameters
    ----------
    fov_reg_stack : np.ndarray (3d)
        stack of fov registered to each plane in the z-stack
    zstack : np.ndarray (3d)
        z-stack

    Returns
    -------
    np.ndarray (2d)
        the best reigstered fov (cropped)
    np.ndarray (3d)
        cropped z-stack
    int
        matched plane index
    """
    # Calcualte correlation in each plane
    corrcoef = np.zeros(fov_reg_stack.shape[0])
    for pi in range(fov_reg_stack.shape[0]):
        fov_reg = fov_reg_stack[pi,:,:]
        valid_y, valid_x = np.where(fov_reg>0)
        corrcoef[pi] = np.corrcoef(zstack[pi,valid_y,valid_x].flatten(), fov_reg[valid_y,valid_x].flatten())[0,1]
    corrcoef = moving_average(corrcoef, 5)
    # Get the plane with highest correlation
    matched_plane_ind = np.argmax(corrcoef)
    best_fov = fov_reg_stack[matched_plane_ind,:,:]
    return best_fov, matched_plane_ind

def fov_stack_register_rigid(fov, stack):
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

    Returns
    -------
    np.ndarray (1D)
        matched plane index
    
    """
    sr = StackReg(StackReg.RIGID_BODY)

    # apply CLAHE and normalize to make it uint16 (for StackReg)
    fov_clahe = image_normalization_uint16(skimage.exposure.equalize_adapthist(fov.astype(np.uint16))) # normalization to make it uint16
    stack_clahe = np.zeros_like(stack)
    for pi in range(stack.shape[0]):
        stack_clahe[pi,:,:] = image_normalization_uint16(skimage.exposure.equalize_adapthist(stack[pi,:,:].astype(np.uint16)))

    # Initialize
    corrcoef_first = np.zeros(stack_clahe.shape[0])
    fov_reg_stack = np.zeros_like(stack)
    tmat_list = []
    for pi in range(len(corrcoef_first)):
        tmat = sr.register(stack_clahe[pi,:,:], fov_clahe)
        # Apply the transformation matrix to the FOV registered using phase correlation
        fov_reg = sr.transform(fov, tmat=tmat)
        valid_y, valid_x = np.where(fov_reg>0)
        corrcoef_first[pi] = np.corrcoef(stack[pi,valid_y,valid_x].flatten(), fov_reg[valid_y, valid_x].flatten())[0,1]
        fov_reg_stack[pi,:,:] = fov_reg
        tmat_list.append(tmat)

    corrcoef_first = moving_average(corrcoef_first, 5)
    max_pi_first = np.argmax(corrcoef_first)

    # Use the best registered FOV to sweep through the first z-stack again
    fov_final = fov_reg_stack[max_pi_first,:,:]
    valid_y, valid_x = np.where(fov_final>0)

    # Calculate correlation between each z-stack to the best RIGID_BODY registered FOV
    corrcoef_second = np.zeros_like(corrcoef_first)
    for pi in range(len(corrcoef_first)):
        corrcoef_second[pi] = np.corrcoef(stack[pi,valid_y,valid_x].flatten(), fov_final[valid_y,valid_x].flatten())[0,1]
    corrcoef_second = moving_average(np.array(corrcoef_second), 5)
    matched_plane_index = np.argmax(corrcoef_second)
    tmat = tmat_list[matched_plane_index]
    
    return matched_plane_index, corrcoef_second, tmat

def estimate_plane_from_ref_zstack(fov: np.array, ref_zstack: np.array):
    """Estimate matched plane of the FOV from the first z-stack.
    Use phase correlation, and then StackReg.RIGID_BODY.
    Apply CLAHE and uint16 conversion before RIGID_BODY registration.
    Apply the StackReg result to original fov (after phase correlation registration).

    Parameters
    ----------
    fov : np.array
        mean FOV
    ref_zstack : np.array
        local z-stack from the reference session corresponding to the FOV
    
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
    # Remove suite2p registration rolling edges
    # For now, select the center of the FOV (2/3 of each dimension)
    # TODO: Get better mask using suite2p registration result
    range_y, range_x = get_valid_xy_range(fov)
    fov_crop = fov[range_y[0]:range_y[1], range_x[0]:range_x[1]]
    ref_zstack_crop = ref_zstack[:,range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # Register using phase correlation (translation only)
    fov_reg_stack_pre, corrcoef_pre, shift_list = fov_stack_register_phase_correlation(fov_crop, ref_zstack_crop)
    corrcoef_pre = moving_average(corrcoef_pre)

    # if np.amax(corrcoef_pre) < 0.5:
    #     raise('Phase correlation did not work.')
    # else:
    if True:
        # TODO: Find a better measure to test first pass registration 
        # Select the best registered FOV
        fov_pre, matched_pi_pre = select_best_registered_fov(fov_reg_stack_pre, ref_zstack_crop)
        translation_shift = shift_list[matched_pi_pre]

        # Crop the best fov and zstack using valid indices
        valid_y, valid_x = np.where(fov_pre>0)
        fov_pre = fov_pre[min(valid_y):max(valid_y), min(valid_x):max(valid_x)]
        z_stack_for_rigid = ref_zstack_crop[:,min(valid_y):max(valid_y), min(valid_x):max(valid_x)] # this works because it is translation

        # Rigid-body registration (allowing rotation)
        # After applying CLAHE and uint16 image normalization
        matched_plane_index, corrcoef_second, rigid_tmat = fov_stack_register_rigid(fov_pre, z_stack_for_rigid)

        # Get the registered fovs matched to the input fov dimension
        registered_fov_pc = scipy.ndimage.shift(fov, translation_shift)
        sr = StackReg(StackReg.RIGID_BODY)
        registered_fov = sr.transform(registered_fov_pc, tmat=rigid_tmat)

    return matched_plane_index, corrcoef_second, registered_fov, corrcoef_pre, registered_fov_pc

def estimate_plane_from_session_zstack(fov: np.array, session_zstack: np.array):
    """Estimate matched depth of the FOV to the session z-stack.
    Use phase correlation.

    Args:
        session_zstack (np.array): local z-stack from the same session as for the FOV.
        fov (np.array): mean FOV
        
    """
    fov_reg = np.zeros_like(session_zstack)
    corrcoef_pre = np.zeros(session_zstack.shape[0])
    for pi in range(session_zstack.shape[0]):
        shift,_,_ = skimage.registration.phase_cross_correlation(session_zstack[pi,:,:], fov, normalization='phase')
        shifted_fov = scipy.ndimage.shift(fov, shift)
        fov_reg[pi,:,:] = shifted_fov

        valid_y, valid_x = np.where(shifted_fov>0)
        corrcoef_pre[pi] = np.corrcoef(session_zstack[pi,valid_y,valid_x].flatten(), shifted_fov[valid_y,valid_x].flatten())[0,1]
    # if np.amax(corrcoef_pre) < 0.5:
    #     raise('Phase correlation did not work.')
    # else:
    if True:
    #TODO: Find a better way to flag if the registration worked.
        # Find the best-matched FOV translation
        corrcoef_pre = moving_average(np.array(corrcoef_pre), 5)
        max_pi = np.argmax(corrcoef_pre)
        fov = fov_reg[max_pi,:,:]
        valid_y, valid_x = np.where(fov>0)
        fov = fov[min(valid_y):max(valid_y), min(valid_x):max(valid_x)]
        session_zstack = session_zstack[:,min(valid_y):max(valid_y), min(valid_x):max(valid_x)] # this works because it is translation
        # Sweep through the stack to get the best-matched plane
        corrcoef_second = np.zeros_like(corrcoef_pre)
        for pi in range(len(corrcoef_pre)):
            corrcoef_second[pi] = np.corrcoef(session_zstack[pi,:,:].flatten(), fov.flatten())[0,1]
        corrcoef_second = moving_average(np.array(corrcoef_second), 5)
    matched_plane_index = np.argmax(corrcoef_second)
    registered_fov = fov_reg[max_pi,:,:]
    return matched_plane_index, corrcoef_second, registered_fov, corrcoef_pre

def get_registered_zstack(ophys_experiment_id, ophys_experiment_dir):
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
        reg_zstack = get_registered_z_stack_step(ophys_experiment_id)
        # Save the registered first z-stack
        with h5py.File(reg_zstack_fp, 'w') as h:
            h.create_dataset('data', data=reg_zstack)
    return reg_zstack

def estimate_matched_plane_from_ref_exp(oeid_reg, oeid_ref, segment_minute = 10):
    """Estimate the matched plane indices from the reference experiment.
    Use two-step registration (first phase correlation, then refine using StackReg RIBID_BODY registration).


    Parameters
    ----------
    oeid_reg : int
        ophyse experiment ID to estimate the plane
    oeid_ref : int
        ophys experiment ID of the reference experiment
    segment_minute : int, optional
        lenght of each segment from the experiment video (minutes), default 10

    Returns
    -------
    np.ndarray (1D)
        Indice of matched plane for each segment of the video
    np.ndarray (2D)
        Final correlation coefficient swept through reference z-stack planes for each segment
    np.ndarray (3D)
        FOV of each segment matched to the stack
    np.ndarray (2D)
        First-pass correlation coefficient swept through reference z-stack planes for each segment
    np.ndarray (3D)
        First-pass FOV of each segment matched to the stack
    """
    # First, check if they are in the same container
    container_id_ref = mqc_from_lims.get_ophys_container_id_for_ophys_experiment_id(oeid_ref)
    container_id_reg = mqc_from_lims.get_ophys_container_id_for_ophys_experiment_id(oeid_reg)
    assert container_id_ref == container_id_reg

    base_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\QC\FOV_matching\LAMF_Ai195_3'.replace('\\','/'))
    container_dp = base_dir / f'container_{container_id_ref}'
    exp_ref_dp = container_dp / f'experiment_{oeid_ref}'
    exp_reg_dp = container_dp / f'experiment_{oeid_reg}'

    # Get the saved result if there is
    plane_est_fn = f'{oeid_reg}_plane_estimated_from_{oeid_ref}.h5'
    plane_est_fp = exp_reg_dp / plane_est_fn
    if os.path.isfile(plane_est_fp):
        with h5py.File(plane_est_fp, 'r') as h:
            matched_plane_indice = h['matched_plane_indice'][:]
            corrcoef = h['corrcoef'][:]
            registered_fov = h['registered_fov'][:]
            corrcoef_pre = h['corrcoef_pre'][:]
            registered_fov_pre = h['registered_fov_pre'][:]
    else:
        ref_zstack = get_registered_zstack(oeid_ref, exp_ref_dp)
        segment_mean_images = get_segment_mean_images(oeid_reg, segment_minute = segment_minute)
        num_segments = segment_mean_images.shape[0]
        # Initialize
        matched_plane_indice = np.zeros(num_segments)
        corrcoef_list = []
        registered_fov_list = []
        corrcoef_pre_list = []
        registered_fov_pre_list = []
        for i in range(num_segments):
            matched_plane_index, corrcoef, registered_fov, corrcoef_pre, registered_fov_pre = \
                estimate_plane_from_ref_zstack(segment_mean_images[i,:,:], ref_zstack)
            # Gather data
            matched_plane_indice[i] = matched_plane_index
            corrcoef_list.append(corrcoef)
            registered_fov_list.append(registered_fov)
            corrcoef_pre_list.append(corrcoef_pre)
            registered_fov_pre_list.append(registered_fov_pre)
        # Rename gathered data
        corrcoef = np.asarray(corrcoef_list)
        registered_fov = np.asarray(registered_fov_list)
        corrcoef_pre = np.asarray(corrcoef_pre_list)
        registered_fov_pre = np.asarray(registered_fov_pre_list)
        # Save the result
        if not os.path.isdir(plane_est_fp.parent):
            os.makedirs(plane_est_fp.parent)
        with h5py.File(plane_est_fp, 'w') as h:
            h.create_dataset('matched_plane_indice', data=matched_plane_indice)
            h.create_dataset('corrcoef', data=corrcoef)
            h.create_dataset('registered_fov', data=registered_fov)
            h.create_dataset('corrcoef_pre', data=corrcoef_pre)
            h.create_dataset('registered_fov_pre', data=registered_fov_pre)
    return matched_plane_indice, corrcoef, registered_fov, corrcoef_pre, registered_fov_pre

def estimate_matched_plane_from_same_exp(ophys_experiment_id,  segment_minute = 10):
    """Estimate the matched plane indices from the same experiment z-stack.
    Use phase correlation

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    segment_minute : int, optional
        lenght of each segment from the experiment video (minutes), default 10

    Returns
    -------
    np.ndarray (1D)
        Indice of matched plane for each segment of the video
    np.ndarray (2D)
        Final correlation coefficient swept through reference z-stack planes for each segment
    np.ndarray (3D)
        FOV of each segment matched to the stack
    np.ndarray (2D)
        First-pass correlation coefficient swept through reference z-stack planes for each segment
    """
    base_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\QC\FOV_matching\LAMF_Ai195_3'.replace('\\','/'))
    container_id = mqc_from_lims.get_ophys_container_id_for_ophys_experiment_id(ophys_experiment_id)

    container_dp = base_dir / f'container_{container_id}'
    exp_dp = container_dp / f'experiment_{ophys_experiment_id}'

    # Get the saved result if there is
    plane_est_fn = f'{ophys_experiment_id}_plane_estimated_from_session.h5'
    plane_est_fp = exp_dp / plane_est_fn
    if os.path.isfile(plane_est_fp):
        with h5py.File(plane_est_fp, 'r') as h:
            matched_plane_indice = h['matched_plane_indice'][:]
            corrcoef = h['corrcoef'][:]
            registered_fov = h['registered_fov'][:]
            corrcoef_pre = h['corrcoef_pre'][:]
    else:
        zstack = get_registered_zstack(ophys_experiment_id, exp_dp)
        segment_mean_images = get_segment_mean_images(ophys_experiment_id, segment_minute=segment_minute)
        # Initialize
        num_segments = segment_mean_images.shape[0]
        matched_plane_indice = np.zeros(num_segments)
        corrcoef_list = []
        registered_fov_list = []
        corrcoef_pre_list = []
        for i in range(num_segments):
            matched_plane_index, corrcoef, registered_fov, corrcoef_pre = \
                estimate_plane_from_session_zstack(segment_mean_images[i,:,:], zstack)
            # Gather data
            matched_plane_indice[i] = matched_plane_index
            corrcoef_list.append(corrcoef)
            registered_fov_list.append(registered_fov)
            corrcoef_pre_list.append(corrcoef_pre)
        # Rename gathered data
        corrcoef = np.asarray(corrcoef_list)
        registered_fov = np.asarray(registered_fov_list)
        corrcoef_pre = np.asarray(corrcoef_pre_list)
        # Save the result
        if not os.path.isdir(plane_est_fp.parent):
            os.makedirs(plane_est_fp.parent)
        with h5py.File(plane_est_fp, 'w') as h:
            h.create_dataset('matched_plane_indice', data=matched_plane_indice)
            h.create_dataset('corrcoef', data=corrcoef)
            h.create_dataset('registered_fov', data=registered_fov)
            h.create_dataset('corrcoef_pre', data=corrcoef_pre)
    return matched_plane_indice, corrcoef, registered_fov, corrcoef_pre