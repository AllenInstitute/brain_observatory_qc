import h5py
import numpy as np
import brain_observatory_qc.data_access.from_lims as from_lims
from brain_observatory_qc.pipeline_dev import paired_plane_registration as ppr
from scipy import ndimage
import skimage
from cellpose import models as cp_models


def decrosstalk_movie(oeid, paired_reg_fn, filter_sigma_um=25, num_epochs=10, num_frames_avg=1000,
                      grid_interval=0.01, max_grid_val=0.3, return_recon=True, return_epoch_results=False):
    """Run FOV-based decrosstalk using constrained ICA on motion corrected movie data
    
    Parameters
    ----------
    oeid : int
        ophys experiment id
    paired_reg_fn : str, Path
        path to paired registration file
        TODO: Once paired plane registration pipeline is finalized,
        this parameter can be removed or replaced with paired_oeid
    filter_sigma_um : float or list
        sigma for gaussian filter in microns
        if list, then sigma for each epoch (dynamic filtering)
    num_epochs : int
        number of epochs to run
    num_frames_avg : int
        number of frames to average for each epoch
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search
    return_recon : bool
        whether to return reconstructed images
    return_epoch_results : bool
        whether to return results for each epoch
    
    Returns
    -------
    recon_signal_data : np.ndarray
        reconstructed signal data
    alpha_list : list
        list of alpha values for each epoch
    beta_list : list
        list of beta values for each epoch
    """
    signal_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    with h5py.File(signal_fn, 'r') as f:
        data_length = f['data'].shape[0]
    epoch_interval = data_length // (num_epochs+1)  # +1 to avoid the very first frame (about half of each epoch)
    num_frames_avg = min(num_frames_avg, epoch_interval)
    start_frames = [num_frames_avg//2 + i * epoch_interval for i in range(num_epochs)]
    assert start_frames[-1] + num_frames_avg < data_length

    pixel_size_um = from_lims.get_pixel_size_um(oeid)
    if filter_sigma_um is None:
        filter_sigma_um = 0
    assert len(filter_sigma_um) > 0
    if len(filter_sigma_um) > 1:
        assert len(filter_sigma_um) == num_epochs
        sigma_list = [sigma / pixel_size_um for sigma in filter_sigma_um]
    elif filter_sigma_um == 0:
        sigma_list = [None] * num_epochs
    else:
        sigma_list = [filter_sigma_um / pixel_size_um] * num_epochs

    alpha_list = []
    beta_list = []
    if return_epoch_results:
        signal_mean_list = []
        paired_mean_list = []
        recon_signal_list = []
        recon_paired_list = []
    for i in range(num_epochs):
        sigma = sigma_list[i]
        alpha, beta, signal_mean, paired_mean, recon_signal, recon_paired = \
            decrosstalk_single_mean_image(oeid, paired_reg_fn, sigma,
                                         start_frame=start_frames[i],
                                         num_frames_avg=num_frames_avg,
                                         grid_interval=grid_interval,
                                         max_grid_val=max_grid_val,
                                         get_recon=False)
        alpha_list.append(alpha)
        beta_list.append(beta)
        if return_epoch_results:
            signal_mean_list.append(signal_mean)
            paired_mean_list.append(paired_mean)
            recon_signal_list.append(recon_signal)
            recon_paired_list.append(recon_paired)
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    if return_recon:
        with h5py.File(signal_fn, 'r') as f:
            signal_data = f['data'][:]    
        with h5py.File(paired_reg_fn, 'r') as f:
            paired_data = f['data'][:]
        recon_signal_data = np.zeros_like(signal_data)
        for i in range(data_length):
            recon_signal_data[i, :, :] = apply_mixing_matrix(alpha, beta, signal_data[i, :, :], paired_data[i, :, :])[0]
    else:
        recon_signal_data = None
    if return_epoch_results:
        return recon_signal_data, alpha_list, beta_list, signal_mean_list, paired_mean_list, recon_signal_list, recon_paired_list
    else:
        return recon_signal_data, alpha_list, beta_list


def decrosstalk_single_mean_image(oeid, paired_reg_fn, sigma, start_frame=500, num_frames_avg=300,
                                  motion_buffer=5, grid_interval=0.01, max_grid_val=0.3, get_recon=True):
    """Run FOV-based decrosstalk using constrained ICA on a pair of mean images

    Parameters
    ----------
    oeid : int
        ophys experiment id
    paired_reg_fn : str, Path
        path to paired registration file
        TODO: Once paired plane registration pipeline is finalized,
        this parameter can be removed or replaced with paired_oeid
    sigma : float
        sigma for gaussian filter in pixels
        if None, then no filtering
    start_frame : int
        starting frame index
    num_frames_avg : int
        number of frames to average
    motion_buffer : int
        number of pixels to crop from the edge of the motion corrected image (for nonrigid registration)
        TODO: Get this information from nonrigid registration parameters
        (e.g., from suite2p ops, maxShiftNR)
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search
    get_recon : bool
        whether to return reconstructed images

    Returns
    -------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    """
    signal_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    with h5py.File(signal_fn, 'r') as f:
        signal_mean = f['data'][start_frame:start_frame+num_frames_avg].mean(axis=0)
    with h5py.File(paired_reg_fn, 'r') as f:
        paired_mean = f['data'][start_frame:start_frame+num_frames_avg].mean(axis=0)

    p1y, p1x = ppr.get_motion_correction_crop_xy_range_from_both_planes(oeid)
    signal_mean = signal_mean[p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                              p1x[0]+motion_buffer:p1x[1]-motion_buffer]
    paired_mean = paired_mean[p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                              p1x[0]+motion_buffer:p1x[1]-motion_buffer]

    if sigma is None:
        signal_filtered = signal_mean
        paired_filtered = paired_mean
    else:
        signal_filtered = signal_mean - ndimage.gaussian_filter(signal_mean, sigma)
        paired_filtered = paired_mean - ndimage.gaussian_filter(paired_mean, sigma)

    # TODO: change grid searching to constrained gradient descent to speed up
    alpha, beta ,_, _ = unmixing_using_grid_search(signal_filtered, paired_filtered,
                                                   grid_interval=grid_interval, max_grid_val=max_grid_val)
    if get_recon:
        recon_signal, recon_paired = apply_mixing_matrix(alpha, beta, signal_mean, paired_mean)
    else:
        recon_signal, recon_paired = None, None
    
    return alpha, beta, signal_mean, paired_mean, recon_signal, recon_paired


def unmixing_using_grid_search(signal_mean, paired_mean, yrange_for_mi=None, xrange_for_mi=None,
                               grid_interval=0.01, max_grid_val=0.3):
    """Calculate unmixing matrix using grid search for minimum mutual information
    
    Parameters:
    -----------
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    yrange_for_mi : list
        y range of the image to calculate mutual information
    xrange_for_mi : list
        x range of the image to calculate mutual information
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search

    Returns:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    """
    assert signal_mean.shape == paired_mean.shape
    if yrange_for_mi is None:
        yrange_for_mi = [0, signal_mean.shape[0]]
    if xrange_for_mi is None:
        xrange_for_mi = [0, signal_mean.shape[1]]
    crop_signal_mean = signal_mean[yrange_for_mi[0]:yrange_for_mi[1], xrange_for_mi[0]:xrange_for_mi[1]]
    crop_paired_mean = paired_mean[yrange_for_mi[0]:yrange_for_mi[1], xrange_for_mi[0]:xrange_for_mi[1]]
    _, _, mi_values, _, _, ab_pair = \
        calculate_mutual_info_grid_search(crop_signal_mean, crop_paired_mean, grid_interval=grid_interval, max_grid_val=max_grid_val)
    min_idx = np.argmin(mi_values)
    alpha = ab_pair[min_idx][0]
    beta = ab_pair[min_idx][1]
    recon_signal, recon_paired = apply_mixing_matrix(alpha, beta, signal_mean, paired_mean)
        
    return alpha, beta, recon_signal, recon_paired


def calculate_mutual_info_grid_search(signal_mean, paired_mean, grid_interval=0.01, max_grid_val=0.3):
    """Calculate mutual information using grid search
    
    Parameters:
    -----------
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search

    Returns:
    -----------
    alpha_list : np.array
        list of alpha values
    beta_list : np.array
        list of beta values
    mi_values : np.array
        list of mutual information values
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    ab_pair : np.array
        list of alpha and beta pairs
    """
    data = np.vstack((signal_mean.ravel(), paired_mean.ravel()))

    alpha_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    beta_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    recon_signal = []
    recon_paired = []
    ab_pair = []
    mi_values = []
    for alpha in alpha_list:
        for beta in beta_list:
            temp_mixing = np.array([[1-alpha, alpha], [beta, 1-beta]])
            temp_unmixing = np.linalg.inv(temp_mixing)
            temp_unmixed_data = np.dot(temp_unmixing, data)
            recon_signal.append(temp_unmixed_data[0,:].reshape(signal_mean.shape))
            recon_paired.append(temp_unmixed_data[1,:].reshape(signal_mean.shape))
            temp_mi = skimage.metrics.normalized_mutual_information(temp_unmixed_data[0,:],
                                                                    temp_unmixed_data[1,:])
            mi_values.append(temp_mi)
            ab_pair.append([alpha, beta])
    recon_signal = np.array(recon_signal)
    recon_paired = np.array(recon_paired)
    mi_values = np.array(mi_values)

    return alpha_list, beta_list, mi_values, recon_signal, recon_paired, ab_pair


def apply_mixing_matrix(alpha, beta, signal_mean, paired_mean):
    """Apply mixing matrix to the mean images to get reconstructed images
    
    Parameters:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane

    Returns:
    -----------
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    """
    mixing_mat = [[1-alpha, alpha], [beta, 1-beta]]
    unmixing_mat = np.linalg.inv(mixing_mat)
    raw_data = np.vstack([signal_mean.ravel(), paired_mean.ravel()])
    recon_data = np.dot(unmixing_mat, raw_data)
    recon_signal = recon_data[0, :].reshape(signal_mean.shape)
    recon_paired = recon_data[1, :].reshape(paired_mean.shape)
    return recon_signal, recon_paired

#######################
## Optimal sigma search using mutual information grid search
#######################
def find_optimal_sigma_exp(oeid, paired_reg_fn, num_epochs=10, fixed_sigma=True):
    signal_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    with h5py.File(signal_fn, 'r') as f:
        data_length = f['data'].shape[0]
    epoch_interval = data_length // (num_epochs+1)  # +1 to avoid the very first frame (about half of each epoch)
    num_frames_avg = min(num_frames_avg, epoch_interval)
    start_frames = [num_frames_avg//2 + i * epoch_interval for i in range(num_epochs)]
    assert start_frames[-1] + num_frames_avg < data_length

    mi_filtered_all_list = []
    mi_raw_list = []
    for start_frame in start_frames:
        signal_mean, paired_mean = get_signal_paired_mean(oeid, paired_reg_fn, start_frame=start_frame, num_frames_to_run=num_frames_avg)
        _, other_results = find_optimal_sigma_pair_images_constrained_ica(oeid, signal_mean, paired_mean)
        mi_filtered_all_list.append(other_results['mi_filtered_all'])
        mi_raw_list.append(other_results['mi_raw'])
    if fixed_sigma:
        mi_filtered_all = np.mean(np.stack(mi_filtered_all_list), axis=0)
        mi_raw = np.mean(mi_raw_list)
        if mi_raw < mi_filtered_all.min():
            sigma_optimal = 0
        else:
            sigma_optimal = other_results['sigma_list_um'][np.argmin(mi_filtered_all)]
    else:
        mi_filtered_all = np.stack(mi_filtered_all_list)
        mi_raw = np.array(mi_raw_list)
        sigma_optimal = np.zeros(num_epochs)
        for i in range(num_epochs):
            if mi_raw[i] < mi_filtered_all[i].min():
                sigma_optimal[i] = 0
            else:
                sigma_optimal[i] = other_results['sigma_list_um'][np.argmin(mi_filtered_all[i])]
    return sigma_optimal


def get_signal_paired_mean(oeid, paired_reg_fn,
                           start_frame=1000, num_frames_to_run=1000, 
                           nonrigid_buffer=5, use_paired_reg=True):
    signal_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    if not use_paired_reg:
        paired_id = ppr.get_paired_plane_id(oeid)
        paired_reg_fn = from_lims.get_motion_corrected_movie_filepath(paired_id)
    frames_to_test = np.arange(start_frame, start_frame + num_frames_to_run)
    with h5py.File(signal_fn, 'r') as signal_h5:
        signal_frames = signal_h5['data'][frames_to_test]
    with h5py.File(paired_reg_fn, 'r') as paired_h5:
        paired_frames = paired_h5['data'][frames_to_test]

    yrange, xrange = ppr.get_motion_correction_crop_xy_range_from_both_planes(oeid)
    
    signal_mean = signal_frames[:, yrange[0] + nonrigid_buffer : yrange[1] - nonrigid_buffer, 
                                xrange[0] + nonrigid_buffer : xrange[1] - nonrigid_buffer].mean(axis=0)
    paired_mean = paired_frames[:, yrange[0] + nonrigid_buffer : yrange[1] - nonrigid_buffer, 
                                xrange[0] + nonrigid_buffer : xrange[1] - nonrigid_buffer].mean(axis=0)
    return signal_mean, paired_mean        


def find_optimal_sigma_pair_images_constrained_ica(oeid, signal_mean, paired_mean,
                                   sigma_list_um=np.arange(5,125,5), # in um
                                   ):
    
    # Create top intensity ROIs and bounding boxes
    pix_size = from_lims.get_pixel_size_um(oeid)
    signal_top_masks, paired_top_masks = get_signal_paired_top_masks(signal_mean, paired_mean,
                                                                     pix_size=pix_size)
    signal_bb_masks = get_bounding_box(signal_top_masks)
    paired_bb_masks = get_bounding_box(paired_top_masks)
    bb_masks = np.concatenate([signal_bb_masks, paired_bb_masks])

    # Getting template mutual information data from the bounding boxes
    mi_arr = np.zeros(len(bb_masks))
    for bi, mask in enumerate(bb_masks):
        bb_yx = np.where(mask)
        temp_mi = skimage.metrics.normalized_mutual_information(signal_mean[bb_yx],
                                                                paired_mean[bb_yx])
        mi_arr[bi] = temp_mi
    mi_template = mi_arr.copy()

    # raw data unmixing
    alpha_raw, beta_raw, signal_dc_raw, paired_dc_raw = unmixing_using_grid_search(signal_mean, paired_mean)
    # unmixed raw data mutual information normalized to the raw data template
    mi_arr = np.zeros(len(bb_masks))
    for bi, mask in enumerate(bb_masks):
        bb_yx = np.where(mask)
        temp_mi = skimage.metrics.normalized_mutual_information(signal_dc_raw[bb_yx],
                                                                paired_dc_raw[bb_yx])
        mi_arr[bi] = temp_mi
    mi_raw_norm = mi_arr / mi_template
    mi_raw = np.mean(mi_raw_norm)
    mi_raw_std = np.std(mi_raw_norm)

    # filtered data unmixing
    
    sigma_list_pix = sigma_list_um / pix_size
    signal_dc_filtered_list = []
    paired_dc_filtered_list = []
    alpha_list = []
    beta_list = []
    for si, sigma_pix in enumerate(sigma_list_pix):
        signal_filtered = signal_mean - ndimage.gaussian_filter(signal_mean, sigma=sigma_pix)
        paired_filtered = paired_mean - ndimage.gaussian_filter(paired_mean, sigma=sigma_pix)
        alpha, beta, _, _ = unmixing_using_grid_search(signal_filtered, paired_filtered)
        recon_signal, recon_paired = apply_mixing_matrix(alpha, beta, signal_mean, paired_mean)
        signal_dc_filtered_list.append(recon_signal)
        paired_dc_filtered_list.append(recon_paired)
        alpha_list.append(alpha)
        beta_list.append(beta)
    # unmixed filtered data mutual information normalized to the raw data template
    mi_filtered_all = np.zeros(len(sigma_list_um))
    mi_filtered_std_all = np.zeros(len(sigma_list_um))
    for si in range(len(sigma_list_um)):
        signal_dc_filtered = signal_dc_filtered_list[si]
        paired_dc_filtered = paired_dc_filtered_list[si]
        mi_arr = np.zeros(len(bb_masks))
        for bi, mask in enumerate(bb_masks):
            bb_yx = np.where(mask)
            temp_mi = skimage.metrics.normalized_mutual_information(signal_dc_filtered[bb_yx],
                                                                    paired_dc_filtered[bb_yx])
            mi_arr[bi] = temp_mi
        mi_arr_norm = mi_arr / mi_template
        mi_filtered_all[si] = np.mean(mi_arr_norm)
        mi_filtered_std_all[si] = np.std(mi_arr_norm)
    
    sigma_optimal = sigma_list_um[np.argmin(mi_filtered_all)]
    if mi_raw < mi_filtered_all.min():
        sigma_optimal = 0
    other_results = {'mi_raw': mi_raw,
                     'mi_raw_std': mi_raw_std,
                     'mi_filtered_all': mi_filtered_all,
                     'mi_filtered_std_all': mi_filtered_std_all,
                     'sigma_list_um': sigma_list_um,
                     'signal_dc_filtered_list': signal_dc_filtered_list,
                     'paired_dc_filtered_list': paired_dc_filtered_list,
                     'signal_dc_raw': signal_dc_raw,
                     'paired_dc_raw': paired_dc_raw,
                     'alpha_list': alpha_list,
                     'beta_list': beta_list,
                     'alpha_raw': alpha_raw,
                     'beta_raw': beta_raw,
                     }
    return sigma_optimal, other_results


def filter_dendrite(masks, dendrite_diameter_pix=10/0.78):
    """ Filter dendrites from masks based on area threshold

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value
    dendrite_diameter: int, diameter of dendrite in pix
    
    Returns
    -------
    filtered_mask: 2d array, after filtering
        Note - the filtered mask is not necessarily contiguous
    """
    dendrite_radius = dendrite_diameter_pix / 2
    area_threshold = np.pi * dendrite_radius**2
    num_roi = np.max(masks)
    filtered_mask = masks.copy()
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        roi_area = np.sum(roi_mask)
        if roi_area < area_threshold:
            filtered_mask[roi_mask] = 0
    return filtered_mask


def filter_border_roi(masks, buffer_pix=5):
    """ Filter ROIs that are too close to the border of the FOV

    Input Parameters
    ----------------
    masks: 2d array, each ROI has a unique integer value
    border_width_pix: int, width of border in pix
    
    Returns
    -------
    filtered_mask: 2d array, after filtering
    """
    border_mask = np.zeros(masks.shape, dtype=bool)
    border_mask[:buffer_pix, :] = 1
    border_mask[-buffer_pix:, :] = 1
    border_mask[:, :buffer_pix] = 1
    border_mask[:, -buffer_pix:] = 1

    filtered_mask = masks.copy()    
    num_roi = np.max(masks)
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        if np.any(roi_mask & border_mask):
            filtered_mask[roi_mask] = 0

    return filtered_mask


def get_top_intensity_mask(mimg, mask, num_top_rois=15):
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    num_roi = len(roi_inds)
    if num_roi < num_top_rois:
        return mask
    else:
        mean_intensities = [mimg[mask==i].mean() for i in roi_inds]
        sorted_inds = np.argsort(mean_intensities)[::-1]
        top_inds = sorted_inds[:num_top_rois]
        top_ids = roi_inds[top_inds]

        top_mask = mask.copy()
        for i in roi_inds:
            if i not in top_ids:
                top_mask[mask==i] = 0
        return top_mask
    

def get_ranks_roi_inds(img, mask):
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    mean_intensities = np.array([img[mask==i].mean() for i in roi_inds])
    ranks = np.zeros_like(mean_intensities)
    ranks[np.argsort(mean_intensities)[::-1]] = np.arange(len(mean_intensities))
    return ranks, roi_inds


def reorder_mask(mask):
    mask_reordered = np.zeros_like(mask)
    roi_inds = np.setdiff1d(np.unique(mask), 0)
    for i, ind in enumerate(roi_inds):
        mask_reordered[mask==ind] = i+1
    return mask_reordered


def get_signal_paired_top_masks(signal_mean, paired_mean,
                                dendrite_diameter_um=10,
                                pix_size=0.78,
                                nrshiftmax=5,
                                overlap_threshold=0.7,
                                num_top_rois=15):
    model = cp_models.Cellpose(gpu=False, model_type='cyto')
    signal_masks, _, _, _ = model.eval(signal_mean, diameter=None, channels=[0,0])
    
    dendrite_diameter_px = dendrite_diameter_um / pix_size
    signal_masks_dendrite_filtered = filter_dendrite(signal_masks, dendrite_diameter_pix=dendrite_diameter_px)
    signal_masks_filtered = filter_border_roi(signal_masks_dendrite_filtered, buffer_pix=nrshiftmax)

    paired_masks, _, _, _ = model.eval(paired_mean, diameter=None, channels=[0,0])
    paired_masks_dendrite_filtered = filter_dendrite(paired_masks, dendrite_diameter_pix=dendrite_diameter_px)
    paired_masks_filtered = filter_border_roi(paired_masks_dendrite_filtered, buffer_pix=nrshiftmax)

    signal_masks_filtered = reorder_mask(signal_masks_filtered)
    paired_masks_filtered = reorder_mask(paired_masks_filtered)

    num_signal_masks = np.max(signal_masks_filtered)
    num_paired_masks = np.max(paired_masks_filtered)
    overlap_matrix = np.zeros((num_signal_masks, num_paired_masks))
    for i in range(1, num_signal_masks + 1):
        for j in range(1, num_paired_masks + 1):
            overlap_matrix[i-1, j-1] = np.sum((signal_masks_filtered == i) & (paired_masks_filtered == j)) / np.sum((signal_masks_filtered == i) | (paired_masks_filtered == j))

    signal_ranks, _ = get_ranks_roi_inds(signal_mean, signal_masks_filtered)
    paired_ranks, _ = get_ranks_roi_inds(paired_mean, paired_masks_filtered)

    signal_masks_overlap_filtered = signal_masks_filtered.copy()
    paired_masks_overlap_filtered = paired_masks_filtered.copy()
    # remove lower rank overlaps
    for si, pi in zip(np.where(overlap_matrix > overlap_threshold)[0],
                      np.where(overlap_matrix > overlap_threshold)[1]):
        rank_signal = signal_ranks[si]
        rank_paired = paired_ranks[pi]
        if rank_signal < rank_paired:
            paired_masks_overlap_filtered[paired_masks_overlap_filtered == pi+1] = 0
        else:
            signal_masks_overlap_filtered[signal_masks_overlap_filtered == si+1] = 0
    
    signal_top_masks = get_top_intensity_mask(signal_mean, signal_masks_overlap_filtered, num_top_rois=num_top_rois)
    paired_top_masks = get_top_intensity_mask(paired_mean, paired_masks_overlap_filtered, num_top_rois=num_top_rois)
    return signal_top_masks, paired_top_masks


def get_bounding_box(masks, area_extension_factor=2):
    bb_extension = np.sqrt(area_extension_factor * np.pi / 4)
    mask_inds = np.setdiff1d(np.unique(masks), 0)

    bb_masks = np.zeros((len(mask_inds), *masks.shape), dtype=np.uint16)
    for i, mask_i in enumerate(mask_inds):
        y, x = np.where(masks==mask_i)
        bb_y_tight = [y.min(), y.max()]
        bb_x_tight = [x.min(), x.max()]
        bb_y_tight_len = bb_y_tight[1] - bb_y_tight[0]
        bb_x_tight_len = bb_x_tight[1] - bb_x_tight[0]
        bb_y = [max(0, int(np.round(bb_y_tight[0] - bb_extension * bb_y_tight_len/2))),
                min(masks.shape[0],
                    int(np.round(bb_y_tight[1] + bb_extension * bb_y_tight_len / 2)))]
        bb_x = [max(0, int(np.round(bb_x_tight[0] - bb_extension * bb_x_tight_len/2))),
                min(masks.shape[1],
                    int(np.round(bb_x_tight[1] + bb_extension * bb_x_tight_len / 2)))]
        bb_masks[i, bb_y[0]:bb_y[1], bb_x[0]:bb_x[1]] = mask_i
    return bb_masks