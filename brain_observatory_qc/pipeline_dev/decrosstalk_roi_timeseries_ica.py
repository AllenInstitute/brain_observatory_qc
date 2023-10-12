import numpy as np
import matplotlib.pyplot as plt
import h5py
import skimage
import imageio
from cellpose import models as cp_models

import brain_observatory_qc.data_access.from_lims as from_lims
from brain_observatory_qc.pipeline_dev import paired_plane_registration as ppr


def decrosstalk_timeseries_exp(oeid, paired_reg_fn, dendrite_diameter_um=10, num_top_rois=15, nrshiftmax=5,
                               grid_interval=0.01, max_grid_val=0.3, return_recon=True):
    """Decrosstalk timeseries for an experiment

    Parameters
    ----------
    oeid : int
        ophys experiment id
    paired_reg_fn : str
        path to paired plane registration file
    dendrite_diameter_um : float
        diameter of dendrite in microns
    num_top_rois : int
        number of top intensity ROIs to keep
    nrshiftmax : int
        maximum number of pixels to shift ROIs to avoid border ROIs
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search
    return_recon : bool
        whether to return reconstructed signal data

    Returns
    -------
    recon_signal_data : np.array
        reconstructed signal data
    alpha_list : list
        list of alpha values for each ROI
    beta_list : list
        list of beta values for each ROI
    """
    timeseries_signal, timeseries_paired = get_time_series_from_paired_planes(oeid, paired_reg_fn,
                                                                              dendrite_diameter_um=dendrite_diameter_um,
                                                                              num_top_rois=num_top_rois,
                                                                              nrshiftmax=nrshiftmax)
    alpha_list, beta_list = constrained_ica_timeseries(timeseries_signal, timeseries_paired,
                                                       grid_interval=grid_interval,
                                                       max_grid_val=max_grid_val)
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    signal_fn = from_lims.get_motion_corrected_movie_filepath(oeid)

    if return_recon:
        with h5py.File(signal_fn, 'r') as f:
            signal_data = f['data'][:]
            data_length = signal_data.shape[0]
        with h5py.File(paired_reg_fn, 'r') as f:
            paired_data = f['data'][:]
        recon_signal_data = np.zeros_like(signal_data)
        for i in range(data_length):
            recon_signal_data[i, :, :] = apply_mixing_matrix(alpha, beta, signal_data[i, :, :], paired_data[i, :, :])[0]
    else:
        recon_signal_data = None
    return recon_signal_data, alpha_list, beta_list


def constrained_ica_timeseries(timeseries_signal, timeseries_paired,
                               method='grid_search', grid_interval=0.01, max_grid_val=0.3):
    """Constrained ICA for timeseries

    Parameters
    ----------
    timeseries_signal : np.array
        timeseries data from signal plane
    timeseries_paired : np.array
        timeseries data from paired plane
    method : str
        method for unmixing, either 'grid_search' or 'gradient_descent'
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search

    Returns
    -------
    alpha_list : list
        list of alpha values for each ROI
    beta_list : list
        list of beta values for each ROI
    """

    assert timeseries_signal.shape == timeseries_paired.shape
    num_roi = timeseries_signal.shape[0]
    alpha_list = np.zeros(num_roi)
    beta_list = np.zeros(num_roi)
    for i in range(num_roi):
        if method == 'grid_search':
            alpha, beta = unmixing_using_grid_search(timeseries_signal[i,:], timeseries_paired[i,:],
                                                     grid_interval=grid_interval, max_grid_val=max_grid_val)
        # elif method == 'gradient_descent':
        #     alpha, beta = unmixing_using_gradient_descent(timeseries_signal[i,:], timeseries_paired[i,:])
        alpha_list[i] = alpha
        beta_list[i] = beta
    return alpha_list, beta_list


def unmixing_using_grid_search(ts_signal, ts_paired, grid_interval=0.01, max_grid_val=0.3):
    """Unmixing using grid search

    Parameters
    ----------
    ts_signal : np.array
        timeseries data from signal plane
    ts_paired : np.array
        timeseries data from paired plane
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search

    Returns
    -------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    """
    assert len(ts_signal) == len(ts_paired)
    _, _, mi_values, ab_pair = \
        calculate_mutual_info_grid_search(ts_signal, ts_paired, grid_interval=grid_interval, max_grid_val=max_grid_val)
    min_idx = np.argmin(mi_values)
    alpha = ab_pair[min_idx][0]
    beta = ab_pair[min_idx][1]

    return alpha, beta


def calculate_mutual_info_grid_search(ts_signal, ts_paired, grid_interval=0.01, max_grid_val=0.3):
    """Calculate mutual information using grid search

    Parameters
    ----------
    ts_signal : np.array
        timeseries data from signal plane
    ts_paired : np.array
        timeseries data from paired plane
    grid_interval : float
        interval for grid search
    max_grid_val : float
        maximum value for grid search

    Returns
    -------
    alpha_list : list
        list of alpha values for each ROI
    beta_list : list
        list of beta values for each ROI
    mi_values : list
        list of mutual information values for each ROI
    ab_pair : list
        list of alpha and beta pairs for each ROI
    """
    assert len(ts_signal) == len(ts_paired)
    data = np.vstack((ts_signal, ts_paired))

    alpha_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    beta_list = np.arange(0, max_grid_val + grid_interval, grid_interval)
    mi_values = []
    ab_pair = []
    for alpha in alpha_list:
        for beta in beta_list:
            temp_mixing = np.array([[1-alpha, alpha], [beta, 1-beta]])
            # TODO: whiten them, match the pair (using correlation),
            # and then match the DC component (how?)
            temp_unmixing = np.linalg.inv(temp_mixing)
            temp_unmixed_data = np.dot(temp_unmixing, data)
            temp_mi = skimage.metrics.normalized_mutual_information(temp_unmixed_data[0,:],
                                                                    temp_unmixed_data[1,:])
            mi_values.append(temp_mi)
            ab_pair.append([alpha, beta])
    mi_values = np.array(mi_values)

    return alpha_list, beta_list, mi_values, ab_pair


def get_time_series_from_paired_planes(oeid, paired_reg_fn, dendrite_diameter_um=10, num_top_rois=15, nrshiftmax=5):
    """Get timeseries data from paired planes

    Parameters
    ----------
    oeid : int
        ophys experiment id
    paired_reg_fn : str
        path to paired plane registration file
    dendrite_diameter_um : float
        diameter of dendrite in microns
    num_top_rois : int
        number of top intensity ROIs to keep
    nrshiftmax : int
        maximum number of pixels to shift ROIs to avoid border ROIs

    Returns
    -------
    timeseries_signal : np.array
        timeseries data from signal plane
    timeseries_paired : np.array
        timeseries data from paired plane
    """    
    oeid_mask = get_top_mask_from_cellpose(oeid, dendrite_diameter_um=dendrite_diameter_um,
                                           num_top_rois=num_top_rois,
                                           nrshiftmax=nrshiftmax)
    roi_ids = np.setdiff1d(np.unique(oeid_mask), 0)
    oeid_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    with h5py.File(oeid_fn, 'r') as f:
        signal_data = f['data'][()]
    timeseries_signal = np.zeros((len(roi_ids), signal_data.shape[0]))
    for i, id in enumerate(roi_ids):
        timeseries_signal[i,:] = signal_data[:, oeid_mask==id].mean(axis=1)
    del signal_data
    with h5py.File(paired_reg_fn, 'r') as f:
        paired_data = f['data'][()]
    assert paired_data.shape[0] == timeseries_signal.shape[1]
    
    timeseries_paired = np.zeros((len(roi_ids), paired_data.shape[0]))
    for i, id in enumerate(roi_ids):
        timeseries_paired[i,:] = paired_data[:, oeid_mask==id].mean(axis=1)
    del paired_data

    return timeseries_signal, timeseries_paired


def get_top_mask_from_cellpose(oeid, dendrite_diameter_um=10, num_top_rois=15, nrshiftmax=5):
    """ Get top intensity masks from cellpose segmentation

    Parameters
    ----------------
    oeid: int
        ophys experiment id
    dendrite_diameter_um: float
        diameter of dendrite in microns
    num_top_rois: int
        number of top intensity ROIs to keep
    nrshiftmax: int
        maximum number of pixels to shift ROIs to avoid border ROIs
        TODO: Get this information from nonrigid registration parameters
        (e.g., from suite2p ops, maxShiftNR)

    Returns
    -------
    top_mask: 2d array
        binary mask of top intensity ROIs (non-overlapping, because of CellPose)
    """
    mimg = imageio.v3.imread(from_lims.get_average_intensity_projection_filepath(oeid))
    model = cp_models.Cellpose(gpu=False, model_type='cyto')
    masks, _, _, _ = model.eval(mimg, diameter=None, channels=[0,0])
    pix_size = from_lims.get_pixel_size_um(oeid)
    dendrite_diameter_px = dendrite_diameter_um / pix_size
    dendrite_filtered_mask = filter_dendrite(masks, dendrite_diameter_pix=dendrite_diameter_px)
    xrange, yrange = ppr.get_motion_correction_crop_xy_range_from_both_planes(oeid)
    border_filtered_mask = filter_border_roi(dendrite_filtered_mask, xrange=xrange, yrange=yrange, buffer_pix=nrshiftmax)
    top_mask = get_top_intensity_mask(mimg, border_filtered_mask, num_top_rois=num_top_rois)
    return top_mask


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


def filter_border_roi(masks, xrange, yrange, buffer_pix=5):
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
    border_mask[:yrange[0]+buffer_pix, :] = 1
    border_mask[yrange[1]-buffer_pix:, :] = 1
    border_mask[:, :xrange[0]+buffer_pix] = 1
    border_mask[:, xrange[1]-buffer_pix:] = 1

    filtered_mask = masks.copy()    
    num_roi = np.max(masks)
    for roi_id in range(1, num_roi + 1):
        roi_mask = masks == roi_id
        if np.any(roi_mask & border_mask):
            filtered_mask[roi_mask] = 0

    return filtered_mask


def get_top_intensity_mask(mimg, mask, num_top_rois=15):
    """ Get top intensity ROIs from mask

    Parameters
    ----------------
    mimg: 2d array
        mean intensity image
    mask: 2d array
        binary mask of ROIs
    num_top_rois: int
        number of top intensity ROIs to keep

    Returns
    -------
    top_mask: 2d array
        binary mask of top intensity ROIs (non-overlapping, because of CellPose)
    """

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


def draw_masks_on_image(img, masks):
    """ Draw masks on image

    Parameters
    ----------------
    img: 2d array
        image
    masks: 2d array
        binary mask of ROIs

    Returns
    -------
    fig: matplotlib figure
        figure with masks drawn on image
    """

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray')

    num_roi = np.max(masks)
    for i in range(1, num_roi + 1):
        ax.contour(masks == i, colors='r', linewidths=1)
    ax.axis('off')
    return fig