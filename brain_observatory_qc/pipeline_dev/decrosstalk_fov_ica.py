import h5py
import numpy as np
import brain_observatory_qc.data_access.from_lims as from_lims
from brain_observatory_qc.pipeline_dev import paired_plane_registration as ppr
from scipy import ndimage
import skimage


def decrosstalk_movie(oeid, paired_reg_fn, filter_sigma_um=25, num_epochs=15, num_frames=300):
    """Run FOV-based decrosstalk using constrained ICA on motion corrected movie data
    
    Parameters
    ----------
    oeid : int
        ophys experiment id
    paired_reg_fn : str, Path
        path to paired registration file
        TODO: Once paired plane registration pipeline is finalized,
        this parameter can be removed or replaced with paired_oeid
    filter_sigma_um : float
        sigma for gaussian filter in microns
    num_epochs : int
        number of epochs to run
    num_frames : int
        number of frames to use for each epoch
    
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
    num_frames = min(num_frames, epoch_interval)
    start_frames = [num_frames//2 + i * epoch_interval for i in range(num_epochs)]
    assert start_frames[-1] + num_frames < data_length

    pixel_size_um = from_lims.get_pixel_size_um(oeid)
    sigma = filter_sigma_um / pixel_size_um

    alpha_list = []
    beta_list = []
    for i in range(num_epochs):
        alpha, beta, _, _, _, _= decrosstalk_single_mean_image(oeid, paired_reg_fn, sigma,
                                                               start_frame=start_frames[i],
                                                               num_frames=num_frames,
                                                               get_recon=False)
        alpha_list.append(alpha)
        beta_list.append(beta)
    alpha = np.mean(alpha_list)
    beta = np.mean(beta_list)

    with h5py.File(signal_fn, 'r') as f:
        signal_data = f['data'][:]    
    with h5py.File(paired_reg_fn, 'r') as f:
        paired_data = f['data'][:]
    recon_signal_data = np.zeros_like(signal_data)
    for i in range(data_length):
        recon_signal_data[i, :, :] = apply_mixing_matrix(alpha, beta, signal_data[i, :, :], paired_data[i, :, :])[0]
    return recon_signal_data, alpha_list, beta_list


def decrosstalk_single_mean_image(oeid, paired_reg_fn, sigma, start_frame=500, num_frames=300,
                                  motion_buffer=5, get_recon=True):
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
    start_frame : int
        starting frame index
    num_frames : int
        number of frames to use
    motion_buffer : int
        number of pixels to crop from the edge of the motion corrected image (for nonrigid registration)
        TODO: Get this information from nonrigid registration parameters
        (e.g., from suite2p ops, maxShiftNR)
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
        signal_mean = f['data'][start_frame:start_frame+num_frames].mean(axis=0)
    with h5py.File(paired_reg_fn, 'r') as f:
        paired_mean = f['data'][start_frame:start_frame+num_frames].mean(axis=0)

    p1y, p1x = ppr.get_motion_correction_crop_xy_range_from_both_planes(oeid)
    signal_mean = signal_mean[p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                              p1x[0]+motion_buffer:p1x[1]-motion_buffer]
    paired_mean = paired_mean[p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                              p1x[0]+motion_buffer:p1x[1]-motion_buffer]

    signal_filtered = signal_mean - ndimage.gaussian_filter(signal_mean, sigma)
    paired_filtered = paired_mean - ndimage.gaussian_filter(paired_mean, sigma)

    # TODO: change grid searching to constrained gradient descent to speed up
    alpha, beta ,_, _ = unmixing_using_grid_search(signal_filtered, paired_filtered)
    if get_recon:
        recon_signal, recon_paired = apply_mixing_matrix(alpha, beta, signal_mean, paired_mean)
    else:
        recon_signal, recon_paired = None, None
    
    return alpha, beta, signal_mean, paired_mean, recon_signal, recon_paired


def unmixing_using_grid_search(signal_mean, paired_mean, yrange_for_mi=None, xrange_for_mi=None,
                               grid_interval=0.01, max_val=0.3):
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
    max_val : float
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
        calculate_mutual_info_grid_search(crop_signal_mean, crop_paired_mean, grid_interval=grid_interval, max_val=max_val)
    min_idx = np.argmin(mi_values)
    alpha = ab_pair[min_idx][0]
    beta = ab_pair[min_idx][1]
    recon_signal, recon_paired = apply_mixing_matrix(alpha, beta, signal_mean, paired_mean)
        
    return alpha, beta, recon_signal, recon_paired


def calculate_mutual_info_grid_search(signal_mean, paired_mean, grid_interval=0.01, max_val=0.3):
    """Calculate mutual information using grid search
    
    Parameters:
    -----------
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane
    grid_interval : float
        interval for grid search
    max_val : float
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

    alpha_list = np.arange(0, max_val + grid_interval, grid_interval)
    beta_list = np.arange(0, max_val + grid_interval, grid_interval)
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

