import h5py
import numpy as np
from brain_observatory_qc.data_access import from_lims
from brain_observatory_qc.pipeline_dev import paired_plane_registration as ppr
from suite2p.registration import nonrigid


def episodic_mean_fov(movie_fn, max_num_epochs=10, num_frames_to_avg=1000):
    """
    Calculate the mean FOV image for each epoch in a movie
    Parameters
    ----------
    movie_fn : str or Path
        Path to the movie file
        h5 file
    max_num_epochs : int
        Maximum number of epochs to calculate the mean FOV image for
    num_frames_to_avg : int
        Number of frames to average to calculate the mean FOV image
    Returns
    -------
    mean_fov : np.ndarray
        Array of shape (num_epochs, height, width) containing the mean FOV image for each epoch
    """
    # Load the movie
    if not str(movie_fn).endswith('.h5'):
        raise ValueError('movie_fn must be an h5 file')
    
    with h5py.File(movie_fn, 'r') as f:
        data_length = f['data'].shape[0]
        num_epochs = min(max_num_epochs, data_length // num_frames_to_avg)
        epoch_interval = data_length // (num_epochs+1)  # +1 to avoid the very first frame (about half of each epoch)
        num_frames = min(num_frames_to_avg, epoch_interval)
        start_frames = [num_frames//2 + i * epoch_interval for i in range(num_epochs)]
        assert start_frames[-1] + num_frames < data_length

        # Calculate the mean FOV image for each epoch
        mean_fov = np.zeros((num_epochs, f['data'].shape[1], f['data'].shape[2]))
        for i in range(num_epochs):
            start_frame = start_frames[i]
            mean_fov[i] = np.mean(f['data'][start_frame : start_frame + num_frames_to_avg], axis=0)

    return mean_fov


def episodic_mean_fov_reg_to_the_paired(oeid, max_num_epochs=10, num_frames_to_avg=1000):
    """
    Calculate the mean FOV image for each epoch in a movie, registered to the paired plane
    Parameters
    ----------
    oeid : int
        ophys experiment id
    max_num_epochs : int
        Maximum number of epochs to calculate the mean FOV image for
    num_frames_to_avg : int
        Number of frames to average to calculate the mean FOV image
    Returns
    -------
    mean_fov : np.ndarray
        Array of shape (num_epochs, height, width) containing the mean FOV image for each epoch
    """
    # Load the paired plane registration info
    try:
        paired_plane_id = from_lims.get_paired_plane_id(oeid)
    except:
        raise ValueError('No paired plane found for oeid {}'.format(oeid))
    paired_shifts_df = ppr.get_s2p_motion_transform(paired_plane_id)
    if_nonrigid = True if 'nonrigid' in paired_shifts_df.columns else False
    oeid_path = from_lims.get_motion_xy_offset_filepath(oeid).parent.parent
    raw_movie_h5 = oeid_path / (str(oeid) + '.h5')
    if not raw_movie_h5.exists():
        raise ValueError('No raw movie found for oeid {}'.format(oeid))
    
    with h5py.File(raw_movie_h5, 'r') as f:
        data_length = f['data'].shape[0]
        assert data_length == paired_shifts_df.shape[0]

        num_epochs = min(max_num_epochs, data_length // num_frames_to_avg)
        epoch_interval = data_length // (num_epochs+1)  # +1 to avoid the very first frame (about half of each epoch)
        num_frames = min(num_frames_to_avg, epoch_interval)
        start_frames = [num_frames//2 + i * epoch_interval for i in range(num_epochs)]
        assert start_frames[-1] + num_frames < data_length

        # Calculate the mean FOV image for each epoch, after paired plane registration
        mean_fov = np.zeros((num_epochs, f['data'].shape[1], f['data'].shape[2]))
        for i in range(num_epochs):
            start_frame = start_frames[i]
            epoch_data = f['data'][start_frame : start_frame + num_frames]
            y = paired_shifts_df['y'].values[start_frame : start_frame + num_frames]
            x = paired_shifts_df['x'].values[start_frame : start_frame + num_frames]
            if if_nonrigid:
                nonrigid_y = paired_shifts_df['nonrigid_y'].values[start_frame : start_frame + num_frames]
                nonrigid_x = paired_shifts_df['nonrigid_x'].values[start_frame : start_frame + num_frames]
                # from default parameters:
                # TODO: read from a file
                Ly1 = 512
                Lx1 = 512
                block_size = (128, 128)
                blocks = nonrigid.make_blocks(Ly=Ly1, Lx=Lx1, block_size=block_size)
            epoch_registered = epoch_data.copy()
            for frame, dy, dx in zip(epoch_registered, y, x):
                frame[:] = ppr.shift_frame(frame=frame, dy=dy, dx=dx)
            if if_nonrigid:
                epoch_registered = nonrigid.transform_data(epoch_registered, yblock=blocks[0], xblock=blocks[1], nblocks=blocks[2],
                                                            ymax1=nonrigid_y, xmax1=nonrigid_x, bilinear=True)
    
            mean_fov[i] = np.mean(epoch_registered, axis=0)

    return mean_fov


def get_signal_paired_ep_meanfov(oeid, max_num_epochs=10, num_frames_to_avg=1000, motion_buffer=5, remove_motion_boundary=True):
    exp_movie_fn = from_lims.get_motion_corrected_movie_filepath(oeid)
    signal_ep_meanfov = episodic_mean_fov(exp_movie_fn, max_num_epochs=max_num_epochs, num_frames_to_avg=num_frames_to_avg)
    paired_ep_meanfov = episodic_mean_fov_reg_to_the_paired(oeid, max_num_epochs=max_num_epochs, num_frames_to_avg=num_frames_to_avg)
    assert signal_ep_meanfov.shape == paired_ep_meanfov.shape

    if remove_motion_boundary:
        p1y, p1x = ppr.get_motion_correction_crop_xy_range_from_both_planes(oeid)
        signal_ep_meanfov = signal_ep_meanfov[:, p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                                            p1x[0]+motion_buffer:p1x[1]-motion_buffer]
        paired_ep_meanfov = paired_ep_meanfov[:, p1y[0]+motion_buffer:p1y[1]-motion_buffer,
                                            p1x[0]+motion_buffer:p1x[1]-motion_buffer]
        
    return signal_ep_meanfov, paired_ep_meanfov
