from typing import Union
import mindscope_qc.data_access.from_lims as from_lims
from pathlib import Path
from scipy import stats
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import h5py
import dask.array as da


def get_paired_planes_list(session_path: Union[str, Path]) -> list:
    """Get list of paired planes experiment IDs within a session, for
    example 4x2 configuration on mesoscope has 8 experiments with 
    4 pairs of 2.

    Parameters
    ----------
    session_path : str or pathlib.Path
        Path to session directory

    Returns
    -------
    list
        List of lists of paired plane experiment IDs

    """
    split_str = 'MESOSCOPE_FILE_SPLITTING_QUEUE*input.json'

    session_path = Path(session_path)

    split_file = list(session_path.glob(f'{split_str}'))
    assert len(split_file) == 1, f'Found {len(split_file)} mesoscope split files, expected 1'
    split_file = split_file[0]

    with open(split_file, 'r') as f:
        split_json = json.load(f)

    all_paired = []
    for pg in split_json['plane_groups']:

        paired = []
        for expt in pg["ophys_experiments"]:
            paired.append(expt['experiment_id'])

        all_paired.append(paired)

    return all_paired


def session_path_from_eid(eid: int) -> Path:
    """Get session path from ophsy_experiment_id

    TODO: likley to be replaced by from_lims

    Parameters
    ----------
    eid : int
        ophys_experiment_id

    Returns
    -------
    pathlib.Path
        Path to session directory

    """
    info = from_lims.get_general_info_for_ophys_experiment_id(eid)
    session_path = info.session_storage_directory.loc[0]

    return Path(session_path)


def get_paired_plane_id(ophys_experiment_id: int) -> int:
    """Get experiment ID of paired plane

    Parameters
    ----------
    ophys_experiment_id : int
        ophys_experiment_id

    Returns
    -------
    int
        ophys_experiment_id of paired plane

    """
    # assert int
    assert isinstance(ophys_experiment_id, int), 'ophys_experiment_id must be int'

    session_path = session_path_from_eid(ophys_experiment_id)

    all_paired = get_paired_planes_list(session_path)

    # find eid pair in all_paired
    for pair in all_paired:
        if ophys_experiment_id in pair:
            pair.remove(ophys_experiment_id)
            other_id = pair[0]
            break

    return other_id


def get_s2p_rigid_motion_transform(eid):
    info = from_lims.get_general_info_for_ophys_experiment_id(eid)
    expt_path = info.experiment_storage_directory.loc[0]
    proc_path = expt_path / 'processed'

    mc_str = '*suite2p_rigid_motion_transform.csv'

    # get splt_str file in session path

    mc_file = list(proc_path.glob(f'{mc_str}'))
    assert len(mc_file) == 1
    mc_file = mc_file[0]

    return pd.read_csv(mc_file)


def get_paired_slope(session_path):
    paired_list = get_paired_planes_list(session_path)

    dfs = []
    for pair in paired_list:
        p1 = get_s2p_rigid_motion_transform(pair[0])
        p2 = get_s2p_rigid_motion_transform(pair[1])
        slope_x, intercept_x, r_value_x, p_valu_x, std_err = stats.linregress(
            p1.x, p2.x)
        slope_y, intercept_y, r_value_x, p_value_x, std_err = stats.linregress(
            p1.y, p2.y)

        dfs.append(pd.DataFrame({'slope_x': slope_x, 'slope_y': slope_y,
                                 'intercept_x': intercept_x, 'intercept_y': intercept_y,
                                 'ophys_experiment_id': pair}))

    df_linear = pd.concat(dfs)

    p1 = df_linear.loc[0].reset_index(drop=True)
    p2 = df_linear.loc[1].reset_index(drop=True)

    p1["paired_id"] = p2["ophys_experiment_id"]

    return p1


def paired_shifts_regression_eid(expts_ids):
    # get all paired slope for all experiments
    all_pairs = []
    for eid in expts_ids:
        print(eid)
        info = from_lims.get_general_info_for_ophys_experiment_id(eid)
        session_path = info.session_storage_directory.loc[0]
        all_pairs.append(get_paired_slope(session_path))

    all_pairs = pd.concat(all_pairs)

    return all_pairs.reset_index(drop=True)


# ALL SESSIONS
def paired_shifts_regression(sids):
    # get all paired slope for all experiments
    all_pairs = []
    for sid in sids:
        try:
            print(sid)
            info = from_lims.get_general_info_for_ophys_session_id(sid)
            session_path = info.session_storage_directory.loc[0]
            all_pairs.append(get_paired_slope(session_path))
        except Exception:
            print(f'failed for {sid}')

    all_pairs = pd.concat(all_pairs)

    return all_pairs.reset_index(drop=True)


def paired_planes_shifted_projections(eid, block_size: int = 10000):
    eid1 = eid
    expt1_path = from_lims.get_general_info_for_ophys_experiment_id(
        eid).experiment_storage_directory.iloc[0]
    raw1_h5 = expt1_path / (str(eid1) + '.h5')
    frames1 = load_h5_dask(raw1_h5)

    expt1_shifts = get_s2p_rigid_motion_transform(eid1)

    eid2 = get_paired_plane_id(eid1)
    expt2_path = from_lims.get_general_info_for_ophys_experiment_id(
        eid2).experiment_storage_directory.iloc[0]
    raw2_h5 = expt2_path / (str(eid2) + '.h5')
    frames2 = load_h5_dask(raw2_h5)
    expt2_shifts = get_s2p_rigid_motion_transform(eid2)

    block1 = frames1[:block_size].compute()
    expt1_img_raw = block1.mean(axis=0)

    e1y, e1x = expt1_shifts.y, expt1_shifts.x
    e2y, e2x = expt2_shifts.y, expt2_shifts.x

    block1_shift = block1.copy()
    for frame, dy, dx in zip(block1_shift, e1y, e1x):
        frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
    expt1_img_og_shifted = block1_shift.mean(axis=0)

    block1_pshift = block1.copy()
    for frame, dy, dx in zip(block1_pshift, e2y, e2x):
        frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
    expt1_img_p2_shifted = block1_pshift.mean(axis=0)

    # plane 2
    block2 = frames2[:block_size].compute()
    expt2_img_raw = block2.mean(axis=0)

    block2_shift = block2.copy()
    for frame, dy, dx in zip(block2_shift, e2y, e2x):
        frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
    expt2_img_og_shifted = block2_shift.mean(axis=0)

    block2_pshift = block2.copy()
    for frame, dy, dx in zip(block2_pshift, e1y, e1x):
        frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)
    expt2_img_p1_shifted = block2_pshift.mean(axis=0)

    # make dict of all images
    images = {'plane1_raw': expt1_img_raw,
              'plane1_original_shifted': expt1_img_og_shifted,
              'plane1_paired_shifted': expt1_img_p2_shifted,
              'plane2_raw': expt2_img_raw,
              'plane2_original_shifted': expt2_img_og_shifted,
              'plane2_paired_shifted': expt2_img_p1_shifted}

    return images


# import union

def shift_and_save_frames(frames,
                          y_shifts: Union[np.ndarray, pd.Series],
                          x_shifts: Union[np.ndarray, pd.Series],
                          block_size: int = None,
                          save_path: Path = None):
    """Shift frames and save to h5 file

    Parameters
    ----------
    frames : np.ndarray or dask.array
        frames to shift
    block_size : int, optional
        number of frames to shift, by default None
    y_shifts : np.ndarray or pd.Series
        y shifts to apply to frames
    x_shifts : np.ndarray or pd.Series
        x shifts to apply to frames

    Returns
    -------
    None"""

    # assert that frames and shifts are the same length
    assert len(frames) == len(y_shifts) == len(x_shifts)
    # make copy of frames
    if block_size is not None:
        sframes = frames[:block_size].compute()
    else:
        sframes = frames.compute()
    for frame, dy, dx in zip(sframes, y_shifts, x_shifts):
        frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

    # save frames
    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('data', data=sframes)

    print(sframes.shape)
    return sframes


def generate_all_pairings_shifted_frames(eid, block_size: int = None, save_path: Path = None):
    """Generate shifted frames for an experiment

    Parameters
    ----------
    eid : int
        experiment id
    block_size : int, optional
        number of frames to shift, if None, shift all frames
    save_path : Path, optional
        path to save shifted frames, by default None

    Returns
    -------
    np.ndarray
        shifted frames
    """
    expt_path = from_lims.get_general_info_for_ophys_experiment_id(
        eid).experiment_storage_directory.iloc[0]
    raw_h5 = expt_path / (str(eid) + '.h5')
    plane1_frames = load_h5_dask(raw_h5)
    plane1_shifts = get_s2p_rigid_motion_transform(eid)

    # get shifts for paired
    paired_id = get_paired_plane_id(eid)
    paired_shifts = get_s2p_rigid_motion_transform(paired_id)
    expt_path_paired = from_lims.get_general_info_for_ophys_experiment_id(
        paired_id).experiment_storage_directory.iloc[0]
    raw_h5_paired = expt_path_paired / (str(paired_id) + '.h5')
    plane2_frames = load_h5_dask(raw_h5_paired)

    # if save path, make all 4 filenames
    if save_path is not None:
        save_path.mkdir(exist_ok=True)

        p1_og_fn = save_path / f'{eid}_original_shift.h5'
        p2_paired_fn = save_path / f'{paired_id}_paired_shift.h5'
        p1_paired_fn = save_path / f'{eid}_paired_shift.h5'
        p2_og_fn = save_path / f'{paired_id}_original_shift.h5'

    p1_original_frames = shift_and_save_frames(frames=plane1_frames,
                                               y_shifts=plane1_shifts.y,
                                               x_shifts=plane1_shifts.x,
                                               block_size=block_size,
                                               save_path=p1_og_fn)

    p2_paired_frames = shift_and_save_frames(frames=plane2_frames,
                                             y_shifts=plane1_shifts.y,
                                             x_shifts=plane1_shifts.x,
                                             block_size=block_size,
                                             save_path=p2_paired_fn)

    p1_paired_frames = shift_and_save_frames(frames=plane1_frames,
                                             y_shifts=paired_shifts.y,
                                             x_shifts=paired_shifts.x,
                                             block_size=block_size,
                                             save_path=p1_paired_fn)

    p2_original_frames = shift_and_save_frames(frames=plane2_frames,
                                               y_shifts=paired_shifts.y,
                                               x_shifts=paired_shifts.x,
                                               block_size=block_size,
                                               save_path=p2_og_fn)

    # TODO: make motio correction cropping
    p1y, p1x = get_motion_correction_crop_xy_range(eid)
    p2y, p2x = get_motion_correction_crop_xy_range(paired_id)

    # crop frames
    p1_original_frames = p1_original_frames[:, p1y[0]:p1y[1], p1x[0]:p1x[1]]
    p2_paired_frames = p2_paired_frames[:, p2y[0]:p2y[1], p2x[0]:p2x[1]]
    p1_paired_frames = p1_paired_frames[:, p1y[0]:p1y[1], p1x[0]:p1x[1]]
    p2_original_frames = p2_original_frames[:, p2y[0]:p2y[1], p2x[0]:p2x[1]]

    # add all to dict
    shifted_frames = {'plane1_original': p1_original_frames,
                      'plane2_paired': p2_paired_frames,
                      'plane1_paired': p1_paired_frames,
                      'plane2_original': p2_original_frames}

    # # concat plane 1
    # plane1 = xr.concat([xr.DataArray(p1_original_frames, dims=['frame', 'y', 'x'], name='plane1_original'),
    #                     xr.DataArray(p1_paired_frames, dims=['frame', 'y', 'x'], name='plane1_paired')],
    #                     dim='plane')

    # # concat plane 2
    # plane2 = xr.concat([xr.DataArray(p2_original_frames, dims=['frame', 'y', 'x'], name='plane2_original'),
    #                     xr.DataArray(p2_paired_frames, dims=['frame', 'y', 'x'], name='plane2_paired')],
    #                     dim='plane')

    # # concat planes
    # concat_frames = xr.concat([plane1, plane2], dim='plane')

    return shifted_frames


def chunk_movie(movie, n_chunks):
    # TODO: move to utils
    # get length of movie divided into 12 chunks
    chunk_len = int(movie.shape[0] / n_chunks)

    # get mean of each chunk
    chunk_means = []
    for i in range(n_chunks):
        chunk = movie[i * chunk_len:(i + 1) * chunk_len, :, :]
        chunk_mean = chunk.mean(axis=0)
        chunk_means.append(chunk_mean)
    return chunk_means


def save_gif(image_stack, gif_folder, fn):
    # TODO: move to utils
    # save im_norm as gif
    import imageio
    import os

    # create folder for gif
    if not os.path.exists(gif_folder):
        os.makedirs(gif_folder)

    # save im_norm as animated gif
    images = []
    for img in image_stack:

        # set max of img to 99th percentile
        vmax = np.percentile(img, 99)
        img = img.clip(0, vmax)

        # convert to 8-bit
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)

        images.append(img)
    imageio.mimsave(gif_folder / fn, images, duration=0.1)


def get_motion_correction_crop_xy_range(oeid):
    """Get x-y ranges to crop motion-correction frame rolling

    TODO: move to utils

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


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Returns frame, shifted by dy and dx

    Parameters
    ----------
    frame: Ly x Lx
    dy: int
        vertical shift amount
    dx: int
        horizontal shift amount

    Returns
    -------
    frame_shifted: Ly x Lx
        The shifted frame

    # TODO: move to utils

    """
    assert frame.ndim == 2, "frame must be 2D"
    assert np.abs(dy) < frame.shape[0], "dy must be less than frame height"
    assert np.abs(dx) < frame.shape[1], "dx must be less than frame width"

    return np.roll(frame, (-dy, -dx), axis=(0, 1))

###############################################################################
# PLOTS
###############################################################################


def plot_paired_shifts(p1, p2, eid, paired_id, ax=None):

    # calculate difference between planes
    p1['x_diff'] = p1.x - p2.x
    p1['y_diff'] = p1.y - p2.y

    # plot diff
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(p1.x_diff, label='x')
    plt.plot(p1.y_diff, label='y')
    ax.set_title('difference between planes')

    # calculate x and y, corrected by first frame
    p1['x_bl'] = np.abs(p1.x - p1.x[0])
    p1['y_bl'] = np.abs(p1.y - p1.y[0])
    p2['x_bl'] = np.abs(p2.x - p2.x[0])
    p2['y_bl'] = np.abs(p2.y - p2.y[0])

    # p1['x_bl'] = np.abs(p1.x)
    # p1['y_bl'] = np.abs(p1.y)
    # p2['x_bl'] = np.abs(p2.x)
    # p2['y_bl'] = np.abs(p2.y)

    # plot x_bl scatter
    fig, ax = plt.subplots()
    plt.scatter(p1.x_bl, p2.x_bl, label='X offset', s=1, marker=',')
    plt.scatter(p1.y_bl, p2.y_bl, label='Y offset', s=1, marker=',')
    ax.set_title('x_bl')
    plot_equality_line(ax)
    plt.legend()

    # make plot square
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Plane 1 absolute shifts")
    ax.set_ylabel("Plane 2 absolute shifts")
    ax.set_title(
        f"Rigid motion correction shifts \n (absolute shifts from first frame) \n p1 = {eid}, p2 = {paired_id}")


def plot_paired_shifts_regression(p1, p2):
    # plot scatter
    fig, ax = plt.subplots()
    plt.scatter(p1.x, p2.x, label='X')
    plt.scatter(p1.y, p2.y, label='Y')
    ax.set_title('x')
    plt.legend()

    # calc liner regression
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        p1.x.values, p2.x.values)
    print(f"r-squared: {r_value**2}")

    # plot linear regression
    x = np.linspace(*ax.get_xlim())
    y = slope * x + intercept

    plt.plot(x, y, color='black', linestyle='--',
             label=f'slope: {slope:.2f}, intercept: {intercept:.2f}, r-squared: {r_value**2:.2f}')

    slope, intercept, r_value, p_value, std_err = stats.linregress(p1.y, p2.y)
    print(f"r-squared: {r_value**2}")

    # plot linear regression
    x = np.linspace(*ax.get_xlim())
    y = slope * x + intercept

    plt.plot(x, y, color='red', linestyle='--',
             label=f'slope: {slope:.2f}, intercept: {intercept:.2f}, r-squared: {r_value**2:.2f}')

    plt.legend()

    # plt.scatter(p1.y, p2.y, label='plane 2')


def fig_paired_planes_shifted_projections(projections_dict: dict):

    # get 99 percentile of all images to set vmax
    images = [v for k, v in projections_dict.items()]
    max_val = np.percentile(np.concatenate(images), 99.9)

    keys = ['plane1_raw', 'plane1_original_shifted', 'plane1_paired_shifted',
            'plane2_raw', 'plane2_original_shifted', 'plane2_paired_shifted']

    # check that all keys are in dict
    assert all([k in projections_dict.keys() for k in keys]), "missing keys in projections_dict"

    # subplots show all images
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].imshow(projections_dict["plane1_raw"], cmap='gray', vmax=max_val)
    ax[0, 0].set_title('plane 1 raw')
    ax[0, 1].imshow(projections_dict["plane1_original_shifted"], cmap='gray', vmax=max_val)
    ax[0, 1].set_title('plane 1 orignal shifts')
    ax[0, 2].imshow(projections_dict["plane1_paired_shifted"], cmap='gray', vmax=max_val)
    ax[0, 2].set_title('plane 1 shifted to plane 2')

    ax[1, 0].imshow(projections_dict["plane2_raw"], cmap='gray')
    ax[1, 0].set_title('plane 2 raw')
    ax[1, 1].imshow(projections_dict["plane2_original_shifted"], cmap='gray', vmax=max_val)
    ax[1, 1].set_title('plane 2 original shifts')
    ax[1, 2].imshow(projections_dict["plane2_paired_shifted"], cmap='gray', vmax=max_val)
    ax[1, 2].set_title('plane 2 shifted to plane 1')

    # turn off axis labels
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()


def histogram_shifts(expt1_shifts, expt2_shifts):
    e1y, e1x = expt1_shifts.y, expt1_shifts.x
    e2y, e2x = expt2_shifts.y, expt2_shifts.x

    fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)
    ax[0, 0].hist(e1y, bins=25)
    ax[0, 0].set_title('plane 1 y shifts')
    ax[0, 1].hist(e1x, bins=25)
    ax[0, 1].set_title('plane 1 x shifts')
    ax[1, 0].hist(e2y, bins=25)
    ax[1, 0].set_title('plane 2 y shifts')
    ax[1, 1].hist(e2x, bins=25)
    ax[1, 1].set_title('plane 2 x shifts')
    plt.tight_layout()
    plt.show()


def acutances_by_blocks(eid):
    from ophys_etl.modules.suite2p_registration.suite2p_utils import compute_acutance

    # make list of linear increasing block size
    block_size = [1000, 2000, 4000, 8000, 16000, 32000]

    dfs = []
    for bs in block_size:

        projs = paired_planes_shifted_projections(eid, block_size=bs)

        # compute acutnace for item in dic
        acutances = []
        for var, proj in projs.items():
            acu = compute_acutance(proj)
            acutances.append(acu)

        # make dataframe, columns = projs keys, rows = acutances
        acutances_df = pd.DataFrame(acutances, index=projs.keys())
        acutances_df.columns = ['acutance']
        acutances_df['block_size'] = bs
        acutances_df['eid'] = eid

        # swap rows and columns
        dfs.append(acutances_df)

    acutances_df = pd.concat(dfs).reset_index(names=['projection'])

    return acutances_df


####################################################################################################
# TODO: move to utils
####################################################################################################

def load_h5_dask(h5_path):

    # load h5 file
    h5 = h5py.File(h5_path, 'r')

    # check if h5 has 'data' key
    if 'data' not in h5.keys():
        raise KeyError(f'{h5_path} does not have "data" key')

    movie = da.from_array(h5["data"])

    # close h5? not sure array not computed yet...

    return movie


def plot_equality_line(ax):
    # TODO: move to utils
    # plot equlaity line, fit current plot
    x = np.linspace(*ax.get_xlim())
    y = x
    plt.plot(x, y, color='black', linestyle='--', label='equality line')