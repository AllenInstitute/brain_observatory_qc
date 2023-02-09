import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import percentile_filter
import os

from visual_behavior.data_access import from_lims, from_lims_utilities
# from mindscope_qc.data_access import from_lims, from_lims_utilities
# TODO: remove dependency from vba by using mindscope_qc.data_access
# Need to implement some functions to do so

# from ophys_etl.utils.traces import noise_std
# Copied noise_std (and robust_std)
# from ophys_etl.utils.traces to remove the dependency


def robust_std(x: np.ndarray) -> float:
    """Compute the median absolute deviation assuming normally
    distributed data. This is a robust statistic.

    Parameters
    ----------
    x: np.ndarray
        A numeric, 1d numpy array
    Returns
    -------
    float:
        A robust estimation of standard deviation.
    Notes
    -----
    If `x` is an empty array or contains any NaNs, will return NaN.
    """
    mad = np.median(np.abs(x - np.median(x)))

    return 1.4826 * mad


def noise_std(x: np.ndarray, filter_length: int) -> float:
    """Compute a robust estimation of the standard deviation of the
    noise in a signal `x`. The noise is left after subtracting
    a rolling median filter value from the signal. Outliers are removed
    in 2 stages to make the estimation robust.

    Parameters
    ----------
    x: np.ndarray
        1d array of signal (perhaps with noise)
    filter_length: int
        Length of the median filter to compute a rolling baseline,
        which is subtracted from the signal `x`.
        Must be an odd number? TODO: check if this is necessary.
        Must be calculated based on the frame rate of each experiment.

    Returns
    -------
    float:
        A robust estimation of the standard deviation of the noise.
        If any valurs of `x` are NaN, returns NaN.
    """
    if any(np.isnan(x)):
        return np.NaN
    # same as median filter,
    noise = x - percentile_filter(x, 50, filter_length)
    # with mode='reflect'

    # first pass removing positive outlier peaks
    # TODO: Confirm with scientific team that this is really what they want
    # (method is fragile if possibly have 0 as min)
    filtered_noise_0 = noise[noise < (1.5 * np.abs(noise.min()))]
    rstd = robust_std(filtered_noise_0)
    # second pass removing remaining pos and neg peak outliers
    filtered_noise_1 = filtered_noise_0[abs(filtered_noise_0) < (2.5 * rstd)]
    return robust_std(filtered_noise_1)


def nanmedian_filter(x, filter_length):
    """ 1D median filtering with np.nanmedian
    Parameters
    ----------
    x: 1D trace to be filtered
    filter_length: length of the filter (in frames)
        Must be calculated based on the frame rate of each experiment.

    Return
    ------
    filtered_trace
    """
    half_length = int(filter_length / 2)
    # Create 'reflect' traces at the extrema
    temp_trace = np.concatenate(
        (np.flip(x[:half_length]), x, np.flip(x[-half_length:])))
    filtered_trace = np.zeros_like(x)
    for i in range(len(x)):
        filtered_trace[i] = np.nanmedian(temp_trace[i:i + filter_length])
    return filtered_trace


def calculate_dff(corrected_trace, baseline, noise_sd):
    """ Calculate dff

    Parameters
    ----------
    corrected_trace : 1d array
        neuropil-corrected trace
    baseline : 1d array
        baseline of the corrected_trace, pre-computed
    noise_sd : float
        noise S.D. from the corrected_trace, pre-computed

    Returns
    -------
    1d array
        dff
    """
    dff = ((corrected_trace - baseline) / np.maximum(baseline, noise_sd))
    return dff


def get_correct_frame_rate(ophys_experiment_id):
    """ Getting the correct frame rate, rather than fixed 11.0 from the metadata

    Args:
        ophys_experiment_id (int): ophys experiment ID

    Returns:
        float: frame rate
        pd.DataFrame: timestamps
    """
    # TODO: change from_lims_utilities from vba to that in mindscope_qc.
    # mindscope_qc currently does not seem to have tools for getting timestamps.
    lims_data = from_lims_utilities.utils.get_lims_data(ophys_experiment_id)
    timestamps = from_lims_utilities.utils.get_timestamps(lims_data)
    frame_rate = 1 / np.mean(np.diff(timestamps.ophys_frames.timestamps))
    return frame_rate, timestamps


def get_new_dff_df(ophys_experiment_id, inactive_kernel_size=30, inactive_percentile=10):
    """ Get the new dff from an experiment, along with the old one
    TODO: Dealing with variable noise S.D. within a session
    TODO: Dealing with variable baseline change rate
        In rare cases, baseline changes faster than the inactive kernel size.
        Usually along with variable noise S.D., so it can be imaging issue (e.g., out-of-FOV)
        Dynamic inactive_kernel_size might help, if it is not from imaging issue.
    TODO: Flag a trace. There can be huge negative ticks (e.g., experiment 1109786074, roi id 1109811452)
        Possibly by traces having values lower than certain threshold based on the baseline.
        (e.g., baseline - 3 * noise_sd)
    Parameters
    ----------
    ophys_experiment_id : int
        ophys_experiment_id
    inactive_kernel_size : int, optional
        kernel size for low_baseline calculation, by default 30
    inactive_percentile : int, optional
        percentile to calculate low_baseline, by default 10

    Returns
    -------
    pd.DataFrame
        DataFrame containing ophys_timestamps, neuropil_corrected traces, and dff
            from the ophys_experiment_id
    np.ndarray
        Ophys timestamps
    """
    # Get the correct frame rate along with the timestamps
    frame_rate, timestamps = get_correct_frame_rate(ophys_experiment_id)

    # Define filter lengths (int was necessary in HPC)
    long_filter_length = int(round(frame_rate * 60 * inactive_kernel_size))
    short_filter_length = int(round(frame_rate * 60 * 10))  # 10 min

    # Get neuropil-corrected traces as DataFrame
    np_corrected_df = get_np_corrected_df(ophys_experiment_id)
    # Calculate dff using the new method (using "inactive frames" to calculate baseline;
    # "inactive frames" defined by those with signals less than 10th percentile + 3 x noise_sd)
    new_dff_all = []
    old_dff_all = []
    dff_h = h5py.File(from_lims.get_dff_traces_filepath(
        ophys_experiment_id), 'r')
    for _, row in np_corrected_df.iterrows():
        corrected_trace = row.np_corrected
        crid = row.cell_roi_id
        # Calculate noise_sd
        noise_sd = noise_std(corrected_trace, filter_length=int(
            round(frame_rate * 3.33)))  # 3.33 s is fixed
        # 10th percentile "low_baseline"
        low_baseline = percentile_filter(
            corrected_trace, size=long_filter_length, percentile=inactive_percentile, mode='reflect')
        # Create trace using inactive frames only, by replacing signals in "active frames" with NaN
        active_frame = np.where(corrected_trace > (
            low_baseline + 3 * noise_sd))[0]
        inactive_trace = corrected_trace.copy()
        for i in active_frame:
            inactive_trace[i] = np.nan
        # Calculate baseline using median filter
        baseline_new = nanmedian_filter(inactive_trace, short_filter_length)
        # Calculate DFF
        new_dff = calculate_dff(corrected_trace, baseline_new, noise_sd)
        # Load old dff
        roi_ind = np.where(
            [int(rn) == crid for rn in dff_h['roi_names']])[0][0]
        old_dff = dff_h['data'][roi_ind]
        # Gather data
        new_dff_all.append(new_dff)
        old_dff_all.append(old_dff)
    np_corrected_df['new_dff'] = new_dff_all
    np_corrected_df['old_dff'] = old_dff_all
    # Add timestamps
    ophys_timestamps = timestamps.ophys_frames.timestamps
    return np_corrected_df, ophys_timestamps


def get_fixed_r_values(ophys_experiment_id, crid_values, num_normal_r_thresh, r_replace):
    """Get fixed r values
    When r >= 1, replace that r value with the average of other r values < 1 ("normal r values").
    If the number of ROIs with "normal r values" are less than a threshold ('num_normal_r_thresh'),
    replace the r values with 'r_replace' (most cases 0.7, a convention in the field).

    Parameters
    ----------
    ophys_experiment_id : int
        ophys experiment ID
    crid_values : list
        of cell ROI IDs
    num_normal_r_thresh : int
        a threshold in number of rois with normal r values to average
    r_replace : float
        r value to replace, when ROIs with "normal r values" are less than a threshold

    Returns
    -------
    list
        list of r, after fixing the error
    list
        list of bool, indicating if the ROI was out of range
    """
    r_list = []
    r_out_of_range = []
    num_normal_r = 0
    np_correction_h = h5py.File(
        from_lims.get_neuropil_correction_filepath(ophys_experiment_id), 'r')
    for crid in crid_values:
        roi_ind = np.where(
            [int(rn) == crid for rn in np_correction_h['roi_names']])[0][0]
        r = np_correction_h['r'][roi_ind]
        r_list.append(r)
        if (r <= 0) or (r >= 1):
            r_out_of_range.append(True)
        else:
            r_out_of_range.append(False)
            num_normal_r += 1
    r_out_of_range = np.array(r_out_of_range).astype(int)
    if num_normal_r < len(crid_values):  # In case there was r out of range
        if num_normal_r > num_normal_r_thresh:
            r_replace = np.mean([r_list[i] for i in np.where(r_out_of_range == 0)[0]])
        for i in np.where(r_out_of_range)[0]:
            r_list[i] = r_replace
    return r_list, r_out_of_range


def get_np_corrected_df(ophys_experiment_id, num_normal_r_thresh=5, r_replace=0.7):
    """ Get neuropil-corrected traces DataFrame
    Fix r value problem (when r > 1, replace r with the mean of other r values < 1.
    If there are less than "num_normal_r_thresh" of those other r values, than replace with 0.7)

    Parameters
    ----------
    ophys_experiment_id : int
        ophys_experiment_id
    num_normal_r_thresh : int, default 5
        number of r values within (0,1) to be used to replace those out of range
    r_replace: float, default 0.7
        value to replace out of range r values in case number of normal r values <= num_normal_r_thresh

    Returns
    -------
    pd.DataFrame
        DataFrame containing neuropil-corrected traces for each cell
            Both cell specimen ID and cell ROI ID are provided
    """
    # Get valid cell ROI ID and cell specimen ID
    cell_rois_table = from_lims.get_cell_rois_table(ophys_experiment_id)
    cell_rois_table = cell_rois_table[cell_rois_table.valid_roi]
    csid_values = cell_rois_table.cell_specimen_id.values
    if csid_values[0] is None:
        csid_values = [0] * len(csid_values)
    crid_values = cell_rois_table.id.values

    # Collect r values and fix if r > 1
    r_list, r_out_of_range = get_fixed_r_values(
        ophys_experiment_id, crid_values, num_normal_r_thresh, r_replace)

    np_corrected_all = []
    demixed_h = h5py.File(
        from_lims.get_demixed_traces_filepath(ophys_experiment_id), 'r')
    neuropil_h = h5py.File(
        from_lims.get_neuropil_traces_filepath(ophys_experiment_id), 'r')
    for crid, r in zip(crid_values, r_list):
        # Get the demixed trace, neuropil trace, neuropil correction coefficient and error
        roi_ind = np.where(
            [int(rn) == crid for rn in demixed_h['roi_names']])[0][0]
        demixed = demixed_h['data'][roi_ind]

        roi_ind = np.where(
            [int(rn) == crid for rn in neuropil_h['roi_names']])[0][0]
        neuropil = neuropil_h['data'][roi_ind]

        # Calculate neuropil-corrected trace
        corrected = demixed - neuropil * r
        # Gather traces
        np_corrected_all.append(corrected)
    # Build the neuropil-corrected DataFrame
    np_corrected_df = pd.DataFrame({'cell_specimen_id': csid_values,
                                    'cell_roi_id': crid_values,
                                    'np_corrected': np_corrected_all,
                                    'r': r_list,
                                    'r_out_of_range': r_out_of_range})
    return np_corrected_df


def save_new_dff_h5(save_dir, new_dff_df, timestamps, oeid):
    new_dff_array = np.zeros((len(new_dff_df), len(timestamps)))
    old_dff_array = np.zeros((len(new_dff_df), len(timestamps)))
    np_corrected_array = np.zeros((len(new_dff_df), len(timestamps)))
    for i, row in new_dff_df.iterrows():
        new_dff_array[i, :] = row.new_dff
        old_dff_array[i, :] = row.old_dff
        np_corrected_array[i, :] = row.np_corrected
    save_fn = save_dir / f'{oeid}_new_dff.h5'
    with h5py.File(save_fn, 'w') as hf:
        hf.create_dataset('cell_specimen_id', data=new_dff_df.cell_specimen_id)
        hf.create_dataset('cell_roi_id', data=new_dff_df.cell_roi_id)
        hf.create_dataset('r', data=new_dff_df.r)
        hf.create_dataset('r_out_of_range', data=new_dff_df.r_out_of_range)
        hf.create_dataset('new_dff', data=new_dff_array)
        hf.create_dataset('old_dff', data=old_dff_array)
        hf.create_dataset('np_corrected', data=np_corrected_array)
        hf.create_dataset('timestamps', data=timestamps)

    return save_fn


def draw_fig_new_dff(save_dir, new_dff_df, timestamps, oeid):
    for _, row in new_dff_df.iterrows():
        crid = row.cell_roi_id
        if os.path.isdir(save_dir / f'fig_new_dff_{oeid}') is not True:
            os.mkdir(save_dir / f'fig_new_dff_{oeid}')
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        ax[0].plot(timestamps, row.np_corrected, 'g',
                   label='corrected', linewidth=0.5)
        ax[0].legend()
        ax[1].plot(timestamps, row.old_dff, 'k',
                   label='old dff', linewidth=0.5)
        ax[1].legend()
        ax[2].plot(timestamps, row.new_dff, 'b',
                   label='new dff', linewidth=0.5)
        ax[2].legend()
        ax[2].set_xlabel('Time (s)')
        fig.suptitle(f'Ophys experiment ID: {oeid}  Cell ROI ID: {crid}')
        fig.tight_layout()
        fig.savefig(
            save_dir / f'fig_new_dff_{oeid}' / f'{crid}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
