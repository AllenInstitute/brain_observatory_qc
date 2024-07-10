import os
import json
# import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle

# warning
gen_depr_str = 'this function is deprecated and will be removed in a future version, ' \
               + 'please use {}.{} instead'

# CONVENIENCE FUNCTIONS TO GET VARIOUS INFORMATION #


class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate


def get_ssim(img0, img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1, gaussian_weights=True)
    return ssim_pair


def get_lims_data(lims_id):
    ld = LimsDatabase(int(lims_id))  # noqa
    lims_data = ld.get_qc_param()
    lims_data.insert(loc=2, column='experiment_id', value=lims_data.lims_id.values[0])
    lims_data.insert(loc=2, column='session_type',
                     value='behavior_' + lims_data.experiment_name.values[0].split('_')[-1])
    lims_data.insert(loc=2, column='ophys_session_dir', value=lims_data.datafolder.values[0][:-28])
    return lims_data


def get_timestamps(lims_data):
    if '2P6' in lims_data.rig.values[0]:
        use_acq_trigger = True
    else:
        use_acq_trigger = False
    sync_data = get_sync_data(lims_data, use_acq_trigger)  # noqa
    timestamps = pd.DataFrame(sync_data)
    return timestamps


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


def replace_cell_specimen_ids(cell_roi_ids):
    # TODO: this will return cell_specimen_ids as None for cells tha are not in the pickle file.
    # Currenlty this function will only work if all cell specimen ids were None. If some of them were int, it will replace them with None.
    # replace cell specimen ids in cell specimen table if they are None (copper mouse)
    # filename = '//allen/programs/mindscope/workgroups/learning/analysis_plots/ophys/' + \
    #     'activity_correlation_lamf/nrsac/roi_match/copper_missing_osid_roi_table_nan_replaced.pkl'
    filename = '//allen/programs/mindscope/workgroups/learning/analysis_plots/ophys/' + \
        'activity_correlation_lamf/nrsac/roi_match/copper_all_roi_table.pkl'
    with open(filename, 'rb') as f:
        good_cids = pickle.load(f)
        good_cids = good_cids.set_index('cell_roi_id')
        f.close()

    cell_specimen_ids = []  # collect new cell specimen ids
    for roi in cell_roi_ids:
        try:
            cid = good_cids.loc[roi]['cell_specimen_id']
        except:  # noqa
            cid = None
        cell_specimen_ids.append(cid)

    # create a cell specimen table
    cell_specimen_table = pd.DataFrame({'cell_roi_id': cell_roi_ids, 'cell_specimen_id': cell_specimen_ids})
    cell_specimen_table = cell_specimen_table.set_index('cell_roi_id')

    return cell_specimen_table


# def get_sync_data(lims_data, use_acq_trigger):
#     logger.info('getting sync data')
#     sync_path = get_sync_path(lims_data)
#     sync_dataset = SyncDataset(sync_path)
#     # Handle mesoscope missing labels
#     try:
#         sync_dataset.get_rising_edges('2p_vsync')
#     except ValueError:
#         sync_dataset.line_labels = ['2p_vsync', '', 'stim_vsync', '', 'photodiode', 'acq_trigger', '', '',
#                                     'behavior_monitoring', 'eye_tracking', '', '', '', '', '', '', '', '', '', '', '',
#                                     '', '', '', '', '', '', '', '', '', '', 'lick_sensor']
#         sync_dataset.meta_data['line_labels'] = sync_dataset.line_labels

#     meta_data = sync_dataset.meta_data
#     sample_freq = meta_data['ni_daq']['counter_output_freq']
#     # 2P vsyncs
#     vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
#     vs2p_f = sync_dataset.get_falling_edges(
#         '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
#     vs2p_rsec = vs2p_r / sample_freq
#     vs2p_fsec = vs2p_f / sample_freq
#     if use_acq_trigger:  # if 2P6, filter out solenoid artifacts
#         vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
#         frames_2p = vs2p_r_filtered
#     else:  # dont need to filter out artifacts in pipeline data
#         frames_2p = vs2p_rsec
#     # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
#     # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
#     # vs2p_rsec = vs2p_r / sample_freq
#     # frames_2p = vs2p_rsec
#     # stimulus vsyncs
#     # vs_r = d.get_rising_edges('stim_vsync')
#     vs_f = sync_dataset.get_falling_edges('stim_vsync')
#     # convert to seconds
#     # vs_r_sec = vs_r / sample_freq
#     vs_f_sec = vs_f / sample_freq
#     # vsyncs = vs_f_sec
#     # add display lag
#     monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
#     vsyncs = vs_f_sec + monitor_delay  # this should be added, right!?
#     # line labels are different on 2P6 and production rigs - need options for both
#     if 'lick_times' in meta_data['line_labels']:
#         lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
#     elif 'lick_sensor' in meta_data['line_labels']:
#         lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
#     else:
#         lick_times = None
#     if '2p_trigger' in meta_data['line_labels']:
#         trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
#     elif 'acq_trigger' in meta_data['line_labels']:
#         trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
#     if 'stim_photodiode' in meta_data['line_labels']:
#         stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
#     elif 'photodiode' in meta_data['line_labels']:
#         stim_photodiode = sync_dataset.get_rising_edges('photodiode') / sample_freq
#     if 'cam2_exposure' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
#     elif 'cam2' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('cam2') / sample_freq
#     elif 'eye_tracking' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
#     if 'cam1_exposure' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
#     elif 'cam1' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('cam1') / sample_freq
#     elif 'behavior_monitoring' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq
#     # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger for 2P6 only
#     if use_acq_trigger:
#         frames_2p = frames_2p[frames_2p > trigger[0]]
#     # print(len(frames_2p))
#     if lims_data.rig.values[0][0] == 'M':  # if Mesoscope
#         roi_group = get_roi_group(lims_data)  # get roi_group order
#         frames_2p = frames_2p[roi_group::4]  # resample sync times
#     # print(len(frames_2p))
#     logger.info('stimulus frames detected in sync: {}'.format(len(vsyncs)))
#     logger.info('ophys frames detected in sync: {}'.format(len(frames_2p)))
#     # put sync data in format to be compatible with downstream analysis
#     times_2p = {'timestamps': frames_2p}
#     times_vsync = {'timestamps': vsyncs}
#     times_lick = {'timestamps': lick_times}
#     times_trigger = {'timestamps': trigger}
#     times_eye_tracking = {'timestamps': eye_tracking}
#     times_behavior_monitoring = {'timestamps': behavior_monitoring}
#     times_stim_photodiode = {'timestamps': stim_photodiode}
#     sync_data = {'ophys_frames': times_2p,
#                  'stimulus_frames': times_vsync,
#                  'lick_times': times_lick,
#                  'eye_tracking': times_eye_tracking,
#                  'behavior_monitoring': times_behavior_monitoring,
#                  'stim_photodiode': times_stim_photodiode,
#                  'ophys_trigger': times_trigger,
#                  }
#     return sync_data


def get_ophys_session_dir(lims_data):
    ophys_session_dir = lims_data.ophys_session_dir.values[0]
    return ophys_session_dir


def get_ophys_experiment_dir(lims_data):
    lims_id = get_lims_id(lims_data)
    ophys_session_dir = get_ophys_session_dir(lims_data)
    ophys_experiment_dir = os.path.join(ophys_session_dir, 'ophys_experiment_' + str(lims_id))
    return ophys_experiment_dir


def get_roi_group(lims_data):
    experiment_id = int(lims_data.experiment_id.values[0])
    ophys_session_dir = get_ophys_session_dir(lims_data)
    import json
    json_file = [file for file in os.listdir(ophys_session_dir) if ('SPLITTING' in file) and ('input.json' in file)]
    json_path = os.path.join(ophys_session_dir, json_file[0])
    with open(json_path, 'r') as w:
        jin = json.load(w)
    # figure out which roi_group the current experiment belongs to
    # plane_data = pd.DataFrame()
    for i, roi_group in enumerate(range(len(jin['plane_groups']))):
        group = jin['plane_groups'][roi_group]['ophys_experiments']
        for j, plane in enumerate(range(len(group))):
            expt_id = int(group[plane]['experiment_id'])
            if expt_id == experiment_id:
                expt_roi_group = i
    return expt_roi_group


def get_lims_id(lims_data):
    lims_id = lims_data.lims_id.values[0]
    return lims_id


def get_cell_timeseries_dict(session, cell_specimen_id):
    '''
    for a given cell_specimen ID, this function creates a dictionary with the following keys
    * timestamps: ophys timestamps
    * cell_roi_id
    * cell_specimen_id
    * dff
    * events
    * filtered events
    This is useful for generating a tidy dataframe

    arguments:
        session object
        cell_specimen_id

    returns
        dict

    '''
    cell_dict = {
        'timestamps': session.ophys_timestamps,
        'cell_roi_id': [session.dff_traces.loc[cell_specimen_id]['cell_roi_id']] * len(session.ophys_timestamps),
        'cell_specimen_id': [cell_specimen_id] * len(session.ophys_timestamps),
        'dff': session.dff_traces.loc[cell_specimen_id]['dff'],
        'events': session.events.loc[cell_specimen_id]['events'],
        'filtered_events': session.events.loc[cell_specimen_id]['filtered_events'],

    }

    return cell_dict


def build_tidy_cell_df(session):
    '''
    builds a tidy dataframe describing activity for every cell in session containing the following columns
    * timestamps: the ophys timestamps
    * cell_roi_id: the cell roi id
    * cell_specimen_id: the cell specimen id
    * dff: measured deltaF/F for every timestep
    * events: extracted events for every timestep
    * filtered events: filtered events for every timestep

    Takes a few seconds to build

    arguments:
        session

    returns:
        pandas dataframe
    '''
    return pd.concat([pd.DataFrame(get_cell_timeseries_dict(session, cell_specimen_id)) for cell_specimen_id in session.dff_traces.reset_index()['cell_specimen_id']]).reset_index(drop=True)


def correct_filepath(filepath):
    """using the pathlib python module, takes in a filepath from an
    arbitrary operating system and returns a filepath that should work
    for the users operating system

    Parameters
    ----------
    filepath : string
        given filepath

    Returns
    -------
    string
        filepath adjusted for users operating system
    """
    if filepath is None or filepath=="NA" or filepath=="":
        corrected_path = filepath
    else:
        filepath = filepath.replace('/allen', '//allen')
        corrected_path = Path(filepath)
    return corrected_path


def correct_dataframe_filepath(dataframe, column_string):
    """applies the correct_filepath function to a given dataframe
    column, replacing the filepath in that column in place


    Parameters
    ----------
    dataframe : table
        pandas dataframe with the column
    column_string : string
        the name of the column that contains the filepath to be
        replaced

    Returns
    -------
    dataframe
        returns the input dataframe with the filepath in the given
        column 'corrected' for the users operating system, in place
    """
    dataframe[column_string] = dataframe[column_string].apply(lambda x: correct_filepath(x))
    return dataframe


def dateformat(exp_date):
    """
    reformat date of acquisition for accurate sorting by date
    """

    date = int(datetime.strptime(exp_date, '%Y-%m-%d  %H:%M:%S.%f').strftime('%Y%m%d'))
    return date


def add_date_string(df):
    """
    Adds a new column called "date" that is a string version of the date_of_acquisition column,
    with the format year-month-date, such as 20210921
    """
    df['date'] = df['date_of_acquisition'].apply(dateformat)
    return df


def get_n_relative_to_first_novel(group):
    """
    Function to apply to experiments_table data grouped on 'ophys_container_id'
    For each container, determines the numeric order of sessions relative to the first novel image session
    returns a pandas Series with column 'n_relative_to_first_novel' indicating this value for all session in the container
    If the container does not have a truly novel session, all values are set to NaN
    """
    group = group.sort_values(by='date')  # must sort for relative ordering to be accurate
    if 'Novel 1' in group.experience_level.values:
        novel_ind = np.where(group.experience_level == 'Novel 1')[0][0]
        n_relative_to_first_novel = np.arange(-novel_ind, len(group) - novel_ind, 1)
    else:
        n_relative_to_first_novel = np.empty(len(group))
        n_relative_to_first_novel[:] = np.nan
    return pd.Series({'n_relative_to_first_novel': n_relative_to_first_novel})


def add_n_relative_to_first_novel_column(df):
    """
    Add a column called 'n_relative_to_first_novel' that indicates the session number relative to the first novel session for each experiment in a container.
    If a container does not have a first novel session, the value of n_relative_to_novel for all experiments in the container is NaN.
    Input df must have column 'experience_level' and 'date'
    Input df is typically ophys_experiment_table
    """
    # add simplified string date column for accurate sorting
    df = add_date_string(df)  # should already be in the table, but adding again here just in case
    df = df.sort_values(by=['ophys_container_id', 'date'])  # must sort for ordering to be accurate
    numbers = df.groupby('ophys_container_id').apply(get_n_relative_to_first_novel)
    df['n_relative_to_first_novel'] = np.nan
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'n_relative_to_first_novel'] = list(numbers.loc[container_id].n_relative_to_first_novel)
    return df


def add_last_familiar_column(df):
    """
    adds column to df called 'last_familiar' which indicates (with a Boolean) whether
    a session is the last familiar image session prior to the first novel session for each container
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['last_familiar'] = False
    indices = df[(df.n_relative_to_first_novel == -1) & (df.experience_level == 'Familiar')].index.values
    df.loc[indices, 'last_familiar'] = True
    return df


def get_last_familiar_active(group):
    """
    Function to apply to experiments_table data grouped by 'ophys_container_id'
    determines whether each session in the container was the last active familiar image session prior to the first novel session
    input df must have column 'n_relative_to_first_novel' and 'date'
    """
    group = group.sort_values(by='date')
    last_familiar_active = np.empty(len(group))
    last_familiar_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel < 0))[0]  # noqa: E712
    if len(indices) > 0:
        index = indices[-1]  # use last (most recent) index
        last_familiar_active[index] = True
    return pd.Series({'last_familiar_active': last_familiar_active})


def add_last_familiar_active_column(df):
    """
    Adds a column 'last_familiar_active' that indicates (with a Boolean) whether
    a session is the last active familiar image session prior to the first novel session in each container
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel' and 'date'
    """
    df = df.sort_values(by=['ophys_container_id', 'date'])
    values = df.groupby('ophys_container_id').apply(get_last_familiar_active)
    df['last_familiar_active'] = False
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'last_familiar_active'] = list(values.loc[container_id].last_familiar_active)
    # change to boolean
    df.loc[df[df.last_familiar_active == 0].index.values, 'last_familiar_active'] = False
    df.loc[df[df.last_familiar_active == 1].index.values, 'last_familiar_active'] = True
    return df


def add_second_novel_column(df):
    """
    Adds a column called 'second_novel' that indicates (with a Boolean) whether a session
    was the second passing novel image session after the first truly novel session, including passive sessions.
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel'
    """
    df['second_novel'] = False
    indices = df[(df.n_relative_to_first_novel == 1) & (df.experience_level == 'Novel >1')].index.values
    df.loc[indices, 'second_novel'] = True
    return df


def get_second_novel_active(group):
    """
    Function to apply to experiments_table data grouped by 'ophys_container_id'
    determines whether each session in the container was the second passing novel image session
    after the first novel session, and was an active behavior session
    input df must have column 'n_relative_to_first_novel' and 'date'
    """
    group = group.sort_values(by='date')
    second_novel_active = np.empty(len(group))
    second_novel_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel > 0))[0]  # noqa: E712
    if len(indices) > 0:
        index = indices[0]  # use first (most recent) index
        second_novel_active[index] = True
    return pd.Series({'second_novel_active': second_novel_active})


def add_second_novel_active_column(df):
    """
    Adds a column called 'second_novel_active' that indicates (with a Boolean) whether a session
    was the second passing novel image session after the first truly novel session, and was an active behavior session.
    If a container has no truly first novel session, all sessions are labeled as NaN
    input df must have 'experience_level' and 'n_relative_to_first_novel' and 'date'
    """
    df = df.sort_values(by=['ophys_container_id', 'date'])
    values = df.groupby('ophys_container_id').apply(get_second_novel_active)
    df['second_novel_active'] = False
    for container_id in df.ophys_container_id.unique():
        indices = df[df.ophys_container_id == container_id].index.values
        df.loc[indices, 'second_novel_active'] = list(values.loc[container_id].second_novel_active)
    # change to boolean
    df.loc[df[df.second_novel_active == 0].index.values, 'second_novel_active'] = False
    df.loc[df[df.second_novel_active == 1].index.values, 'second_novel_active'] = True
    return df


def limit_to_last_familiar_second_novel_active(df):
    """
    Drops rows that are not the last familiar active session or the second novel active session
    """
    # drop novel sessions that arent the second active one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel_active == False)].index.values  # noqa: E712
    df = df.drop(labels=indices, axis=0)

    # drop Familiar sessions that arent the last active one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar_active == False)].index.values  # noqa: E712
    df = df.drop(labels=indices, axis=0)

    return df


def limit_to_last_familiar_second_novel(df):
    """
    Drops rows that are not the last familiar session or the second novel session, regardless of active or passive
    """
    # drop novel sessions that arent the second active one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel == False)].index.values  # noqa: E712
    df = df.drop(labels=indices, axis=0)

    # drop Familiar sessions that arent the last active one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar == False)].index.values  # noqa: E712
    df = df.drop(labels=indices, axis=0)

    return df


def limit_to_second_novel_exposure(df):
    """
    Drops rows where Novel >1 sessions are not the second exposure to the novel image set
    input df must have columns 'experience_level' and 'prior_exposures_to_image_set'
    """
    # drop novel >1 sessions that arent the second exposure (prior exposures = 1)
    indices = df[(df.experience_level == 'Novel >1') & (df.prior_exposures_to_image_set != 1)].index.values
    df = df.drop(labels=indices, axis=0)
    return df


def get_containers_with_all_experience_levels(experiments_table):
    """
    identifies containers with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    returns a list of container_ids
    """
    experience_level_counts = experiments_table.groupby(['ophys_container_id', 'experience_level']).count().reset_index().groupby(['ophys_container_id']).count()[['experience_level']]
    containers_with_all_experience_levels = experience_level_counts[experience_level_counts.experience_level == 3].index.unique()
    return containers_with_all_experience_levels


def limit_to_containers_with_all_experience_levels(experiments_table):
    """
    returns experiment_table limited to containers with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    input dataframe is typically ophys_experiment_table but can be any df with columns 'ophys_container_id' and 'experience_level'
    """
    containers_with_all_experience_levels = get_containers_with_all_experience_levels(experiments_table)
    experiments_table = experiments_table[experiments_table.ophys_container_id.isin(containers_with_all_experience_levels)]
    return experiments_table


def get_cell_specimen_ids_with_all_experience_levels(cells_table):
    """
    identifies cell_specimen_ids with all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1'] in the input dataframe
    input dataframe must have column 'cell_specimen_id', such as in ophys_cells_table
    """
    experience_level_counts = cells_table.groupby(['cell_specimen_id', 'experience_level']).count().reset_index().groupby(['cell_specimen_id']).count()[['experience_level']]
    cell_specimen_ids_with_all_experience_levels = experience_level_counts[experience_level_counts.experience_level == 3].index.unique()
    return cell_specimen_ids_with_all_experience_levels


def limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table):
    """
    returns dataframe limited to cell_specimen_ids that are present in all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    input dataframe is typically ophys_cells_table but can be any df with columns 'cell_specimen_id' and 'experience_level'
    """
    cell_specimen_ids_with_all_experience_levels = get_cell_specimen_ids_with_all_experience_levels(cells_table)
    matched_cells_table = cells_table[cells_table.cell_specimen_id.isin(cell_specimen_ids_with_all_experience_levels)].copy()
    return matched_cells_table


def value_counts(df, conditions=['cell_type', 'experience_level', 'mouse_id']):
    """
    group by the first conditions and count the last one
    """
    counts = df.groupby(conditions).count().reset_index().groupby(conditions[:-1]).count()
    counts = counts[[conditions[-1]]].rename(columns={conditions[-1]: 'n_' + conditions[-1]})
    return counts
