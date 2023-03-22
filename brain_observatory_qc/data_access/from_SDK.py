# import os
import pandas as pd
import numpy as np
from warnings import warn
# from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
# from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc


def dateformat(exp_date):
    """
    reformat date of acquisition for accurate sorting by date
    """
    warn('This is deprecated. Dateformat was moved to BOA', DeprecationWarning, stacklevel=2)
    from datetime import datetime
    date = int(datetime.strptime(exp_date, '%Y-%m-%d  %H:%M:%S.%f').strftime('%Y%m%d'))
    return date


def add_date_string(df):
    """
    Adds a new column called "date" that is a string version of the date_of_acquisition column,
    with the format year-month-date, such as 20210921
    """
    warn('This is deprecated. add_date_string was moved to BOA', DeprecationWarning, stacklevel=2)
    df['date'] = df['date_of_acquisition'].apply(dateformat)
    return df


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


def get_n_relative_to_first_novel(group):
    """
    Function to apply to experiments_table data grouped on 'ophys_container_id'
    For each container, determines the numeric order of sessions relative to the first novel image session
    returns a pandas Series with column 'n_relative_to_first_novel' indicating this value for all session in the container
    If the container does not have a truly novel session, all values are set to NaN
    """
    warn('This is deprecated. get_n_relative_to_first_novel was moved to BOA', DeprecationWarning, stacklevel=2)
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
    warn('This is deprecated. add_n_relative_to_first_novel_column was moved to BOA', DeprecationWarning, stacklevel=2)

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
    warn('This is deprecated. add_last_familiar_column was moved to BOA', DeprecationWarning, stacklevel=2)

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
    warn('This is deprecated. Use get_last_familiar_column function for mFish data in BOA.', DeprecationWarning, stacklevel=2)

    group = group.sort_values(by='date')
    last_familiar_active = np.empty(len(group))
    last_familiar_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel < 0))[0]  # noqa : E712
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
    warn('This is deprecated. Use add_last_familiar_column function for mFish data in BOA.', DeprecationWarning, stacklevel=2)

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
    warn('This is deprecated. add_second_novel_column was moved to BOA', DeprecationWarning, stacklevel=2)

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
    warn('This is deprecated. Use get_second_novel_column for mFish data in BOA', DeprecationWarning, stacklevel=2)

    group = group.sort_values(by='date')
    second_novel_active = np.empty(len(group))
    second_novel_active[:] = False
    indices = np.where((group.passive == False) & (group.n_relative_to_first_novel > 0))[0]  # noqa : E712
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
    warn('This is deprecated. Use add_second_novel_column for mFish data in BOA', DeprecationWarning, stacklevel=2)

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
    warn('This is deprecated. build_tidy_cell_df funtion has been moved to BOA', DeprecationWarning, stacklevel=2)

    return pd.concat([pd.DataFrame(get_cell_timeseries_dict(session, cell_specimen_id)) for cell_specimen_id in session.dff_traces.reset_index()['cell_specimen_id']]).reset_index(drop=True)


def limit_to_last_familiar_second_novel_active(df):
    """
    Drops rows that are not the last familiar active session or the second novel active session
    """
    warn('This is deprecated. Use limit_to_last_familiar_second_novel for mFish data in BOA', DeprecationWarning, stacklevel=2)

    # drop novel sessions that arent the second active one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel_active == False)].index.values  # noqa : E712
    df = df.drop(labels=indices, axis=0)

    # drop Familiar sessions that arent the last active one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar_active == False)].index.values  # noqa : E712
    df = df.drop(labels=indices, axis=0)

    return df


def limit_to_last_familiar_second_novel(df):
    """
    Drops rows that are not the last familiar session or the second novel session, regardless of active or passive
    """
    warn('This is deprecated. limit_to_last_familiar_second_novel funtion has been moved to BOA', DeprecationWarning, stacklevel=2)

    # drop novel sessions that arent the second active one
    indices = df[(df.experience_level == 'Novel >1') & (df.second_novel == False)].index.values  # noqa : E712
    df = df.drop(labels=indices, axis=0)

    # drop Familiar sessions that arent the last active one
    indices = df[(df.experience_level == 'Familiar') & (df.last_familiar == False)].index.values  # noqa : E712
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
    warn('This is deprecated. get_cell_specimen_ids_with_all_experience_levels funtion has been moved to BOA', DeprecationWarning, stacklevel=2)

    experience_level_counts = cells_table.groupby(['cell_specimen_id', 'experience_level']).count().reset_index().groupby(['cell_specimen_id']).count()[['experience_level']]
    cell_specimen_ids_with_all_experience_levels = experience_level_counts[experience_level_counts.experience_level == 3].index.unique()
    return cell_specimen_ids_with_all_experience_levels


def limit_to_cell_specimen_ids_matched_in_all_experience_levels(cells_table):
    """
    returns dataframe limited to cell_specimen_ids that are present in all 3 experience levels in ['Familiar', 'Novel 1', 'Novel >1']
    input dataframe is typically ophys_cells_table but can be any df with columns 'cell_specimen_id' and 'experience_level'
    """
    warn('This is deprecated. limit_to_cell_specimen_ids_matched_in_all_experience_levels funtion has been moved to BOA', DeprecationWarning, stacklevel=2)

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
