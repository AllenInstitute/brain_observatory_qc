import os
import time
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta,  timezone


#####################################################################
#
#           CONSTANTS
#
#####################################################################


PROJECT_GROUPS_DICT = {
    "learning_mfish": ['LearningmFISHDevelopment','LearningmFISHTask1A'],               # noqa: E241
    "omFISH_V1_U01":  ['omFISHRbp4Meso','omFISHSstMeso', 'U01BFCT', 'omFISHGad2Meso'],  # noqa: E241, E501
    "open_scope":     ['OpenScopeDendriteCoupling','OpenScopeSequenceLearning'],        # noqa: E241, E501
    "hardware_dev":   ['MesoscopeDevelopment']                                          # noqa: E241, E501
}

QC_SUBMIT_STATUS_DICT = {
    "complete":   ["pass", "flag", "fail"],
    "incomplete": ["incomplete", "ready"],
    "error":      ["error"],
    "missing":    ["missing"]
}

QC_GENERATION_STATUS_DICT = {
    "complete":   ["ready"],
    "error":      ["error"]
}

MANUAL_OVERRIDE_OUTCOMES_DICT = {
    "pass": ["manual_override_pass"],
    "flag": ["manual_override_flag"],
    "fail": ["manual_override_fail"]
}

QC_OUTCOME_STATUS_DICT = {
    "pass" : ["pass"],
    "flag" : ["flag"],
    "fail" : ["fail"],
    "error": ["error"],
    "missing"   : ["missing"],
    "incomplete": ["incomplete", "ready"],
}

MANUAL_OVERRIDE_LIST = ['manual_override_pass',
                        'manual_override_flag',
                        'manual_override_fail']


############################
#    IMPACTED DATA
############################

OPHYS_SESSION_IMPACTED_DATA = ["physio_sync",
                               "eye_tracking",
                               "face_tracking",
                               "body_tracking",
                               "running_wheel",
                               "water_delivery",
                               "lick_detection",
                               "visual_stimulus",
                               "metadata",
                               "columnar_zstack",
                               "mouse_stress",
                               "mouse_health",
                               "brain_health",
                               "stimulus_sequence_progression",
                               "task_performance",
                               "mouse_behavior"]

OPHYS_SESSION_DATA_STREAMS = ["physio_sync",
                               "eye_tracking",
                               "face_tracking",
                               "body_tracking",
                               "running_wheel",
                               "water_delivery",
                               "lick_detection",
                               "visual_stimulus",
                               "metadata",
                               "columnar_zstack"]

OPHYS_SESSION_DATA_CONTEXT = ["mouse_stress",
                               "mouse_health",
                               "brain_health",
                               "stimulus_sequence_progression",
                               "task_performance",
                               "mouse_behavior"]

OPHYS_EXPERIMENT_IMPACTED_DATA = ["fov_centered_zstack",
                                  "physiology_recording",
                                  "metadata",
                                  "brain_health",
                                  "FOV_matching"]

OPHYS_EXPERIMENT_DATA_STREAMS = ["fov_centered_zstack",
                                 "physiology_recording",
                                  "metadata"]

OPHYS_EXPERIMENT_DATA_CONTEXT = ["brain_health",
                                 "FOV_matching"]
#####################################################################
#
#           CONNECT TO MONGODB
#
#####################################################################

############################
#    FUNCTS FOR JUPYTER NOTEBOOKS
############################

def connect_to_mouseqc_production(username = 'public',
                                  password = 'public_password',
                                  mongo_host = 'qc-sys-db',
                                  mongo_port = '27017'):
    """ READ ONLY connection to the production instance of mouseqc


    Parameters
    ----------
    username : str, optional
        _description_, by default 'public'
    password : str, optional
        _description_, by default 'public_password'
    mongo_host : str, optional
        _description_, by default 'qc-sys-db'
    mongo_port : str, optional
        _description_, by default '27017'

    Returns
    -------
    connection
        connection to the database
    """
    
    connection_string = 'mongodb://{}:{}@{}:{}/'.format(username,
                                                        password,
                                                        mongo_host,
                                                        mongo_port)
    mongo_connection = MongoClient(connection_string)
    return mongo_connection


def get_records_collections(mongo_connection):
    """connect to the relevant qc records collections

    Parameters
    ----------
    mongo_connection : database connection
        read only connection to the production
        qc database

    Returns
    -------
    connections
        connections to the following collections:
         - qc_logs
         - metrics_records
         - images_records
         - report_generation_status
         - module_group_status
         - report_review_status
    """
    records_db = mongo_connection['records']
    qc_metadata_db = mongo_connection['qc_metadata']

    qc_logs = records_db['qc_logs']
    metrics_records = records_db['metrics']
    images_records = records_db['images']

    report_generation_status = qc_metadata_db['qc_generation_status']
    module_group_status = qc_metadata_db['qc_module_group_status']
    report_review_status = qc_metadata_db['qc_submit_status']

    return qc_logs, metrics_records, images_records,\
           report_generation_status, module_group_status, report_review_status


def get_report_components_collections(mongo_connection):
    """connect to the relevant qc report components collections

    Parameters
    ----------
    mongo_connection : database connection
        read only connection to the production
        qc database

    Returns
    -------
    database connections
        connections to the following collections:
        - metrics
        - controlled_language_tags
    """
    report_components_db = mongo_connection['report_components']
    
    metrics = report_components_db['metrics']
    controlled_language_tags = report_components_db['controlled_language_tags']
    return metrics, controlled_language_tags


############################
#    CONNECTION HERE SO QUERY FUNCTIONS WORK
###########################
mongo_connection = connect_to_mouseqc_production()

qc_logs, metrics_records,\
images_records, report_generation_status,\
module_group_status, report_review_status = get_records_collections(mongo_connection)

metrics, controlled_language_tags = get_report_components_collections(mongo_connection)


TODAY = datetime.today().date() # today's date

#####################################################################
#
#         QUERY BASIC INFORMATION/ METADATA
#
#####################################################################


############################
#   GET ID LISTS
###########################

def get_session_ids_for_date_range(start_date:datetime, 
                                   end_date:datetime)-> tuple:
    """queries the production database for all ophys session ids
    within a date range

    Parameters
    ----------
    start_date : datatime object
        just the date
    end_date : datatime object
        just the date

    Returns
    -------
    pd.DataFrame
        table with columns:
            date_time
            ophys_session_id
    list
        list of ophys_session_ids for the given date range
    """
    session_ids = metrics_records.aggregate([
        {'$match': {
            'lims_ophys_session.date_of_acquisition': {
                '$gte': start_date, 
                '$lt' : end_date}}}, 
        {'$addFields': {
            'date'            :'$lims_ophys_session.date_of_acquisition',
            'ophys_session_id': '$lims_ophys_session.id'}}, 
        {'$project': {
            'ophys_session_id': 1, 
            'date'            : 1}}
    ])
    
    ids_df = query_results_to_df(session_ids)
    ids_df = ids_df.rename(columns={"date": "date_time"})
    ids_list = ids_df["ophys_session_id"].tolist()
    return ids_df, ids_list


def get_experiment_ids_for_session_ids(ophys_session_list:list)-> tuple:
    """looks up the experiment ids for a list of ophys session ids

    Parameters
    ----------
    ophys_session_list : list
        list of ophys session ids

    Returns
    -------
    pd.DataFrame
       dataframe with the experiment ids for the ophys session ids
       columns:
            ophys_session_id,
            ophys_experiment_id
    list
        list of just ophys experiment ids

    """
    exps_for_sess = metrics_records.aggregate([
        {'$match': {
            'data_id': {'$in': ophys_session_list}}}, 
        {'$addFields': {
            'ophys_session_id'   : '$lims_ophys_session.id', 
            'ophys_experiment_id': '$lims_ophys_session.ophys_experiment_ids'}}, 
        {'$project': {
            'ophys_session_id'   : 1, 
            'ophys_experiment_id': 1}},
        {'$unwind': {
            'path': '$ophys_experiment_id', 
            'preserveNullAndEmptyArrays': True}}
        ])
    exp_id_df = query_results_to_df(exps_for_sess)
    exp_id_list = exp_id_df["ophys_experiment_id"].tolist()
    return exp_id_df, exp_id_list


def get_session_ids_for_mouse_ids(mouse_ids_list:list)-> pd.DataFrame:
    """gets all ophys session ids for a list of mouse ids

    Parameters
    ----------
    mouse_ids_list : list of strings
        list of mouse ids 

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            ophys_session_id
            date
            mouse_id
    """
    session_ids = metrics_records.aggregate([
        {'$match': {
            'lims_ophys_session.external_specimen_name': {'$in': mouse_ids_list}}}, 
        {'$addFields': {
            'date'            :'$lims_ophys_session.date_of_acquisition',
            'ophys_session_id': '$lims_ophys_session.id',
            'mouse_id'        : '$lims_ophys_session.external_specimen_name'}}, 
        {'$project': {
            'ophys_session_id': 1, 
            'date'            : 1, 
            'mouse_id'        : 1}}
    ])
    
    ids_df = query_results_to_df(session_ids)
    ids_df = ids_df.rename(columns={"date": "date_time"})
    return ids_df


def get_experiment_ids_for_mouse_ids(mouse_ids_list:list)-> pd.DataFrame:
    """gets all ophys experiment ids for a list of mouse ids

    Parameters
    ----------
    mouse_ids_list : list of strings
        list of mouse ids

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            ophys_session_id,
            ophys_experiment_id,
            mouse_id
            date
    """
    exps_for_mice = metrics_records.aggregate([
        {'$match': {
            'lims_ophys_session.external_specimen_name': {'$in': mouse_ids_list}}}, 
        {'$addFields': {
            'ophys_session_id'   : '$lims_ophys_session.id', 
            'ophys_experiment_id': '$lims_ophys_session.ophys_experiment_ids',
            'mouse_id'           : '$lims_ophys_session.external_specimen_name',
            'date'               : '$lims_ophys_session.date_of_acquisition'}}, 
        {'$project': {
            'ophys_session_id'   : 1, 
            'ophys_experiment_id': 1, 
            'mouse_id'           : 1,
            'date'               : 1}},
        {'$unwind': {
            'path': '$ophys_experiment_id', 
            'preserveNullAndEmptyArrays': True}}
        ])
    exp_id_df = query_results_to_df(exps_for_mice)
    return exp_id_df


##################################
#  GET METADATA 
#  Session & Experiment
##################################


def get_metadata_for_session_ids(session_ids_list:list)-> pd.DataFrame:
    """pulls basic information/metadata for ophys sessions
    from the database

    Parameters
    ----------
    session_ids_list : list
        ophys session ids

    Returns
    -------
    pd.DataFrame
        gets basic information for ophys sessions
        columns:
            ophys_session_id,
            genotype,
            mouse_id,
            operator,
            rig,
            project,
            date_time,
    """
    session_records = metrics_records.aggregate([
    {'$match': {
            'lims_ophys_session.id':{'$in': session_ids_list}}}, 
    {'$addFields': {
            'ophys_session_id': '$lims_ophys_session.id', 
            'genotype':         '$lims_ophys_session.genotype', 
            'mouse_id':         '$lims_ophys_session.external_specimen_name', 
            'operator':         '$lims_ophys_session.operator', 
            'rig':              '$lims_ophys_session.rig', 
            'project':          '$lims_ophys_session.project', 
            'date':             '$lims_ophys_session.date_of_acquisition', 
            'stimulus':         '$lims_ophys_session.stimulus_name'}}, 
    { '$project': {
            'ophys_session_id': 1, 
            'genotype': 1, 
            'mouse_id': 1, 
            'operator': 1, 
            'rig'     : 1, 
            'project' : 1, 
            'date'    : 1, 
            'stimulus': 1}}
     ])
    
    sessions_df = query_results_to_df(session_records)
    sessions_df = sessions_df.rename(columns={"data_id":"ophys_session_id", 
                                              "date": "date_time"})
    sessions_df = clean_metadata_records_df(sessions_df)
    return sessions_df


def get_experiment_metadata_for_ids(experiment_ids_list:list)-> pd.DataFrame:
    """gets basic information/metadata for ophys experiments
    from the database

    Parameters
    ----------
    experiment_ids_list : list
       ophys experiment ids

    Returns
    -------
    pd.DataFrame
        dataframe with basic info for ophys experiments
        columns:
            genotype,
            mouse_id,
            project,
            stimulus,
            ophys_experiment_id,
            ophys_session_id,
            area,
            depth,
            datetime,
            date,
            week,
    """
    exp_info = metrics_records.aggregate([
        {'$match': {
            'lims_ophys_experiment.id': {'$in': experiment_ids_list}}}, 
        {'$addFields': {
            'ophys_experiment_id': '$lims_ophys_experiment.id', 
            'ophys_session_id'   : '$lims_ophys_experiment.ophys_session.id', 
            'area'               : '$lims_ophys_experiment.area', 
            'depth'              : '$lims_ophys_experiment.depth'}},
        {'$lookup': {
            'from'        : 'metrics', 
            'localField'  : 'ophys_session_id', 
            'foreignField': 'lims_ophys_session.id', 
            'as'          : 'session_info'}},
        {'$unwind': {
            'path': '$session_info', 
            'preserveNullAndEmptyArrays': False}},
        {'$addFields': {
            'datetime': '$session_info.lims_ophys_session.date_of_acquisition', 
            'genotype': '$session_info.lims_ophys_session.genotype', 
            'mouse_id': '$session_info.lims_ophys_session.external_specimen_name', 
            'project' : '$session_info.lims_ophys_session.project', 
            'stimulus': '$session_info.lims_ophys_session.stimulus_name'}},
        {'$project': {
            'genotype': 1, 
            'mouse_id': 1, 
            'project' : 1, 
            'stimulus': 1, 
            'ophys_experiment_id': 1, 
            'ophys_session_id'   : 1, 
            'area'    : 1, 
            'depth'   : 1, 
            'datetime': 1}}
    ])
    exp_df = query_results_to_df(exp_info)
    exp_df = clean_metadata_records_df(exp_df,
                                       date_col="datetime",
                                       project_col= None)
    return exp_df


def session_metadata_within_dates(start_date:datetime, 
                                  end_date:datetime)-> pd.DataFrame:
    """gets basic information for ophys sessions within a date range

    Parameters
    ----------
    start_date : datetime
        earliest date
    end_date : datetime
        latest date

    Returns
    -------
    pd.DataFrame
        dataframe of basic information for ophys sessions
        columns:
            ophys_session_id,
            genotype,
            mouse_id,
            operator,
            rig,
            project,
            date_time,
            stimulus
    """
    timed_records = metrics_records.aggregate([
    {'$match': {
            'lims_ophys_session.date_of_acquisition': {
                '$gte': start_date, 
                '$lt': end_date}}}, 
    {'$addFields': {
            'ophys_session_id': '$lims_ophys_session.id', 
            'genotype':         '$lims_ophys_session.genotype', 
            'mouse_id':         '$lims_ophys_session.external_specimen_name', 
            'operator':         '$lims_ophys_session.operator', 
            'rig':              '$lims_ophys_session.rig', 
            'project':          '$lims_ophys_session.project', 
            'date':             '$lims_ophys_session.date_of_acquisition', 
            'stimulus':         '$lims_ophys_session.stimulus_name'}}, 
    { '$project': {
            'ophys_session_id': 1, 
            'genotype': 1, 
            'mouse_id': 1, 
            'operator': 1, 
            'rig'     : 1, 
            'project' : 1, 
            'date'    : 1, 
            'stimulus': 1}}
    ])
    
    sessions_df = query_results_to_df(timed_records)
    sessions_df = sessions_df.rename(columns={"data_id":"ophys_session_id", 
                                              "date": "date_time"})
    sessions_df = clean_metadata_records_df(sessions_df)
    return sessions_df


#####################################################################
#
#         QUERY QC STATUS (Report generation, review, qc outcomes)
#
#####################################################################

def get_report_generation_status(id_list:list)-> pd.DataFrame:
    """gets the report generation status from the database
    for a list of ids

    Parameters
    ----------
    id_list : list
        list of ids to query the generation
    Returns
    -------
    pd.DataFrame
       dataframe with the generation status for the ids
       with the columns: 
        data_id,
        generation_status
    """
    gen_status = report_generation_status.aggregate([
        {'$match': {
            'data_id': {'$in': id_list}, 
            'current': True}}, 
        {'$project': {
            'data_id': 1, 
            'status' : 1}}
    ])
    gen_df = query_results_to_df(gen_status)
    gen_df = gen_df.rename(columns={"status" : "generation_status"})
    gen_df.loc[gen_df["generation_status"] == "ready", "generation_status"] = "complete"
    gen_df = replace_all_nan_with_missing(gen_df)
    return gen_df


def map_qc_review_status(qc_submit_status:str)-> str:
    """maps the database qc review status to whether or not the 
    review process is completed. The review status is broken down into
    status:     
    "complete":   ["pass", "flag", "fail"],
    "incomplete": ["incomplete", "ready"],
    "error":      ["error"]

    Parameters
    ----------
    qc_submit_status : str
        the input status from the database:
        ["pass", "flag", "fail", "incomplete", "ready", "error"]


    Returns
    -------
    str
        complete, incomplete or error
    """
    for key, value in QC_SUBMIT_STATUS_DICT.items():
        if qc_submit_status in value:
            return key
    return "Cannot find review status for {}".format(qc_submit_status)


def map_qc_outcome_status(qc_submit_status:str)-> str:
    """maps the database qc review status to the qc outcome.
    "pass" : ["pass"],
    "flag" : ["flag"],
    "fail" : ["fail"],
    "error": ["error"],
    "missing"   : ["missing"],
    "incomplete": ["incomplete", "ready"],

    Parameters
    ----------
    qc_submit_status : str
       the input status from the database:
        ["pass", "flag", "fail", "incomplete", "ready", 
        "error", "missing"]

    Returns
    -------
    str
        pass, flag, fail, error, missing, incomplete
    """
    for key, value in QC_OUTCOME_STATUS_DICT.items():
        if qc_submit_status in value:
            return key
    return "Cannot find review status for {}".format(qc_submit_status)


def gen_review_status_and_qc_outcomes(id_list:list)-> pd.DataFrame:
    """pulls the review status from the DB,
    creates a new column "review status" that separates whether the 
    review process is complete from the qc outcome.
    Maps the review status to complete, incomplete, error
    Maps the qc outcome to pass, flag, fail, error


    Parameters
    ----------
    id_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    pd.DataFrame
        dataframe with the review status for the ids
        with the columns: 
            data_id,
            review_status,
            qc_outcome
    """
    # get review status from DB
    rvw_status = report_review_status.aggregate([
        {'$match': {
            'data_id': {'$in': id_list}, 
            'current': True}}, 
        {'$project': {
            'data_id': 1, 
            'status' : 1}}
    ])
    rvw_df = query_results_to_df(rvw_status)

    # clean up and rename columns
    rvw_df = rvw_df.rename(columns={"status" : "qc_status"})

    # map review status from db to complete, incomplete, error
    rvw_df["review_status"] = rvw_df["qc_status"].apply(map_qc_review_status)

    # map qc outcome from db to pass, flag, fail, error
    rvw_df["qc_outcome"] = rvw_df["qc_status"].apply(map_qc_outcome_status)
    rvw_df = rvw_df.drop(columns=["qc_status"])
    rvw_df = replace_all_nan_with_missing(rvw_df)
    return rvw_df


def generate_qc_status_df_for_ids(id_list:list)-> pd.DataFrame:
    """pulls the qc generation status and the review status
    for a list of ids

    Parameters
    ----------
    id_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    pd.DataFrame
        dataframe with the following columns:
            data_id,
            generation_status,
            review_status,
            qc_outcome
    """
    qc_gen_status_df = get_report_generation_status(id_list)
    rev_status_df    = gen_review_status_and_qc_outcomes(id_list)
    report_status_df = qc_gen_status_df.merge(rev_status_df,
                                              how = "left",
                                              left_on="data_id",
                                              right_on = "data_id")
    report_status_df = replace_all_nan_with_missing(report_status_df)
    return report_status_df



#####################################################################
#
#         FINE LEVEL QC DETAIL (Controlled language tags, data stream outcomes etc.)
#
#####################################################################

##################################
#  Controlled Language Tags
#
##################################

def get_tags_for_ids(ids_list:list)-> pd.DataFrame:
    """gets all controlled langaugae tags
    associated with the listed ids

    Parameters
    ----------
    ids_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            data_id,
            qc_tag,
            qc_outcome,
            impacted_data
    """
    # get tag qc outcomes
    clt_outcomes = build_CLtag_outcomes_table()
    # get impacted data
    impacted_data = build_impacted_data_table()

    # query to get tag records
    id_tags = qc_logs.aggregate([
        {'$match': {
            'data_id': {'$in': ids_list},
            'qc_tag' : {'$nin': MANUAL_OVERRIDE_LIST},
            'current': True}},
        {'$match': {
            'qc_tag':{'$nin':['qc_pass', 'qc_flag_other']}}},
        {'$project': {
            'data_id'    : 1, 
            'qc_tag'     : 1, 
            'metric_name': 1}}
    ])
    tags_records = query_results_to_df(id_tags)

    # merge tag records with outcomes
    tags_df = tags_records.merge(clt_outcomes,
                           how = "left", 
                           left_on = "qc_tag",
                           right_on = "qc_tag")
    # just one qc_tags column
    tags_df['metric_name']= tags_df['metric_name'].fillna(tags_df['qc_tag'])
    tags_df = tags_df.drop(columns=['qc_tag'])
    tags_df = tags_df.rename(columns={"metric_name":"qc_tag"})
    
    # merge with impacted data
    tags_df = tags_df.merge(impacted_data,
                            how = "left",
                            left_on = "qc_tag",
                            right_on = "qc_tag")
    tags_df = tags_df[["data_id", 'qc_tag',"qc_outcome", "impacted_data"]]
    return tags_df


def get_other_tags_for_ids(ids_list:list)-> pd.DataFrame:
    """gets all other tags for the listed ids "other" tags are tags that
    are not distinct controlled language tags yet- so the notes column describes
    the issue

    Parameters
    ----------
    ids_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    dataframe
        table with the following columns:
            data_id,
            qc_tag,
            module,
            note
    """
    other_tags = qc_logs.aggregate([
        {'$match': {
            'data_id': {'$in': ids_list},
            'current': True, 
            'qc_tag' : 'qc_flag_other'}}, 
        {'$unwind': {
            'path': '$module', 
            'preserveNullAndEmptyArrays': True}}, 
        {'$project': {
            'data_id': 1, 
            'qc_tag' : 1, 
            'module' : 1, 
            'note'   : 1}}
    ])
    other_tags_df = query_results_to_df(other_tags)
    return other_tags_df


def get_manual_overrides_for_ids(ids_list:list)-> pd.DataFrame:
    """gets all manual override tags for the listed ids

    Parameters
    ----------
    ids_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            data_id,
            qc_tag,
            module
    """
    manual_overrides = qc_logs.aggregate([
        {'$match': {
            'data_id': {'$in': ids_list},
            'qc_tag' : {'$in': MANUAL_OVERRIDE_LIST},
            'current': True}},
        {'$unwind': {
            'path': '$module', 
            'preserveNullAndEmptyArrays': True}}, 
        {'$project': {
            'data_id': 1, 
            'qc_tag' : 1, 
            'module' : 1}}
    ])
    
    overrides_df = pd.DataFrame(list(manual_overrides))
    return overrides_df


def get_all_tags_for_ids(ids_list:list)->tuple:
    """gets all controlled langauge tags, other tags and manual override
    tags for the listed ids

    Parameters
    ----------
    ids_list : list
        ids- ophys session or ophys experiment

    Returns
    -------
    pd.DataFrame
        tags_df, other_tags_df, overrides_df
        tags_df: table with the following columns:
            data_id,
            qc_tag,
            qc_outcome,
            impacted_data

    pd.DataFrame
        other_tags_df: table with the following columns:
            data_id,
            qc_tag,
            module,
            note

    pd.DataFrame
        overrides_df: table with the following columns:
            data_id,
            qc_tag,
            module
    """
    tags_df = get_tags_for_ids(ids_list)
    other_tags_df = get_other_tags_for_ids(ids_list)
    overrides_df = get_manual_overrides_for_ids(ids_list)
    return tags_df, other_tags_df, overrides_df


##################################
#  DATA STREAM OUTCOMES
#
##################################


def gen_impacted_data_df(qc_outcome_df:pd.DataFrame,
                         tags_df:pd.DataFrame,
                         impacted_data:list,
                         id_column="data_id")-> pd.DataFrame:
    """a function to generate the impacted data for a list of data ids
    that have completed the qc review process.

    Parameters
    ----------
    qc_outcome_df : pd.DataFrame
        dataframe of qc outcomes, consists of the following columns:
            data_id (or another column that has the data id)
            qc_outcome
    tags_df : pd.DataFrame
        dataframe of all controlled langauge tags and metric tags, consists
        of the following columns:
            data_id (or another column that has the data id)
            qc_tag
            qc_outcome
            impacted_data
    impacted_data : list
        a list that contains all the impacted data to check for
    id_column : str, optional
        the column that contains the ids, for example "ophys_session_id" 
        or "ophys_experiment_id, by default "data_id"

    Returns
    -------
    pd.DataFrame
        dataframe with the qc outcomes for the impacted data
        a column for each item in the impacted data list, and 
        the id column
    """
    
    # Initialize the result DataFrame
    result_df = pd.DataFrame(columns=[id_column] + impacted_data)
    result_df[id_column] = qc_outcome_df[id_column]
    
    # Set all initial values to "pass"
    for data in impacted_data:
        result_df[data] = 'pass'
    
    # Iterate through each row in the session_qc_outcome DataFrame
    for index, row in qc_outcome_df.iterrows():
        data_id = row[id_column]
        qc_outcome = row['qc_outcome']
        
        # If qc_outcome is "pass", continue (all default to "pass")
        if qc_outcome == 'pass':
            continue
        
        # If qc_outcome is "flag" or "fail", update outcomes based on session_tags
        tag_rows = tags_df[tags_df[id_column] == data_id]
        for _, tag_row in tag_rows.iterrows():
            impacted_data_field = tag_row['impacted_data']
            if impacted_data_field in impacted_data:
                result_df.loc[result_df[id_column] == data_id, impacted_data_field] = tag_row['qc_outcome']
    
    return result_df




#####################################################################
#
#          MAIN FUNCTIONS
#
#####################################################################


def gen_session_qc_info_for_date_range(end_date_str:int = None, 
                                       range_in_days:int = 21,
                                       csv_name: str = "session_qc_info_{}.csv".format(TODAY),
                                       csv_path: str = None)-> pd.DataFrame:
    """gets metadata and qc status information for ophys sessions
    within a date range. 

    Will default to the last 21 days if no end date or range is provided

    Parameters
    ----------
    end_date_str : int, optional
        last date, by default None- will use today's date
    range_in_days : int, optional
        number of days before the end date, by default 21
    csv_name : str, optional
        name of the csv file, 
        by default "session_qc_info_{todays date}.csv"
    csv_path : str, optional
        path to save the csv file, by default None

    Returns
    -------
    pd.DataFrame
        dataframe of basic info and qc status for ophys sessions
        columns:
            project_group:     str
            project:           str
            mouse_id:          str
            genotype:          str
            date:              datetime.date
            ophys_session_id:  int64
            stimulus:          str
            generation_status: str
            review_status:     str
            qc_outcome:        str
            operator:          str
            rig:               str
            date_time:         datetime64[ns]
            week:              str
    """
    
    start_date, end_date = set_date_range(end_date_str, range_in_days)
    
    metadata_df = session_metadata_within_dates(start_date, end_date)
    sessions_list = metadata_df["ophys_session_id"].tolist()
    
    qc_gen_status_df = get_report_generation_status(sessions_list)
    qc_gen_status_df = qc_gen_status_df.rename(columns={"data_id":"ophys_session_id"})
    
    qc_rev_status_df = gen_review_status_and_qc_outcomes(sessions_list)
    qc_rev_status_df = qc_rev_status_df.rename(columns={"data_id":"ophys_session_id"})
    
    session_qc_info_df = metadata_df.merge(qc_gen_status_df,
                                           how = "left",
                                           left_on="ophys_session_id",
                                           right_on = "ophys_session_id")
    
    session_qc_info_df = session_qc_info_df.merge(qc_rev_status_df,
                                                  how = "left",
                                                  left_on= "ophys_session_id",
                                                  right_on= "ophys_session_id")
    
    # clean up final dataframe
    session_qc_info_df = replace_all_nan_with_missing(session_qc_info_df)
    session_qc_info_df.sort_values(by=["project_group",
                                             "mouse_id","date"], inplace=True)
    session_qc_info_df = session_qc_info_df[["project_group",
                                             "project",
                                             "mouse_id",
                                             "genotype",
                                             "date",
                                             "week",
                                             "ophys_session_id",
                                             "stimulus",
                                             "generation_status",
                                             "review_status",
                                             "qc_outcome",
                                             "operator",
                                             "rig",
                                             "date_time"]]
    session_qc_info_df.reset_index(drop=True, inplace=True)

    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        session_qc_info_df.to_csv(full_path, index=False)

    return session_qc_info_df


def gen_session_qc_info_for_ids(session_ids_list:list,
                                csv_name: str = "session_qc_info_{}.csv".format(TODAY),
                                csv_path: str = None)-> pd.DataFrame:
    """table with basic information/metadata and qc status for ophys sessions

    Parameters
    ----------
    session_ids_list : list
        ophys session ids

    Returns
    -------
    pd.DataFrame
        dataframe of basic info and qc status for ophys sessions
        columns:
            project_group:     str
            project:           str
            mouse_id:          str
            genotype:          str
            date:              datetime.date
            week:              str
            ophys_session_id:  int64
            stimulus:          str
            generation_status: str
            review_status:     str
            qc_outcome:        str
            operator:          str
            rig:               str
            date_time:         datetime64[ns]
    """
    metadata_df = get_metadata_for_session_ids(session_ids_list)
    
    qc_gen_status_df = get_report_generation_status(session_ids_list)
    qc_gen_status_df = qc_gen_status_df.rename(columns={"data_id":"ophys_session_id"})
    
    qc_rev_status_df = gen_review_status_and_qc_outcomes(session_ids_list)
    qc_rev_status_df = qc_rev_status_df.rename(columns={"data_id":"ophys_session_id"})
    
    session_qc_info_df = metadata_df.merge(qc_gen_status_df,
                                           how      = "left",
                                           left_on  ="ophys_session_id",
                                           right_on = "ophys_session_id")
    
    session_qc_info_df = session_qc_info_df.merge(qc_rev_status_df,
                                                  how      = "left",
                                                  left_on  = "ophys_session_id",
                                                  right_on = "ophys_session_id")
    
    # clean up final dataframe
    session_qc_info_df = replace_all_nan_with_missing(session_qc_info_df)
    session_qc_info_df.sort_values(by=["project_group",
                                       "mouse_id",
                                       "date"], inplace=True)
    
    session_qc_info_df = session_qc_info_df[["project_group",
                                             "project",
                                             "mouse_id",
                                             "genotype",
                                             "date",
                                             "week",
                                             "ophys_session_id",
                                             "stimulus",
                                             "generation_status",
                                             "review_status",
                                             "qc_outcome",
                                             "operator",
                                             "rig",
                                             "date_time"]]
    session_qc_info_df.reset_index(drop=True, inplace=True)
    
    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        session_qc_info_df.to_csv(full_path, index=False)

    return session_qc_info_df


def current_mice_df(df:pd.DataFrame,
                    csv_name: str = "active_mouse_summary_{}.csv".format(TODAY),
                    csv_path: str = None)-> pd.DataFrame:
    """ takes the session_qc_info_df and returns the
    last entry for each mouse

    Parameters
    ----------
    df : pd.DataFrame
        session_qc_info_df

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            project_group: str,
            project:  str,
            mouse_id: str,
            genotype: str,
            date:     datetime.date,
            stimulus: str
    """
    # Ensure the dataframe is sorted by mouse_id and date_time
    df = df.sort_values(by=['mouse_id', 'date_time'])
    
    # Get the last entry for each mouse
    last_entries = df.groupby('mouse_id').last().reset_index()
    
    # Select only the necessary columns
    result = last_entries[['project_group',
                           'project',
                           'mouse_id',
                           'genotype',
                           'date',
                           'stimulus']]
    result = result.sort_values(by=['project_group', 'project', 'genotype']).reset_index(drop=True)
    
    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        result.to_csv(full_path, index=False)
    
    return result


def gen_experiment_qc_info_for_ids(experiment_ids_list:list,
                                   csv_name: str = "experiment_qc_info_{}.csv".format(TODAY),
                                   csv_path: str = None)-> pd.DataFrame:
    """gets basic information for ophys experiments
    and the qc status for the experiments

    Parameters
    ----------
    experiment_ids_list : list
        ophys experiment ids

    Returns
    -------
    pd.DataFrame
       dataframe with the qc info for the experiments.
       Includes columns:
        - mouse_id:            str,
        - genotype:            str,
        - ophys_session_id:    int64,
        - ophys_experiment_id: int64,
        - area:                str,
        - depth:               int64,
        - generation_status:   str,
        - review_status:       str,
        - qc_outcome:          str,
        - stimulus:            str,
        - datetime:            datetime64[ns],
        - project:             str
    """

    metadata_df = get_experiment_metadata_for_ids(experiment_ids_list)
    
    qc_gen_status_df = get_report_generation_status(experiment_ids_list)
    qc_gen_status_df = qc_gen_status_df.rename(columns={"data_id":"ophys_experiment_id"})
    
    qc_rev_status_df = gen_review_status_and_qc_outcomes(experiment_ids_list)
    qc_rev_status_df = qc_rev_status_df.rename(columns={"data_id":"ophys_experiment_id"})
    
    exp_qc_info_df = metadata_df.merge(qc_gen_status_df,
                                       how      = "left",
                                       left_on  = "ophys_experiment_id",
                                       right_on = "ophys_experiment_id")
    
    exp_qc_info_df = exp_qc_info_df.merge(qc_rev_status_df,
                                          how      = "left",
                                          left_on  = "ophys_experiment_id",
                                          right_on = "ophys_experiment_id")
    # clean up final dataframe
    exp_qc_info_df = replace_all_nan_with_missing(exp_qc_info_df)
    exp_qc_info_df.sort_values(by=["ophys_session_id",
                                   "area",
                                   "depth"], inplace=True)
    exp_qc_info_df.reset_index(drop=True, inplace=True)
    exp_qc_info_df = exp_qc_info_df[["mouse_id",
                                     "genotype",
                                     "ophys_session_id",
                                     "ophys_experiment_id",
                                     "date",
                                     "area",
                                     "depth",
                                     "generation_status",
                                     "review_status",
                                     "qc_outcome",
                                     "stimulus",
                                     "week",
                                     "datetime",
                                     "project"]]
    
    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        exp_qc_info_df.to_csv(full_path, index=False)

    return exp_qc_info_df


def gen_tags_df_for_ids(ophys_session_ids:list,
                        ophys_experiment_ids:list)-> tuple:
    """gets all controlled langauge tags, other tags and manual override
    tags for the listed ids, and whether those are experiments or sessions

    Parameters
    ----------
    ophys_session_ids : list
        session ids
    ophys_experiment_ids : list
        experiment ids

    Returns
    -------
    tuple: tags_df, other_tags_df, overrides_df
    pd.DataFrame
        tags_df: table with the following columns:
            data_id:       int64,
            qc_tag:        str,
            qc_outcome:    str,
            impacted_data: str
            report_type:   str

    pd.DataFrame
        other_tags_df: table with the following columns:
            data_id:     int64,
            qc_tag:      str,
            module:      str,
            note:        str,
            report_type: str

    pd.DataFrame
        overrides_df: table with the following columns:
            data_id:     int64,
            qc_tag:      str,
            module:      str,
            report_type: str
    """

    session_tags_df, \
    session_other_tags_df, \
    session_overrides_df = get_all_tags_for_ids(ophys_session_ids)
    
    session_tags_df["report_type"] = "session"
    session_other_tags_df["report_type"] = "session"
    session_overrides_df["report_type"] = "session"

    experiment_tags_df, \
    experiment_other_tags_df, \
    experiment_overrides_df = get_all_tags_for_ids(ophys_experiment_ids)
    
    experiment_tags_df["report_type"] = "experiment"
    experiment_other_tags_df["report_type"] = "experiment"
    experiment_overrides_df["report_type"] = "experiment"

    tags_df = pd.concat([session_tags_df, \
                         experiment_tags_df],\
                        ignore_index=True)
    
    other_tags_df = pd.concat([session_other_tags_df, \
                               experiment_other_tags_df],\
                              ignore_index=True)
    
    overrides_df = pd.concat([session_overrides_df,\
                              experiment_overrides_df],\
                             ignore_index=True)
    
    return tags_df, other_tags_df, overrides_df


def gen_session_impacted_data_outcome_df(ophys_session_ids:list,
                                         csv_name: str = "session_impacted_data_{}.csv".format(TODAY),
                                         csv_path: str = None)-> pd.DataFrame:
    """the main function to generate the impacted data for a list of session ids
    that have completed the qc review process. 

    Parameters
    ----------
    ophys_session_ids : list
        session ids, these sessions must have completed qc review

    Returns
    -------
    pd.DataFrame
        dataframe with the ophys_session_id + columns for each entry
        in the OPHYS_SESSION_IMPACTED_DATA list
        columns:
            "ophys_session_id",
            "physio_sync",
            "eye_tracking",
            "face_tracking",
            "body_tracking",
            "running_wheel",
            "water_delivery",
            "lick_detection",
            "visual_stimulus",
            "metadata",
            "columnar_zstack",
            "mouse_stress",
            "mouse_health",
            "brain_health",
            "stimulus_sequence_progression",
            "task_performance",
            "mouse_behavior"
    """
    qc_outcome_df = generate_qc_status_df_for_ids(ophys_session_ids)
    tags_df, _, _ = get_all_tags_for_ids(ophys_session_ids)
    impacted_data = OPHYS_SESSION_IMPACTED_DATA
    session_impacted_data_df = gen_impacted_data_df(qc_outcome_df,
                                                    tags_df,
                                                    impacted_data)
    
    session_impacted_data_df = session_impacted_data_df.rename(columns={"data_id":"ophys_session_id"})
    
    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        session_impacted_data_df.to_csv(full_path, index=False)
    return session_impacted_data_df


def gen_experiment_impacted_data_outcome_df(ophys_experiment_ids:list,
                                            csv_name: str = "experiment_impacted_data_{}.csv".format(TODAY),
                                            csv_path: str = None)-> pd.DataFrame:
    """the main function to generate the impacted data for a list of
    experiment ids that have completed the qc review process

    Parameters
    ----------
    ophys_experiment_ids : list
        ophys experiment ids, these experiments must have completed qc review

    Returns
    -------
    pd.DataFrame
        table with experiment id and a column for each entry in the
        OPHYS_EXPERIMENT_IMPACTED_DATA list
        columns:
            "ophys_experiment_id",
            "fov_centered_zstack",
            "physiology_recording",
            "metadata",
            "brain_health",
            "FOV_matching"
    """
    qc_outcome_df = gen_experiment_qc_info_for_ids(ophys_experiment_ids)
    tags_df, _, _ = get_all_tags_for_ids(ophys_experiment_ids)
    impacted_data = OPHYS_EXPERIMENT_IMPACTED_DATA
    experiment_impacted_data_df = gen_impacted_data_df(qc_outcome_df,
                                                       tags_df,
                                                       impacted_data)
    # Save as CSV if a path is provided
    if csv_path:
        full_path = os.path.join(csv_path, csv_name)
        experiment_impacted_data_df.to_csv(full_path, index=False)

    return experiment_impacted_data_df


#####################################################################
#
#         CURRENT DATABASE DATA QUERIES
#
#####################################################################

def get_unique_qc_submit_statuses()->pd.DataFrame:
    """creates a dataframe of all unique submit statuses

    Returns
    -------
    pd.DataFrame
        _description_
    """
    qc_statuses = report_review_status.aggregate([
    {'$match': {
            'current': True}},
    {'$group': {
            '_id': '$status'}}, 
    {'$project': {
            '_id': 0, 
            'submit_status': '$_id'}}
    ])
    submit_statuses_df = pd.DataFrame(list(qc_statuses))
    return submit_statuses_df


def get_unique_qc_generation_statuses()->pd.DataFrame:
    """creates a dataframe of all unique qc report
    generation statuses

    Returns
    -------
    pd.DataFrame
        _description_
    """
    gen_statuses = report_generation_status.aggregate([
    {'$match': {
            'current': True}},
    {'$group': {
            '_id': '$status'}}, 
    {'$project': {
            '_id': 0, 
            'generation_status': '$_id'}}
    ])
    generation_statuses_df = pd.DataFrame(list(gen_statuses))
    return generation_statuses_df


def build_CLtag_outcomes_table()-> pd.DataFrame:
    """queries the production database and builds a table with the
    qc outcomes for all controlled language tags

    Returns
    -------
    dataframe
        dataframe with following columns:
            qc_tag,
            qc_outcome
    """
    tag_qc_outcomes = controlled_language_tags.aggregate([{
        '$project': {
            'name_db'   : 1, 
            'qc_outcome': 1}}
    ])
    outcomes_df = query_results_to_df(tag_qc_outcomes)
    outcomes_df = outcomes_df.rename(columns={"name_db":"qc_tag"})
    return outcomes_df


def build_impacted_data_table()-> pd.DataFrame:
    """Queries the production database and builds a table with the 
    impacted data for all controlled language tags and metrics that
    have active thresholds

    Returns
    -------
    pd.DataFrame
        table with the following columns:
            qc_tag,
            impacted_data

    """
    # Query controlled language tags and metrics for impacted data
    tag_impacted_data = controlled_language_tags.aggregate([
        {'$addFields': {
                'data_context': '$impacted_data.data_context', 
                'data_streams': '$impacted_data.data_streams'}}, 
        {'$project': {
                'name_db'     : 1, 
                'data_context': 1, 
                'data_streams': 1}}
    ])
    
    metric_impacted_data = metrics.aggregate([
        {'$match': {
                'threshold_info.thresholds_active': True}},
        {'$addFields': {
                'data_streams': '$impacted_data.data_streams', 
                'data_context': '$impacted_data.data_context'}},
        {'$project': {
                'name_db'     : 1, 
                'data_streams': 1, 
                'data_context': 1}}
    ])
    # build a dataframe from query results
    metrics_df = query_results_to_df(metric_impacted_data)
    clt_df = query_results_to_df(tag_impacted_data)
    # unify and clean up dataframe
    impacted_data = metrics_df.append(clt_df)
    impacted_data["impacted_data"] = impacted_data["data_context"] + impacted_data["data_streams"]
    impacted_data = impacted_data.drop(columns=['data_streams', 'data_context'])
    impacted_data = impacted_data.explode("impacted_data").reset_index(drop=True)
    impacted_data = impacted_data.rename(columns={"name_db":"qc_tag"})
    return impacted_data

def get_CLT_data_contexts()-> pd.DataFrame:
    """queries all controlled language tags and pulls
    all unique entries for data context

    Returns
    -------
    pd.DataFrame
        dataframe of all the data context entries
        associated with the controlled language tags
    """
    contexts = controlled_language_tags.aggregate([
    {
        '$unwind': {
            'path': '$impacted_data', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$impacted_data.data_context'}}, 
        {'$project': {
            '_id': 0, 
            'data_context': '$_id'}}, 
        {'$unwind': {
            'path': '$data_context', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$data_context'}},
        {'$project': {
            '_id': 0, 
            'data_context': '$_id'}}
    ])
    contexts_df = pd.DataFrame(list(contexts))
    return contexts_df


def get_metric_data_contexts()-> pd.DataFrame:
    """queries all metrics and pulls
    all unique entries for data context

    Returns
    -------
    pd.DataFrame
        dataframe of all the data context entries
        associated with the metrics
    """
    contexts = metrics.aggregate([
        {'$unwind': {
            'path': '$impacted_data', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$impacted_data.data_context'}}, 
        {'$project': {
            '_id': 0, 
            'data_context': '$_id'}}, 
        {'$unwind': {
            'path': '$data_context', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$data_context'}},
        {'$project': {
            '_id': 0, 
            'data_context': '$_id'}}
    ])
    contexts_df = pd.DataFrame(list(contexts))
    return contexts_df


def get_active_data_contexts()-> pd.DataFrame:
    """pulls all unique data context entries
    from both controlled language tags and metrics 


    Returns
    -------
    pd.DataFrame
        dataframe of all the data context entries
        associated with the controlled language tags and metrics
    """
    CLT_contexts = get_CLT_data_contexts()
    metric_contexts = get_metric_data_contexts()
    all_contexts = CLT_contexts.append(metric_contexts).reset_index(drop=True)
    all_contexts = all_contexts.drop_duplicates()
    return all_contexts


def get_CLT_data_streams()-> pd.DataFrame:
    """queries all controlled language tags and pulls
    all unique entries for data streams

    Returns
    -------
    pd.DataFrame
        dataframe of all the datastream entries
        associated with the controlled language tags
    """
    streams = controlled_language_tags.aggregate([
    {
        '$unwind': {
            'path': '$impacted_data', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$impacted_data.data_steams'}}, 
        {'$project': {
            '_id': 0, 
            'data_streams': '$_id'}}, 
        {'$unwind': {
            'path': '$data_streams', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$data_streams'}},
        {'$project': {
            '_id': 0, 
            'data_streams': '$_id'}}
    ])
    streams_df = pd.DataFrame(list(streams))
    return streams_df


def get_metric_data_streams()-> pd.DataFrame:
    """queries all metrics and pulls
    all unique entries for data streams

    Returns
    -------
    pd.DataFrame
        dataframe of all the data stream entries
        associated with metrics
    """
    streams = metrics.aggregate([
    {
        '$unwind': {
            'path': '$impacted_data', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$impacted_data.data_steams'}}, 
        {'$project': {
            '_id': 0, 
            'data_streams': '$_id'}}, 
        {'$unwind': {
            'path': '$data_streams', 
            'preserveNullAndEmptyArrays': False}}, 
        {'$group': {
            '_id': '$data_streams'}},
        {'$project': {
            '_id': 0, 
            'data_streams': '$_id'}}
    ])
    streams_df = pd.DataFrame(list(streams))
    return streams_df


def get_active_data_streams()-> pd.DataFrame:
    """pulls all unique data stream entries
    from both controlled language tags and metrics

    Returns
    -------
    pd.DataFrame
        dataframe of all the data stream entries
        associated with the controlled language tags and metrics
    """
    CLT_streams = get_CLT_data_streams()
    metric_streams = get_metric_data_streams()
    all_streams = CLT_streams.append(metric_streams).reset_index(drop=True)
    all_streams = all_streams.drop_duplicates()
    return all_streams



#####################################################################
#
#          UTILITY FUNCTIONS
#
#####################################################################


def query_results_to_df(query_results)-> pd.DataFrame:
    """takes mongodb query results and puts them in a dataframe

    Parameters
    ----------
    query_results : _type_
        _description_

    Returns
    -------
    pd.DataFrame
        a dataframe of the query results
    """
    df = pd.DataFrame(list(query_results))
    df = df.drop(['_id'], axis=1)
    return df


def set_date_range(end_date_str=None, range_in_days=21):
    """Provides a start date and end date in datetime format that cover
    a range of days. If an end date is not provided it defaults to todays date

    Parameters
    ----------
    end_date_str : string, optional
        'mm-dd-yyy', will use today as default if nothing entered
    range_in_days : _int, optional
        _description_, by default 21:int

    Returns
    -------
    tuple of datetime
       start_date and end_date datetime objects
    """
    # If end_date_str is not provided, default to today's date
    if end_date_str is None:
        end_date = datetime.now().replace(hour= 0, minute=0, second=0, microsecond=0)
    else:
        end_date = datetime.strptime(end_date_str, '%m-%d-%Y')

    # Calculate start_date by subtracting the range_in_days from end_date
    start_date = end_date - timedelta(days=range_in_days)

    return start_date, end_date


def get_date_from_dttime(datetime_obj:datetime)-> datetime.date:
    """ strips the time from a datetime object and returns the date

    Parameters
    ----------
    datetime_obj : _type_
        _description_

    Returns
    -------
    date
        date object
    """
    # use if loading directly from mongodb
    return datetime_obj.date() 


def convert_str_to_dttime(date_str:str)-> datetime:
    """ converts a string date to a datetime object

    Parameters
    ----------
    date_str : string
        date in the format "mm-dd-yyyy"

    Returns
    -------
    date time object
        datetime object of the date
    """
    return datetime.strptime(date_str, '%m-%d-%Y')

def assign_week_range(df:pd.DataFrame, date_col:str)->pd.DataFrame:
    """
    Assigns a week range string in the format "mm_dd" - "mm_dd" to each row
    in the DataFrame based on the date column and adds it as a new column named 'week'.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the date column.
    date_col : str
        The name of the column containing the dates.
        
    Returns
    -------
    pd.DataFrame
        The DataFrame with a new column 'week' containing the week range strings for each row.
    """
    # Ensure the DataFrame is a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert the date column to datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Get the start of the week for each date
    df.loc[:, 'week_start'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
    df.loc[:, 'week_end'] = df['week_start'] + pd.to_timedelta(6, unit='d')

    # Format the week range as "mm_dd" - "mm_dd" and assign to new 'week' column
    df.loc[:, 'week'] = df['week_start'].dt.strftime('%m_%d') + ' - ' + df['week_end'].dt.strftime('%m_%d')

    # Drop the temporary columns
    df = df.drop(columns=['week_start', 'week_end'])

    return df

def generate_project_group(project_code:str)-> str:
    """ checks the project code against the project groups dictionary
    and returns the group name

    Parameters
    ----------
    project_code : str
        project code from the database

    Returns
    -------
    str
        the project group name
    """
    for key, value in PROJECT_GROUPS_DICT.items():
        if project_code in value:
            return key
    return "Cannot find project group for {}".format(project_code)


def manual_override_qc_outcome(manual_override_tag:str)-> str:
    """checks the manual override tag against the manual override outcomes
    dictionary and returns the qc outcome

    Parameters
    ----------
    manual_override_tag : str
        the manual override tag from the database

    Returns
    -------
    str
        qc outcome
    """
    for key, value in MANUAL_OVERRIDE_OUTCOMES_DICT.items():
        if manual_override_tag in value:
            return key
    return "Cannot find qc outcome for {}".format(manual_override_tag)


def clean_metadata_records_df(df: pd.DataFrame, 
                              date_col: str = "date_time",
                              project_col: str = "project") -> pd.DataFrame:
    """
    Adds a date column to the sessions dataframe and generates a project group
    and assigns a week range to each session if the respective columns are provided.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of session records
    date_col : str, optional
        Column name containing datetime information, by default None
    project_col : str, optional
        Column name containing project information, by default None
        
    Returns
    -------
    pd.DataFrame
        The input dataframe with the following columns added (if respective columns are provided):
            date: datetime.date
            project_group: str
            week: str
    """
    df = df.copy()
    
    # Add date column if date_col is provided
    if date_col:
        df['date'] = df[date_col].apply(get_date_from_dttime)
        df = assign_week_range(df, 'date')
    
    # Add project_group column if project_col is provided
    if project_col:
        df['project_group'] = df[project_col].apply(generate_project_group)
    
    return df

def replace_all_nan_with_missing(df):
    """
    Replace all NaN entries in a DataFrame with the string 'missing'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    
    Returns:
    pd.DataFrame: The DataFrame with all NaN values replaced.
    """
    df = df.fillna('missing')
    return df
