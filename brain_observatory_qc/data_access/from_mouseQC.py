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
    "learning_mfish": ['LearningmFISHDevelopment','LearningmFISHTask1A'],          # noqa: E241
    "omFISH_V1_U01":  ['omFISHRbp4Meso','omFISHSstMeso', 'U01BFCT'],               # noqa: E241, E501
    "open_scope":     ['OpenScopeDendriteCoupling','OpenScopeSequenceLearning'],   # noqa: E241, E501
    "hardware_dev":   ['MesoscopeDevelopment']                                     # noqa: E241, E501
}

QC_SUBMIT_STATUS_DICT = {
    "complete":   ["pass", "flag", "fail"],
    "incomplete": ["incomplete", "ready"],
    "error":      ["error"]
}

QC_STATUS_NA_LIST = ["incomplete", "ready", "error"]

#####################################################################
#
#           CONNECT TO MONGODB
#
#####################################################################

username = 'public'
password = 'public_password'
mongo_host = 'qc-sys-db'
mongo_port = '27017'

connection_string = 'mongodb://{}:{}@{}:{}/'.format(username, password, mongo_host, int(mongo_port))
mongo_connection = MongoClient(connection_string)


############################
#   COLLECTION CONNECTIONS
############################

## databases
records_db = mongo_connection['records']
qc_metadata_db = mongo_connection['qc_metadata']
report_components_db = mongo_connection['report_components']  

## Collections
qc_logs = records_db['qc_logs']
metrics_records = records_db['metrics']
images_records = records_db['images']

report_generation_status = qc_metadata_db['qc_generation_status']
module_group_status = qc_metadata_db['qc_module_group_status']
report_submit_status = qc_metadata_db['qc_submit_status']

metrics = report_components_db['metrics']
controlled_language_tags = report_components_db['controlled_language_tags']

############################
#    Connection Functs for jupyter notebooks
############################

def connect_to_mouseqc_production(username = 'public', password = 'public_password'):
    mongo_host = 'qc-sys-db'
    mongo_port = '27017'
    connection_string = 'mongodb://{}:{}@{}:{}/'.format(username, password, mongo_host, int(mongo_port))
    mongo_connection = MongoClient(connection_string)
    return mongo_connection


def get_records_collections(mongo_connection):
    records_db = mongo_connection['records']
    qc_metadata_db = mongo_connection['qc_metadata']

    qc_logs = records_db['qc_logs']
    metrics_records = records_db['metrics']
    images_records = records_db['images']

    report_generation_status = qc_metadata_db['qc_generation_status']
    module_group_status = qc_metadata_db['qc_module_group_status']
    report_submit_status = qc_metadata_db['qc_submit_status']

    return qc_logs, metrics_records, images_records, report_generation_status, module_group_status, report_submit_status


def get_report_components_collections(mongo_connection):
    report_components_db = mongo_connection['report_components']
    
    metrics = report_components_db['metrics']
    controlled_language_tags = report_components_db['controlled_language_tags']
    return metrics, controlled_language_tags


#####################################################################
#
#           QUERY FUNCTIONS
#
#####################################################################
def build_impacted_data_table()-> pd.DataFrame:
    """
    Queries the production database and builds a table with the 
    impacted data for all controlled language tags and metrics that
    have active thresholds
    """
    # Query controlled language tags and metrics for impacted data
    tag_impacted_data = controlled_language_tags.aggregate([
        {
            '$addFields': {
                'data_context': '$impacted_data.data_context', 
                'data_streams': '$impacted_data.data_streams'
            }
        }, {
            '$project': {
                'name_db': 1, 
                'data_context': 1, 
                'data_streams': 1
            }
        }])
    
    metric_impacted_data = metrics.aggregate([
        {
            '$match': {
                'threshold_info.thresholds_active': True
            }
        }, {
            '$addFields': {
                'data_streams': '$impacted_data.data_streams', 
                'data_context': '$impacted_data.data_context'
            }
        }, {
            '$project': {
                'name_db': 1, 
                'data_streams': 1, 
                'data_context': 1
            }
        }
    ])
    # build a dataframe from query results
    metrics_df = query_results_to_df(metric_impacted_data)
    clt_df = query_results_to_df(tag_impacted_data)
    # unify and clean up dataframe
    impacted_data = metrics_df.append(clt_df)
    impacted_data["impacted_data"] = impacted_data["data_context"] + impacted_data["data_streams"]
    impacted_data = impacted_data.drop(columns=['data_streams', 'data_context'])
    impacted_data = impacted_data.explode("impacted_data").reset_index(drop=True)
    return impacted_data

def get_session_ids_for_date_range(start_date, end_date)-> pd.DataFrame:
    session_ids = metrics_records.aggregate([
        {'$match': {
            'lims_ophys_session.date_of_acquisition': {
                '$gte': start_date, 
                '$lt': end_date}}}, 
        {'$addFields': {
            'date':'$lims_ophys_session.date_of_acquisition'}}, 
        { '$project': {
            'ophys_session_id': 1, 
            'date': 1}}
    ])
    
    ids_df = query_results_to_df(session_ids)
    ids_df = ids_df.rename(columns={"lims_id":"ophys_session_id", 
                                    "date": "date_time"})
    return ids_df




def session_records_within_dates(start_date, end_date)-> pd.DataFrame:
    timed_records = metrics_records.aggregate([
        {
            '$match': {
                'lims_ophys_session.date_of_acquisition': {
                    '$gte': start_date, 
                    '$lt':end_date
                }
            }
        }, {
            '$addFields': {
                'lims_id': '$lims_ophys_session.id', 
                'genotype': '$lims_ophys_session.genotype', 
                'mouse_id': '$lims_ophys_session.external_specimen_name', 
                'operator': '$lims_ophys_session.operator', 
                'rig': '$lims_ophys_session.rig', 
                'project': '$lims_ophys_session.project', 
                'date': '$lims_ophys_session.date_of_acquisition', 
                'qc_report_staus': '$qc_meta.qc_report_staus', 
                'stimulus': '$lims_ophys_session.stimulus_name'
            }
        }, {
            '$project': {
                'lims_id': 1, 
                'genotype': 1, 
                'mouse_id': 1, 
                'operator': 1, 
                'rig': 1, 
                'project': 1, 
                'date': 1, 
                'qc_report_staus': 1, 
                'stimulus': 1
            }
        }
    ])
    sessions_df = query_results_to_df(timed_records)
    sessions_df = sessions_df.rename(columns={"lims_id":"ophys_session_id", 
                                              "date": "date_time"})
    return sessions_df

def get_qc_status_for_sessions(sessions_list)-> pd.DataFrame:
    ses_qc_df = pd.DataFrame()
    for session_id in sessions_list:
        session_id = int(session_id)
        session_qc_status = report_submit_status.aggregate([{
            '$match': {
                'data_id': session_id, 
                'current': True
            }
        }, {
            '$project': {
                'status': 1, 
                'data_id': 1
            }
        }])
        
        temp_df = pd.DataFrame(list(session_qc_status))
        if temp_df.empty:
            tmp_dict = {"status": "report_not_generated",
                        "data_id": session_id}
            tmp_df = pd.DataFrame(tmp_dict, index=[0])
            ses_qc_df = ses_qc_df.append(tmp_df)
        else:
            temp_df = temp_df.drop(['_id'], axis=1)
            ses_qc_df = ses_qc_df.append(temp_df)
            
    ses_qc_df = ses_qc_df.reset_index(drop=True)

    ses_qc_df = ses_qc_df.rename(columns={"data_id": "ophys_session_id",
                                          "status": "qc_submit_status"})
    return ses_qc_df
    

#####################################################################
#
#          UTILITY FUNCTIONS
#
#####################################################################

def query_results_to_df(query_results)-> pd.DataFrame:
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


def get_date_from_dttime(datetime_obj):
    # use if loading directly from mongodb
    return datetime_obj.date() 


def generate_project_group(project_code:str):
    for key, value in PROJECT_GROUPS_DICT.items():
        if project_code in value:
            return key
    return "Cannot find project group for {}".format(project_code)


def generate_qc_review_status(qc_submit_status):
    for key, value in QC_SUBMIT_STATUS_DICT.items():
        if qc_submit_status in value:
            return key
    return "Cannot find review status for {}".format(qc_submit_status)


def clean_session_records_df(sessions_df):
    sessions_df['date'] = sessions_df['date_time'].apply(get_date_from_dttime)
    sessions_df["project_group"] = sessions_df["project"].apply(generate_project_group)
    return sessions_df
