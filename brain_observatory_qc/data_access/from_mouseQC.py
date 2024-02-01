import os
import time
import pymongo
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta,  timezone

### MONGO CONNECTION

username = 'public'
password = 'public_password'
mongo_host = 'qc-sys-db'
mongo_port = '27017'

connection_string = 'mongodb://{}:{}@{}:{}/'.format(username, password, mongo_host, int(mongo_port))
mongo_connection = MongoClient(connection_string)


### COLLECTION CONNECTIONS

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


####  GET RECORDS  ####
def session_records_within_dates( start_date, end_date):
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

def get_qc_status_for_sessions(sessions_list):
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
    



####  UTILITIES  ####
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

def query_results_to_df(query_results):
    df = pd.DataFrame(list(query_results))
    df = df.drop(['_id'], axis=1)
    return df

def set_timeframe(days_from_enddate = 21):
    # set today for saving files
    today = datetime.now(timezone.utc).replace(hour= 0, minute=0, second=0, microsecond=0)
    
    # create the date range to look up records
    end_date = datetime.now(timezone.utc) 
    end_date = end_date.replace(hour= 0, minute=0, second=0, microsecond=0)
    
    start_date = datetime.now(timezone.utc)-timedelta(days_from_enddate)
    start_date = start_date.replace(hour= 0, minute=0, second=0, microsecond=0)
    return today, end_date, start_date

