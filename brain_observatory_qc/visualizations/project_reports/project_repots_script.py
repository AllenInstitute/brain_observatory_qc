import os
import time
import pymongo
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta,  timezone

import brain_observatory_qc.data_access.from_mouseQC as mqc
import brain_observatory_qc.visualizations.plotting_functions as qc_plots


#################
# CONNECT TO MONGO HOST 
#################
mongo_connection = mqc.connect_to_mouseqc_production()

qc_logs, metrics_records,\
images_records, report_generation_status,\
module_group_status, report_review_status = mqc.get_records_collections(mongo_connection)

metrics, controlled_language_tags = mqc.get_report_components_collections(mongo_connection)

#################
# GET EXPERIMENT & SESSION IDS IN DATE RANGE 
#################

# date range for report - defaults to 21 days from today
start_date, end_date = mqc.set_date_range()

# get sessions for date range
session_ids_df = mqc.get_session_ids_for_date_range(start_date, end_date)
session_ids_list = session_ids_df["ophys_session_id"].tolist()

# get all experiment ids for sessions within date range
exp_ids_df = mqc.get_experiment_ids_for_session_ids(session_ids_list)
exp_ids_list = exp_ids_df["ophys_experiment_id"].tolist()


#################
# GET TAGS 
#################
# SESSION tags: 
session_tags_df, sess_other_tags_df, sess_overrides_df = mqc.get_all_tags_for_ids(session_ids_list)
exp_tags_df, exp_other_tags_df, exp_overrides_df = mqc.get_all_tags_for_ids(exp_ids_list)

# EXPERIMENT tags: 