import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta,  timezone

import brain_observatory_qc.data_access.from_mouseQC as from_mouseQC

############################
#  Constants
############################
TIME_RANGE = 21 #3 weeks
save_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/'


############################
#   Set Time Range
############################
today = str(datetime.now().date())

# end date is today
end_date = datetime.now(timezone.utc) 
end_date = end_date.replace(hour= 0, minute=0, second=0, microsecond=0)

# start date is 3 weeks ago
start_date = datetime.now(timezone.utc)-timedelta(TIME_RANGE)
start_date = start_date.replace(hour= 0, minute=0, second=0, microsecond=0)


############################
#  Get Session & Experiment Report Info
############################

# Get session & experiment ids for date range
session_ids_df = from_mouseQC.get_session_ids_for_date_range(start_date, end_date)
session_ids_list = session_ids_df['ophys_session_id'].tolist()

experiment_ids_df = from_mouseQC.get_experiment_ids_for_session_ids(session_ids_list)
experiment_ids_list = from_mouseQC.experiment_ids_df['ophys_experiment_id'].tolist()

# get all report qc records
session_qc_info_df = from_mouseQC.gen_session_qc_info_for_ids(session_ids_list)
exp_qc_info_df = from_mouseQC.gen_experiment_qc_info(experiment_ids_list)

# save to csv
session_qc_info_df.to_csv(save_path + 'session_qc_report_info_{}.csv'.format(today), index=False)
exp_qc_info_df.to_csv(save_path + 'exp_qc_report_info_{}.csv'.format(today), index=False)

############################
#  Get Tag info for Completed Reports
############################

# get ids for exps and sessions with completed reports
completed_session_ids = session_qc_info_df[session_qc_info_df['review_status'] == 'complete']['ophys_session_id'].tolist()
completed_exp_ids = exp_qc_info_df[exp_qc_info_df['review_status'] == 'complete']['ophys_experiment_id'].tolist()

## SESSIONS
# get tags for completed reports
session_tags_df, session_other_tags_df, session_overrides_df  = from_mouseQC.get_all_tags_for_ids(completed_session_ids)

# rename data_id to ophys_session_id
session_tags_df.rename(columns={'data_id':'ophys_session_id'}, inplace=True)
session_other_tags_df.rename(columns={'data_id':'ophys_session_id'}, inplace=True)
session_overrides_df.rename(columns={'data_id':'ophys_session_id'}, inplace=True)

# save to csv
session_tags_df.to_csv(save_path + 'session_tags_{}.csv'.format(today), index=False)
session_other_tags_df.to_csv(save_path + 'session_other_tags_{}.csv'.format(today), index=False)
session_overrides_df.to_csv(save_path + 'session_overrides_{}.csv'.format(today), index=False)

## EXPERIMENTS
# get tags for completed reports
exp_tags_df, exp_other_tags_df, exp_overrides_df = from_mouseQC.get_all_tags_for_ids(completed_exp_ids)
# rename data_id to ophys_experiment_id
exp_tags_df.rename(columns={'data_id':'ophys_experiment_id'}, inplace=True)
exp_other_tags_df.rename(columns={'data_id':'ophys_experiment_id'}, inplace=True)
exp_overrides_df.rename(columns={'data_id':'ophys_experiment_id'}, inplace=True)

# save to csv
exp_tags_df.to_csv(save_path + 'exp_tags_{}.csv'.format(today), index=False)
exp_other_tags_df.to_csv(save_path + 'exp_other_tags_{}.csv'.format(today), index=False)
exp_overrides_df.to_csv(save_path + 'exp_overrides_{}.csv'.format(today), index=False)






