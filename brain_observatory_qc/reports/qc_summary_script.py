#################
# IMPORTS
#################

import os
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages



import brain_observatory_qc.data_access.from_mouseQC as mqc
from brain_observatory_qc.reports.pdf_utils import PdfReport
from brain_observatory_qc.data_access.utilities import correct_filepath
import brain_observatory_qc.visualizations.plotting_functions as qc_plots

#################
# SET CONSTANTS
#################
NUM_DAYS = 30 # Number of days to look back for data


CSV_SAVE_PATH  = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\csvs")
PLOT_SAVE_PATH = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\plots")
PDF_SAVE_PATH  = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\reports")

# #################
# SET UP PDF REPORT
# #################
report_name="Ophys QC Summary {}-{}.pdf".format(datetime.datetime.now().strftime("%Y-%m-%d"),   )
pdf = PdfReport("aind-ephys-rig-qc v" )


#################
# Get SESSIONS & EXPERIMENTS FOR DATE RANGE
#################

# Sessions & Experiments from last 21 days
sessions_df = mqc.gen_session_qc_info_for_date_range(range_in_days = NUM_DAYS,
                                                     csv_path      = CSV_SAVE_PATH)
__, experiment_ids_list = mqc.get_experiment_ids_for_session_ids(sessions_df["ophys_session_id"].tolist())
experiments_df = mqc.gen_experiment_qc_info_for_ids(experiment_ids_list,
                                                    csv_path = CSV_SAVE_PATH)

# latest session for all current mice
mouse_df = mqc.current_mice_df(sessions_df,
                               csv_path = CSV_SAVE_PATH)

#################
# PLOT QC STATUS FOR SESSIONS & EXPERIMENTS
#################
sess_qc_status = sessions_df[["ophys_session_id", "generation_status", "review_status", "qc_outcome"]]
exp_qc_status = experiments_df[["mouse_id","ophys_experiment_id", "generation_status", "review_status", "qc_outcome"]]

qc_plots.plot_qc_submit_status_matrix(sess_qc_status,
                                      "ophys_session_id",
                                      ylabel ="Session ID")

qc_plots.plot_qc_submit_status_matrix(exp_qc_status,
                                      "ophys_experiment_id",
                                      ylabel ="Experiment ID",
                                      show_labels=False)

# MOUSE SPECIFIC QC GENERATION & SUBMISSION STATUS
mouse_exp_qc_status = experiments_df[["mouse_id",
                                      "ophys_experiment_id",
                                      "generation_status",
                                      "review_status",
                                      "qc_outcome"]]

for mouse_id, group_df in mouse_exp_qc_status.groupby('mouse_id'):
    plot_title = "Mouse {} QC Generation & Submission Status".format(mouse_id)
    qc_plots.plot_qc_submit_status_matrix(group_df,
                                          "ophys_experiment_id",
                                          ylabel = "Experiment ID",
                                          title = plot_title,
                                          show_labels = False)

#################
# PLOT IMPACTED DATA FOR COMPLETED QC
#################
completed_sess_list = sess_qc_status.loc[sess_qc_status["review_status"]=="complete", 
                                         "ophys_session_id"].tolist()
completed_exp_list  = sess_qc_status.loc[sess_qc_status["review_status"]=="complete", 
                                         "ophys_session_id"].tolist()

qc_plots.plot_impacted_data_outcomes_matrix(completed_sess_list,
                                             "ophys_session_id",
                                             ylabel="Session ID")

qc_plots.plot_impacted_data_outcomes_matrix(completed_exp_list, 
                                             "ophys_experiment_id",
                                             ylabel="Experiment ID")
#################
# TAG FREQUENCY PLOTS
#################

# Get all tags for sessions & experiments
tags_df, other_tags_df, overrides_df = mqc.gen_tags_df_for_ids(completed_sess_list,
                                                               completed_exp_list)
qc_plots.plot_qc_tag_frequency(tags_df)


#################
# TAG FREQUENCY PLOTS
#################