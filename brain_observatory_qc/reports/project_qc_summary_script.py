#################
# IMPORTS
#################

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages



import brain_observatory_qc.data_access.from_mouseQC as mqc
from brain_observatory_qc.reports.pdf_utils import PdfReport
from brain_observatory_qc.data_access.utilities import correct_filepath
import brain_observatory_qc.visualizations.plotting_functions as qc_plots

#################
# SET CONSTANTS
#################
NUM_DAYS = 30 # Number of days to look back for data
TODAY = datetime.today().date() # today's date
STRT_DATE = TODAY - timedelta(days=NUM_DAYS)

CSV_SAVE_PATH  = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\csvs")
PLOT_SAVE_PATH = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\plots")
PDF_SAVE_PATH  = correct_filepath("\\allen\programs\mindscope\workgroups\learning\mouse-qc\reports")

PRODUCTION_PROJECT_GROUPS = ["omFISH_V1_U01", "learning_mfish"]

# #################
# SET UP PDF REPORT
# #################
report_name="Ophys QC Summary {}-{}.pdf".format(STRT_DATE, TODAY)
pdf = PdfReport(report_name)


#################
# Get SESSIONS & EXPERIMENTS FOR DATE RANGE
#################

# Get all Sessions 
all_sessions = mqc.gen_session_qc_info_for_date_range(range_in_days = NUM_DAYS,
                                                      csv_path = CSV_SAVE_PATH)
# latest session for all current mice
mouse_df = mqc.current_mice_df(all_sessions,
                               csv_path = CSV_SAVE_PATH)

# Filter to just the production project groups
production_sessions = all_sessions[all_sessions['project_group'].isin(PRODUCTION_PROJECT_GROUPS)]

##################################
# 
#  PLOT QC STATUS FOR SESSIONS & EXPERIMENTS
#
##################################
# SESSIONS 
# group by project
for project_group, group_df in production_sessions.groupby('project_group'):
    group_df = group_df[["ophys_session_id", "week", "generation_status", "review_status", "qc_outcome"]]
    
    ################
    # SESSION PLOTS
    ################

    # plot qc status for each session
    plot_save_name = "{}_ophys_session_qc_status_matrix_{}-{}.png".format(project_group,
                                                                          STRT_DATE,
                                                                          TODAY)
    plot_title = "{} Session QC Generation & Submission Status".format(project_group)
    qc_plots.plot_qc_submit_status_matrix(group_df,
                                          id_col = "ophys_session_id",
                                          week_col = "week",
                                          ylabel = "Session ID",
                                          title = plot_title,
                                          show_labels = False,
                                          save_name = plot_save_name,
                                          save_path = PLOT_SAVE_PATH)
    
    
    # plot qc status frequency
    plot_save_name = "{}_ophys_session_qc_status_frequency_{}-{}.png".format(project_group,
                                                                             STRT_DATE,
                                                                             TODAY)                                                                     
    qc_plots.plot_qc_status_frequency(group_df,
                                      save_name=plot_save_name,
                                      save_path=PLOT_SAVE_PATH)

    # plot impacted data outcomes for each session
    completed_sess_list = group_df.loc[group_df["review_status"]=="complete",
                                       "ophys_session_id"].tolist()
    
    plot_save_name = "{}_ophys_session_impacted_data_outcomes_{}-{}.png".format(project_group,
                                                                                STRT_DATE,
                                                                                TODAY)
    data_outcomes_df = mqc.gen_session_impacted_data_outcome_df(completed_sess_list)
    qc_plots.plot_impacted_data_outcomes_matrix(data_outcomes_df,
                                             id_col = "ophys_session_id",
                                             week_col = "week",
                                             save_name = plot_save_name,
                                             save_path = PLOT_SAVE_PATH,
                                             ylabel="Session ID")
    
    ################
    # MOUSE PLOTS
    ################

    # latest session for all current mice
    mouse_df = mqc.current_mice_df(group_df,
                                   csv_path = CSV_SAVE_PATH)
    ################
    # EXPERIMENT PLOTS
    ################

    # get experiments for sessions
    __, experiment_ids_list = mqc.get_experiment_ids_for_session_ids(group_df["ophys_session_id"].tolist())
    exp_info = mqc.gen_experiment_qc_info_for_ids(experiment_ids_list)
    plot_df = exp_info[["ophys_session_id", "week", "generation_status", "review_status", "qc_outcome"]]


    # plot qc status for each experiment
    plot_save_name = "ophys_experiment_qc_status_matrix_{}-{}.png".format(STRT_DATE, TODAY)
    plot_title = "{} Experiment QC Generation & Submission Status".format(project_group)
    qc_plots.plot_qc_submit_status_matrix(plot_df,
                                          id_col = "ophys_experiment_id",
                                          week_col = "week",
                                          save_name = plot_save_name,
                                          save_path = PLOT_SAVE_PATH,
                                          ylabel = "Experiment ID",
                                          title = plot_title,
                                          show_labels = False)
    
    # plot qc status frequency
    plot_save_name = "{}_ophys_experiment_qc_status_frequency_{}-{}.png".format(project_group,
                                                                                STRT_DATE,
                                                                                TODAY)                                                                       
    qc_plots.plot_qc_status_frequency(plot_df,
                                      save_name=plot_save_name,
                                      save_path=PLOT_SAVE_PATH)
    
    # plot impacted data for each experiment
    completed_exp_list = group_df.loc[plot_df["review_status"]=="complete",
                                       "ophys_experiment_id"].tolist()
    
    plot_save_name = "{}_ophys_experiment_impacted_data_outcomes_{}-{}.png".format(project_group,
                                                                                   STRT_DATE,
                                                                                   TODAY)
    
    data_outcomes_df = mqc.gen_experiment_impacted_data_outcome_df(completed_exp_list)
    qc_plots.plot_impacted_data_outcomes_matrix(data_outcomes_df,
                                             "ophys_experiment_id",
                                             ylabel="Experiment ID")
    

    #################
    # TAG FREQUENCY PLOTS
    #################

    # Get all tags for sessions & experiments
    tags_df, other_tags_df, overrides_df = mqc.gen_tags_df_for_ids(completed_sess_list,
                                                                   completed_exp_list)
    qc_plots.plot_qc_tag_frequency(tags_df,
                                   save_path = PLOT_SAVE_PATH,
                                   save_name = "{}_tag_frequency_{}-{}.png".format(project_group,
                                                                                STRT_DATE,
                                                                                TODAY))
    

