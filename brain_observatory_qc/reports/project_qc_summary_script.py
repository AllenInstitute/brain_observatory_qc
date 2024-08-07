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
all_sessions = mqc.gen_session_qc_info_for_date_range(range_in_days=NUM_DAYS, csv_path=CSV_SAVE_PATH)

# Filter to just the production project groups
production_sessions = all_sessions[all_sessions['project_group'].isin(PRODUCTION_PROJECT_GROUPS)]

for project_group, group_df in production_sessions.groupby('project_group'):
    print ('currently on {}'.format(project_group))
    
    # Create an instance of PdfReport
    report_title = "Ophys QC Status Summary Report: {}".format(project_group)
    pdf_filename = "Ophys_QC_Summary_{}.pdf".format(project_group)
    pdf = PdfReport(title=report_title)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Header
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, report_title, ln=True, align="C")
    
    ################
    # MOUSE INFORMATION
    ################

    # Latest session for all current mice
    mouse_df = mqc.current_mice_df(group_df, csv_path=CSV_SAVE_PATH)
   
    # Embed the mouse DataFrame in the PDF
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "{} Mouse Information".format(project_group), ln=True, align="C")
    pdf.ln(10)  # Add some space before the table
    pdf.set_font("helvetica", "", 10)
    pdf.embed_table(mouse_df)

   
    ################
    # SESSION PLOTS
    ################
     # Subset to just relevant columns
    sess_plt_df = group_df[["ophys_session_id", "date", "week", "generation_status", "review_status", "qc_outcome"]]


    # Plot QC status for each session
    plot_save_name = "{}_ophys_session_qc_status_{}-{}.png".format(project_group, STRT_DATE, TODAY)
    plot_title = "{} Session QC Generation & Submission Status".format(project_group)
    qc_plots.plot_qc_generation_status(sess_plt_df,
                                       save_name=plot_save_name,
                                       save_path=PLOT_SAVE_PATH)

    # Embed the session QC plot in the PDF
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, plot_title, ln=True, align="C")
    pdf.image(os.path.join(PLOT_SAVE_PATH, plot_save_name), w=190)

    # Plot impacted data outcomes for each session
    plot_save_name = "{}_ophys_session_impacted_data_outcomes_{}-{}.png".format(project_group, STRT_DATE, TODAY)
    plot_title = "{} Session impacted data outcomes".format(project_group)
    
    completed_sess_list = sess_plt_df.loc[sess_plt_df["review_status"] == "complete", "ophys_session_id"].tolist()
    sess_outcomes_df = mqc.gen_session_impacted_data_outcome_df(completed_sess_list)
    
    qc_plots.plot_impacted_data_outcomes_matrix(sess_outcomes_df,
                                                id_col="ophys_session_id",
                                                title=plot_title,
                                                save_name=plot_save_name,
                                                save_path=PLOT_SAVE_PATH,
                                                ylabel="Session ID",
                                                show_labels=False)

    # Embed the impacted data outcomes plot in the PDF
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, plot_title, ln=True, align="C")
    pdf.image(os.path.join(PLOT_SAVE_PATH, plot_save_name), w=190)

    
    ################
    # EXPERIMENT PLOTS
    ################

    # Get experiments for sessions
    _, experiment_ids_list = mqc.get_experiment_ids_for_session_ids(group_df["ophys_session_id"].tolist())
    exp_info = mqc.gen_experiment_qc_info_for_ids(experiment_ids_list)

    # Subset to just relevant columns
    exp_plot_df = exp_info[["ophys_experiment_id", "date", "week", "generation_status", "review_status", "qc_outcome"]]

    # Plot QC status for each experiment
    plot_save_name = "ophys_experiment_qc_status{}_{}-{}.png".format(project_group, STRT_DATE, TODAY)
    plot_title = "{} Experiment QC Generation & Submission Status".format(project_group)
    qc_plots.plot_qc_generation_status(exp_plot_df,
                                       height_ratios = (30,4),
                                       save_name=plot_save_name,
                                       save_path=PLOT_SAVE_PATH)

    # Embed the experiment QC plot in the PDF
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, plot_title, ln=True, align="C")
    pdf.image(os.path.join(PLOT_SAVE_PATH, plot_save_name), w=190)

    ### IMPACTED DATA
    # plot inputs
    plot_save_name = "{}_ophys_experiment_impacted_data_outcomes_{}-{}.png".format(project_group, STRT_DATE, TODAY)
    plot_title = "{} Experiment impacted data outcomes".format(project_group)
    
    # data to plot
    completed_exp_list = exp_plot_df.loc[exp_plot_df["review_status"] == "complete", "ophys_experiment_id"].tolist()
    exp_outcomes_df = mqc.gen_experiment_impacted_data_outcome_df(completed_exp_list)
    
    # plot
    qc_plots.plot_impacted_data_outcomes_matrix(exp_outcomes_df,
                                                id_col="ophys_experiment_id",
                                                title=plot_title,
                                                save_name=plot_save_name,
                                                save_path=PLOT_SAVE_PATH,
                                                ylabel="Experiment ID",
                                                show_labels=False)

    # Embed the impacted data outcomes plot in the PDF
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, plot_title, ln=True, align="C")
    pdf.image(os.path.join(PLOT_SAVE_PATH, plot_save_name), w=190)

    #################
    # TAG FREQUENCY PLOTS
    #################

    # Get all tags for sessions & experiments
    all_tags_df, all_other_tags_df, all_overrides_df = mqc.gen_tags_df_for_ids(completed_sess_list, completed_exp_list)
    plot_save_name = "{}_tag_frequency_{}-{}.png".format(project_group, STRT_DATE, TODAY)
    qc_plots.plot_qc_tag_frequency(all_tags_df, save_path=PLOT_SAVE_PATH, save_name=plot_save_name)

    # Embed the tag frequency plot in the PDF
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Tag Frequency", ln=True, align="C")
    pdf.image(os.path.join(PLOT_SAVE_PATH, plot_save_name), w=190)

    
    # Save the PDF report
    try:
        save_path = os.path.join(PDF_SAVE_PATH, pdf_filename)
        pdf.output(save_path)
        print(f"PDF report saved to {save_path}")
    except PermissionError as e:
        print(f"Permission error: {e}")
        # Try saving with a different name
        alt_save_path = os.path.join(PDF_SAVE_PATH, "blank_pdf_alternate.pdf")
        try:
            pdf.output(alt_save_path)
            print(f"PDF report saved to {alt_save_path}")
        except PermissionError as e:
            print(f"Failed to save the PDF report: {e}")

