import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime, timedelta




import from_mouseQC as mqc
from pdf_utils import PdfReport
from utilities import correct_filepath
import report_summary_plots as qc_plots


#################
# SET CONSTANTS
#################

TODAY = datetime.today().date() # today's date


PRODUCTION_PROJECT_GROUPS = ["omFISH_V1_U01", "learning_mfish"]


def generate_project_report(project_group:str,
                            num_days:int = 15,
                            csv_save_path:str,
                            plot_save_path:str,
                            pdf_save_path:str,
                            report_name:str = None):
    """_summary_

    Parameters
    ----------
    project_group : str
        The name of the project group, for example "omFISH_V1_U01" or "learning_mfish"
    csv_save_path : str
       location to save the csv files
    plot_save_path : str
        location to save the plot files
    pdf_save_path : str
        location to save the pdf files
    report_name : str
        The name of the report, if none is provide it will be named 
        Ophys_QC_Summary_project_group_startdate-enddate.pdf
    num_days : int, optional
        The number of days to look back for data, by default 15
    """
    # set date range
    STRT_DATE = TODAY - timedelta(days=num_days)
    
    # set report name
    if report_name is None:
        report_name = "Ophys_QC_Summary_{}_{}-{}.pdf".format(project_group, STRT_DATE, TODAY)

    # set up pdf report    
    pdf = PdfReport(report_name)
    
    all_sessions = mqc.gen_session_qc_info_for_date_range(range_in_days=num_days,
                                                                 csv_path=csv_save_path)
    mouse_df = mqc.current_mice_df(all_sessions,
                                          csv_path=csv_save_path)
        
    production_sessions = all_sessions[all_sessions['project_group'].isin(PRODUCTION_PROJECT_GROUPS)]
        
    for project_group, group_df in production_sessions.groupby('project_group'):
        group_df = group_df[["ophys_session_id", "week", "generation_status", "review_status", "qc_outcome"]]
        
        qc_plots.plot_qc_submit_status_matrix(group_df,
                                              "ophys_session_id",
                                              ylabel="Session ID")
        
        pdf.add_page()
        pdf.set_font("Helvetica", "B", size=12)
        pdf.set_y(30)
        pdf.write(h=12, text=f"Overview of {project_group} QC")
        pdf.set_y(40)
        pdf.set_font("Helvetica", size=10)
        pdf.write(h=8, text=f"Date range: {start_date} to {today}")
        
        pdf.set_font("Helvetica", "", size=10)
        pdf.set_y(60)
        pdf.embed_table(group_df, width=pdf.epw)
        
    pdf.output(os.path.join(pdf_save_path, report_name))
    print("Finished.")
        
):





def generate_qc_report(
    directory,
    report_name="QC.pdf",
    timestamp_alignment_method="local",
    original_timestamp_filename="original_timestamps.npy",
    num_chunks=3,
    plot_drift_map=True,
    flip_NIDAQ=False,
):
    """
    Generates a PDF report from an Open Ephys data directory

    Saves QC.pdf

    Parameters
    ----------
    directory : str
        The path to the Open Ephys data directory
    report_name : str
        The name of the PDF report
    timestamp_alignment_method : str
        The type of alignment to perform
        Option 1: 'local' (default)
        Option 2: 'harp' (extract Harp timestamps from the NIDAQ stream)
        Option 3: 'none' (don't align timestamps)
    original_timestamp_filename : str
        The name of the file for archiving the original timestamps
    num_chunks : int
        The number of chunks to split the data into for plotting raw data
        and PSD
    plot_drift_map : bool
        Whether to plot the drift map

    """

    output_stream = io.StringIO()
    sys.stdout = output_stream

    pdf = PdfReport("aind-ephys-rig-qc v" + package_version)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", size=12)
    pdf.set_y(30)
    pdf.write(h=12, text="Overview of recordings in:")
    pdf.set_y(40)
    pdf.set_font("Helvetica", size=10)
    pdf.write(h=8, text=f"{directory}")

    pdf.set_font("Helvetica", "", size=10)
    pdf.set_y(60)
    stream_info = get_stream_info(directory)
    pdf.embed_table(stream_info, width=pdf.epw)

    if (
        timestamp_alignment_method == "local"
        or timestamp_alignment_method == "harp"
    ):
        # perform local alignment first in either case
        print("Aligning timestamps to local clock...")
        align_timestamps(
            directory,
            original_timestamp_filename=original_timestamp_filename,
            flip_NIDAQ=flip_NIDAQ,
            pdf=pdf,
        )

        if timestamp_alignment_method == "harp":
            # optionally align to Harp timestamps
            print("Aligning timestamps to Harp clock...")
            align_timestamps_harp(
                directory, pdf=pdf,
            )

    print("Creating QC plots...")
    create_qc_plots(
        pdf, directory, num_chunks=num_chunks, plot_drift_map=plot_drift_map
    )

    print("Saving QC report...")
    pdf.output(os.path.join(directory, report_name))
    print("Finished.")
    output_content = output_stream.getvalue()

    outfile = os.path.join(directory, "ephys-rig-QC_output.txt")

    with open(outfile, "a") as output_file:
        output_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        output_file.write(output_content)