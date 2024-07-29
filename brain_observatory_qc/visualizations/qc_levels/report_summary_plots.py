import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt


TODAY = datetime.today().date() # today's date
pass_flag_fail_palette = {
    'pass'   : '#4CAF50', # Green
    'flag'   : '#FFC107', # Amber
    'fail'   : '#F44336', # Red
    'missing': '#D3D3D3', # Light gray
}

# Submission status palette
qc_gen_sub_status_palette = {
     "error"     :"#a6611a",  # brown
     "complete"  :"#940094",  # purple
     "incomplete":"#4a0094",  # blue
     "pass"      :"#8dd3c7",  # slightly desaturated cyan
     "flag"      :"#ffffb3",  # light yellow
     "fail"      :"#fb8072"   # light red
}

general_qc_status_palette = {
    'error'     :'#a6611a',  # brown
    'complete'  :'#940094',  # purple
    'incomplete':'#4a0094',  # blue
    'pass'   : '#4CAF50', # Green
    'flag'   : '#FFC107', # Amber
    'fail'   : '#F44336', # Red
    'missing': '#D3D3D3', # Light gray
}

######################################
#
#          GENERIC PLOTTING FUNCTIONS
######################################
def create_qc_tag_bar_plot(df:pd.DataFrame, 
                           ax:matplotlib.axes.Axes, 
                           title:str,
                           palette:dict=pass_flag_fail_palette,):
    """Create a qc tag frequency bar plot for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        qc tags df: table with the following columns:
            - data_id: int64
            - qc_tag: str
            - qc_outcome: str - flag or fail
            - impacted_data: str
            - report_type: str - session or experiment
    ax : matplotlib.axes.Axes
       Axes object to draw the plot on.
    title : str
        Title for the plot.
    """
    # Get frequency of each qc_tag
    freq_df = df['qc_tag'].value_counts().reset_index()
    freq_df.columns = ['qc_tag', 'frequency']

    # Merge frequency with the original DataFrame to get qc_outcome
    merged_df = pd.merge(freq_df, df[['qc_tag', 'qc_outcome']], on='qc_tag', how='left').drop_duplicates()

    # Order by frequency
    merged_df = merged_df.sort_values('frequency', ascending=False)

    if palette is None:
        palette = {'pass': '#4CAF50', 
                   'flag': '#FFC107',
                   'fail': '#F44336'}

    # Create a bar plot
    sns.barplot(x='frequency', y='qc_tag', hue='qc_outcome', data=merged_df,
                palette=palette, dodge=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('QC Tag')


def create_qc_status_bar_plot(df:pd.DataFrame,
                              status_col:str,
                              ax:matplotlib.axes.Axes,
                              title:str,
                              palette:dict = None):
    """Create a qc status frequency bar plot for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        qc status df: table with following structure:
            - session or experiment id
            - column with qc status such as: 'generation_status', 'review_status', 'qc_outcome'
    status_col : str
        name of the column that contains the status values
    ax : matplotlib.axes.Axes
        Axes object to draw the plot
    title : str
        title for the plot
    palette : dict, optional
        dictionary with following structure: {status_string: color, status_string: color},
        by default general_qc_status_palette
    """
    # Get frequency of each status
    freq_df = df[status_col].value_counts().reset_index()
    freq_df.columns = [status_col, 'frequency']

    # Order by frequency
    freq_df = freq_df.sort_values('frequency', ascending=False)

    # Create a bar plot
    sns.barplot(x=status_col,
                y='frequency',
                hue=status_col,
                data=freq_df,
                palette=palette,
                dodge=False,
                ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.legend_.remove()


def plot_status_by_id_matrix(df: pd.DataFrame,
                             id_col: str,
                             color_mapping: dict,
                             week_col: str = None,
                             save_path: str = None,
                             save_name: str = None,
                             xlabel: str = None,
                             ylabel: str = None,
                             title: str = None,
                             figsize: tuple = (10, 8),
                             show_labels: bool = True):
    """
    A generic function that will take a dataframe of a specific structure
    and create a colored grid plot based on the values in the dataframe.
    Y axis: ids 
    X axis: all other columns in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with a specific structure:
        - id_col: a column (e.g. ophys_session_id, ophys_experiment_id)
        - all remaining columns contain string values that are keys in the color_mapping dictionary.
          The strings represent the status of the given column.

        Example: df with columns: ophys_session_id, data_stream_1, data_stream_2, data_stream_3
        where all the string data_stream columns can be "pass", "flag", "fail"

    color_mapping : dict
        The color mapping dictionary that maps the string values in the dataframe to colors.
    week_col : str, optional
        Column name for weeks. If provided, session ids will be grouped by week.
    figsize : tuple, optional
        Size of the figure, by default (10,8).
    save_path : str, optional
        Location to save the plot, by default None.
    show_labels : bool, optional
        Whether to show labels on the patches, by default True.
    """
    # Define default color for NaN values
    default_color = color_mapping.get('missing', '#D3D3D3')  # Light gray as default

    # Exclude the week column from the plotted matrix
    if week_col:
        plot_df = df.drop(columns=[week_col])
    else:
        plot_df = df

    # Create a color matrix based on the DataFrame values
    color_matrix = plot_df.iloc[:, 1:].applymap(lambda x: color_mapping.get(x, default_color))

    # Plotting the colored grid
    fig, ax = plt.subplots(figsize=figsize)

    # If week_col is provided, group the session IDs by week
    if week_col:
        unique_weeks = df[week_col].unique()
        id_labels = []
        for week in unique_weeks:
            week_ids = df[df[week_col] == week][id_col].tolist()
            id_labels.extend([f"{id} ({week})" for id in week_ids])
    else:
        id_labels = df[id_col].tolist()

    # Create a grid of colored boxes
    for (row_idx, col_idx), val in np.ndenumerate(plot_df.iloc[:, 1:].values):
        color = color_mapping.get(val, default_color)
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, color=color))
        if show_labels:
            ax.text(col_idx + 0.5, row_idx + 0.5, str(val), ha='center', va='center', color='black')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(plot_df.columns[1:])) + 0.5)
    ax.set_yticks(np.arange(len(plot_df)) + 0.5)
    ax.set_xticklabels(plot_df.columns[1:], rotation=90)
    ax.set_yticklabels(id_labels)

    # Add gridlines
    ax.hlines(np.arange(len(plot_df) + 1), *ax.get_xlim(), color='gray')
    ax.vlines(np.arange(len(plot_df.columns[1:]) + 1), *ax.get_ylim(), color='gray')

    # Remove the default spines
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add axis labels if provided
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add title if provided
    if title:
        ax.set_title(title)

    # Create a custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[key]) for key in color_mapping.keys()]
    labels = list(color_mapping.keys())
    ax.legend(handles, labels, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path and save_name:
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file)
        print(f'Saved plot to {save_file}')
 
    plt.show()


######################################
#
#          MAIN PLOTTING FUNCTIONS
######################################
def plot_qc_tag_frequency(df,
                          save_path:str=None,
                          save_name:str="qc_tag_frequency_{}.png".format(TODAY),
                          palette:dict=pass_flag_fail_palette):
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        qc_tags_df: table with the following columns:
            - data_id: int64
            - qc_tag: str
            - qc_outcome: str - flag or fail
            - impacted_data: str
            - report_type: str - session or experiment
    save_path : str, optional
        Directory path to save the plot, by default None
    save_name : str, optional
        _description_, by default None
    palette : dict, optional
        dictionary with following structure: {status_string: color, status_string: color},
        by default pass_flag_fail_palette
    """
    # Split the DataFrame by report_type
    session_df = df[df['report_type'] == 'session']
    experiment_df = df[df['report_type'] == 'experiment']

    # Plot settings
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # Create plots for session and experiment data
    create_qc_tag_bar_plot(session_df, axs[0], 'Session QC Tag Frequency', palette=palette)
    create_qc_tag_bar_plot(experiment_df, axs[1], 'Experiment QC Tag Frequency', palette=palette)

    plt.tight_layout()

    # Save the plot if a save path and name are provided
    if save_path and save_name:
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file)
        print(f'Saved plot to {save_file}')

    plt.show()

def plot_qc_status_frequency(df,
                             save_path:str=None,
                             save_name:str="qc_status_frequency_{}.png".format(TODAY),
                             palette:dict=general_qc_status_palette,
                             title:str=None):
    """

    Parameters
    ----------
    df : pd.DataFrame
        qc status df: table with following structure:
            - session or experiment id
            - 'generation_status' 
            - 'review_status'
            - 'qc_outcome'
    save_path : str, optional
        Directory path to save the plot, by default None
    save_name : str, optional
        _description_, by default None
    palette : dict, optional
        dictionary with following structure: {status_string: color, status_string: color},
        by default general_qc_status_palette
    """

    # Plot settings
    fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=False)

    # Create plots for session and experiment data
    create_qc_status_bar_plot(df, "generation_status", axs[0], 'Generation Status Frequency', palette=palette)
    create_qc_status_bar_plot(df, "review_status", axs[1], 'Review Status Frequency', palette=palette)
    create_qc_status_bar_plot(df, "qc_outcome", axs[2], 'QC Outcome Frequency', palette=palette)
    
    # add figure title 
    if title is None:
        data_type = identify_experiment_or_session(df)
        title = '{} qc status frequency'.format(data_type)
    
    fig.suptitle(title, fontsize=16)


    plt.tight_layout()

    # Save the plot if a save path and name are provided
    if save_path and save_name:
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file)
        print(f'Saved plot to {save_file}')

    plt.show()



def plot_impacted_data_outcomes_matrix(data_stream_outcomes_df: pd.DataFrame, 
                                       id_col: str = "data_id",
                                       week_col: str = None,
                                       color_mapping:dict=pass_flag_fail_palette,
                                       save_path: str = None,
                                       save_name: str = "impacted_data_qc_outcomes_matrix_{}.png".format(TODAY),
                                       xlabel:str="Impacted Data",
                                       ylabel:str="Data ID",
                                       title:str="Impacted Data QC Outcomes",
                                       show_labels: bool = True):
    """Plot a matrix of the QC outcomes for each impacted data stream or context.
    Input dataframe should be limited to sessions/exeriments that have COMPLETED
    QC Generation and Review.

    Parameters
    ----------
    data_stream_outcomes_df : pd.DataFrame
        table with a column for data_id and columns for each impacted
        data (data streams, data context) with values of "pass", "flag", "fail"
    id_col : str, optional
        name of column that has the data id (i.e. "ophys_session_id", etc),
        by default "data_id"
    color_mapping : dict, optional
        keys are data outcomes like "pass", "flag" etc. 
        values are colors for those outcomes, 
        by default pass_flag_fail_palette
    save_path : str, optional
        location to save the plot, by default None
    xlabel : str, optional
        label for x axis of plot, by default "Impacted Data"
    ylabel : str, optional
        label for y axis of plot, by default "Data ID"
    title : str, optional
        plot title, by default "Impacted Data QC Outcomes"
    show_labels : bool, optional
        whether to show labels on the patches, by default True
    """
    plot_status_by_id_matrix(data_stream_outcomes_df,
                             id_col=id_col,
                             week_col = week_col,
                             color_mapping=color_mapping,
                             save_path=save_path,
                             save_name=save_name,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=title,
                             show_labels=show_labels)


def plot_qc_submit_status_matrix(submit_status_df:pd.DataFrame, 
                                id_col: str = "data_id",
                                week_col: str = None,
                                color_mapping:dict=qc_gen_sub_status_palette,
                                save_path: str = None,
                                save_name: str = "qc_submit_status_matrix_{}.png".format(TODAY),
                                xlabel:str="QC Status",
                                ylabel:str="Data ID",
                                title:str="QC Generation & Submission Status",
                                show_labels: bool = True):
    """ Plot a matrix of the QC generation and submission status for each
     dataset (session, experiment etc.)

    Parameters
    ----------
    submit_status_df : pd.DataFrame
        table with the following columns:
            - generation_status
            - review_status
            - qc_outcome
    id_col : str, optional
        name of column for data unit, examples: "ophys_session_id", by default "data_id"
    color_mapping : dict, optional
        keys are data outcomes like "pass", "flag" etc. 
        values are colors for those outcomes, 
        by default pass_flag_fail_palette
    save_path : str, optional
        location to save the plot, by default None
    xlabel : str, optional
        label for x axis of plot, by default "QC Status"
    ylabel : str, optional
        label for y axis of plot, by default "Data ID"
    title : str, optional
        plot title, by default "QC Generation & Submission Status"
    show_labels : bool, optional
        whether to show labels on the patches, by default True
    """
    plot_status_by_id_matrix(submit_status_df,
                            id_col=id_col,
                            week_col=week_col,
                            color_mapping=color_mapping,
                            save_path=save_path,
                            save_name=save_name,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            title=title,
                            show_labels=show_labels)


######################################
#
#          UTILITY FUNCTIONS
######################################

def identify_experiment_or_session(df:pd.DataFrame, id_column:str=None)->str:
    """Identify if the id column name contains the words 'session' or 'experiment'.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with an id column with the type of id in the column name
    id_column : str, optional
        name of the id column, by default None (will use the first column in the dataframe)

    Returns
    -------
    str
        empty string if the id column does not contain 'session' or 'experiment'
        otherwise returns 'session' or 'experiment'

    Raises
    ------
    ValueError
        if the provided id_column is not a string
    ValueError
        if the provided id_column does not exist in the DataFrame
    """

    # Ensure the id_column is a string
    if id_column is not None and not isinstance(id_column, str):
        raise ValueError("The id_column parameter must be a string.")
    
    if id_column is None:
        id_column = df.columns[0]
    
    # Ensure the id_column exists in the DataFrame
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' does not exist in the DataFrame.")
    
    # Check if the column name contains 'session' or 'experiment'
    result = ''
    if 'experiment' in id_column.lower():
        result = 'experiment'
    elif 'session' in id_column.lower():
        result = 'session'
    
    return result