import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.patches as patches

TODAY = datetime.today().date() # today's date


standard_figure_sizes = {
    'A4': (8.27, 11.69),
    'A5': (5.83, 8.27),
    'A6': (4.13, 5.83)}


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
    'complete'  :'#CBC3E3',  # light purple
    'incomplete':'#4a0094',  # blue
    'pass'   : '#4CAF50', # Greens
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
                           title:str=None,
                           palette:dict=pass_flag_fail_palette):
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
    legend : bool, optional
        Whether to show the legend, by default False
    palette : dict, optional
        dictionary with following structure: {status_string: color, status_string: color},
        by default pass_flag_fail_palette
    """
    # Get frequency of each qc_tag
    freq_df = df['qc_tag'].value_counts().reset_index()
    freq_df.columns = ['qc_tag', 'frequency']

    # Merge frequency with the original DataFrame to get qc_outcome
    merged_df = pd.merge(freq_df, df[['qc_tag', 'qc_outcome']], on='qc_tag', how='left').drop_duplicates()

    # Order by frequency
    merged_df = merged_df.sort_values('frequency', ascending=False)

    # Create a bar plot
    sns.barplot(x='qc_tag', y='frequency', hue='qc_outcome', data=merged_df,
                palette=palette, dodge=False, ax=ax)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('QC Tag')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
    ax.set_ylabel('Frequency')
    
    
    # remove legend because it is redundant
    ax.legend_.remove()
    plt.tight_layout()


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
    # Set title and labels
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xlabel('Status')
    
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # remove legend because it is redundant
    ax.legend_.remove()
    plt.tight_layout()


def plot_status_by_id_matrix(df: pd.DataFrame,
                             id_col: str,
                             color_mapping: dict,
                             week_col: str = None,
                             save_path: str = None,
                             save_name: str = None,
                             xlabel: str = None,
                             ylabel: str = None,
                             title: str = None,
                             figsize: tuple = None,
                             show_labels: bool = True,
                             line_thickness: float = 0.5):
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
    id_col : str
        The column name for the ids (e.g. ophys_session_id, ophys_experiment_id)
    color_mapping : dict
        The color mapping dictionary that maps the string values in the dataframe to colors.
    week_col : str, optional
        Column name for weeks. If provided, session ids will be grouped by week.
    save_path : str, optional
        Location to save the plot, by default None.
    save_name : str, optional
        Name of the saved plot, by default None.
    xlabel : str, optional
        Label for the x-axis, by default None.
    ylabel : str, optional
        Label for the y-axis, by default None.
    title : str, optional
        Title of the plot, by default None.
    figsize : tuple, optional
        Size of the figure, by default (10,8).
    show_labels : bool, optional
        Whether to show labels on the patches, by default True.
    line_thickness : float, optional
        Thickness of the grid lines, by default 0.5.
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
    figure, axes = plt.subplots()
    if figsize:
        figure.set_size_inches(figsize)

    # If week_col is provided, group the session IDs by week
    if week_col:
        unique_weeks = df[week_col].unique()
        id_labels = []
        for week in unique_weeks:
            week_ids = df[df[week_col] == week][id_col].tolist()
            id_labels.extend([f"{session_id} ({week})" for session_id in week_ids])
    else:
        id_labels = df[id_col].tolist()

    # Create a grid of colored boxes
    for (row_index, col_index), value in np.ndenumerate(plot_df.iloc[:, 1:].values):
        color = color_mapping.get(value, default_color)
        axes.add_patch(plt.Rectangle((col_index, row_index), 1, 1, color=color))
        if show_labels:
            axes.text(col_index + 0.5, row_index + 0.5, str(value), ha='center', va='center', color='black')

    # Set ticks and labels
    axes.set_xticks(np.arange(len(plot_df.columns[1:])) + 0.5)
    axes.set_yticks(np.arange(len(plot_df)) + 0.5)
    axes.set_xticklabels(plot_df.columns[1:], rotation=90)
    axes.set_yticklabels(id_labels)

    # Add gridlines with specified line thickness
    axes.hlines(np.arange(len(plot_df) + 1), *axes.get_xlim(), color='gray', linewidth=line_thickness)
    axes.vlines(np.arange(len(plot_df.columns[1:]) + 1), *axes.get_ylim(), color='gray', linewidth=line_thickness)

    # Remove the default spines
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['right'].set_visible(False)

    # Add axis labels if provided
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)

    # Add title if provided
    if title:
        axes.set_title(title)

    # Create a custom legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[key]) for key in color_mapping.keys()]
    legend_labels = list(color_mapping.keys())
    axes.legend(legend_handles, legend_labels, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save the plot if a save path and name are provided
    if save_path and save_name:
        save_file_path = os.path.join(save_path, save_name)
        plt.savefig(save_file_path)
        print(f'Saved plot to {save_file_path}')

    plt.show()


def plot_colored_qc_generation_status_table(df: pd.DataFrame, 
                                            palette: dict = general_qc_status_palette, 
                                            border_thickness: float = 2.0,
                                            ax=None):
    """
    Displays a DataFrame as a table with colored cells based on the cell values and draws thicker
    boundaries around entries within the same week.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be displayed as a table.
    palette : dict
        A dictionary mapping cell values to colors.
        by default qc_gen_sub_status_palette
    border_thickness : float, optional
        The thickness of the boundaries between weeks, by default 2.0.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    """
    # Sort the DataFrame by the "date" column
    df = df.sort_values(by="date")
    
    # Extract the "week" column and exclude it from the final displayed columns
    week_col = df['week']
    display_df = df.drop(columns=['week'])

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots()

    ax.axis('tight')
    ax.axis('off')
    tbl = Table(ax, bbox=[0, 0, 1, 1])

    # Adding table headers
    for col_idx, col_name in enumerate(display_df.columns):
        tbl.add_cell(0, col_idx, width=0.2, height=0.2, text=col_name, loc='center', facecolor='lightgrey')

    # Adding data to the table with colored cells based on status
    previous_week = None
    num_rows = display_df.shape[0]
    for row_idx, (row_label, row) in enumerate(display_df.iterrows()):
        current_week = week_col.iloc[row_idx]
        for col_idx, val in enumerate(row):
            color = palette.get(val, 'white')  # Default to white if status is not in palette
            tbl.add_cell(row_idx + 1, col_idx, width=0.2, height=0.2, text=val, loc='center', facecolor=color)

        # Draw thicker boundaries between different weeks
        if previous_week is not None and current_week != previous_week and row_idx < num_rows:
            for col_idx in range(len(display_df.columns)):
                if (row_idx + 1, col_idx) in tbl._cells:
                    tbl[(row_idx + 1, col_idx)].visible_edges = "TBLR"
                    tbl[(row_idx + 1, col_idx)].set_linewidth(border_thickness)
        previous_week = current_week

    ax.add_table(tbl)
    plt.tight_layout()

    if ax is None:
        plt.show()


def plot_qc_submit_status_matrix(submit_status_df:pd.DataFrame, 
                                id_col: str = "data_id",
                                week_col: str = None,
                                color_mapping:dict=general_qc_status_palette,
                                save_path: str = None,
                                save_name: str = "qc_submit_status_matrix_{}.png".format(TODAY),
                                xlabel:str="QC Status",
                                ylabel:str="Data ID",
                                title:str="QC Generation & Submission Status",
                                show_labels: bool = True,
                                figsize: tuple = None):
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
    figsize : tuple, optional
        size of the figure, by default (6, 5)
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
                            show_labels=show_labels,
                            figsize=figsize)


def plot_qc_status_frequency(df: pd.DataFrame,
                             save_path:str=None,
                             save_name:str="qc_status_frequency_{}.png".format(TODAY),
                             palette: dict = general_qc_status_palette,
                             title: str = 'QC Status Frequency',
                             figsize: tuple = None):  
    """
    Plots the frequency of QC statuses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be displayed.
        dataframe with the following columns:
            - session or experiment id
            - 'generation_status',
            - 'review_status',
            - 'qc_outcome'
    save_path : str, optional
        Directory path to save the plot, by default None.
    save_name : str, optional
        File name to save the plot, by default "qc_status_frequency_{}.png".format(TODAY).
    palette : dict, optional
        Dictionary mapping cell values to colors, by default general_qc_status_palette.
    title : str, optional
        Title of the plot, by default 'QC Status Frequency'.
    figsize : tuple, optional
        Size of the figure, by default (4.13, 5.83) for A6 paper size (1/4th of a standard printer paper).
    """
    # Plot settings
    fig, ax = plt.subplots(1, 3)
    if figsize:
        fig.set_size_inches(figsize)

    fig.suptitle(title)

    # Create plots for session and experiment data
    create_qc_status_bar_plot(df, "generation_status", ax[0], 'Report Generation', palette=palette)
    create_qc_status_bar_plot(df, "review_status", ax[1], 'Report Review', palette=palette)
    create_qc_status_bar_plot(df, "qc_outcome", ax[2], 'QC Outcome', palette=palette)

    # Create custom legend
    handles = [patches.Patch(color=color, label=label) for label, color in palette.items()]
    fig.legend(handles=handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend

    # Save the plot if a save path and name are provided
    if save_path and save_name:
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file)
        print(f'Saved plot to {save_file}')

    plt.show()

######################################
#
#          MAIN PLOTTING FUNCTIONS
######################################
def plot_qc_tag_frequency(df:pd.DataFrame,
                          save_path:str=None,
                          save_name:str="qc_tag_frequency_{}.png".format(TODAY),
                          palette:dict=pass_flag_fail_palette,
                          figsize:tuple=None):
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
    fig, axs = plt.subplots(1, 2, sharey=False)
    if figsize:
        fig.set_size_inches(figsize)
    
    fig.suptitle('QC Tag Frequency')

    # Create plots for session and experiment data
    create_qc_tag_bar_plot(session_df, axs[0], 'Session Tags', palette=palette)
    create_qc_tag_bar_plot(experiment_df, axs[1], 'Experiment Tags', palette=palette)

    plt.tight_layout() 

    # Save the plot if a save path and name are provided
    if save_path and save_name:
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file)
        print(f'Saved plot to {save_file}')

    plt.show()


def plot_qc_generation_status(df: pd.DataFrame,
                              palette: dict = general_qc_status_palette,
                              title: str = "QC Status Frequency",
                              save_path: str = None,
                              save_name: str = "qc_gen_status_test_{}.png".format(TODAY),
                              figsize: tuple = None,
                              height_ratios: tuple = (4, 3)):
    
    # Create a figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios)

    # Create the top plot
    top_ax = fig.add_subplot(gs[0])
    top_ax.set_title(title)
    top_ax.axis('off')  # Hide the axis for the top plot
    plot_colored_qc_generation_status_table(df, ax=top_ax, palette=palette)

    # Create the bottom plots
    # 1 row, 3 columns for bottom plots
    bottom_gs = gs[1].subgridspec(1, 3, wspace=0.3)
    bottom_axs = [fig.add_subplot(bottom_gs[col]) for col in range(3)]

    create_qc_status_bar_plot(df, "generation_status", bottom_axs[0], 'Report Generation', palette=palette)
    create_qc_status_bar_plot(df, "review_status", bottom_axs[1], 'Report Review', palette=palette)
    create_qc_status_bar_plot(df, "qc_outcome", bottom_axs[2], 'QC Outcome', palette=palette)

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
                                       show_labels: bool = True,
                                       figsize: tuple = None):
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
                             show_labels=show_labels,
                             figsize=figsize)


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