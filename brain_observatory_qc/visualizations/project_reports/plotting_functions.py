import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import brain_observatory_qc.data_access.from_mouseQC as mqc

pass_flag_fail_color_mapping = {
    'pass': '#4CAF50',  # Green
    'flag': '#FFC107',  # Amber
    'fail': '#F44336'   # Red
}

qc_submit_status_palette = {
"ready":"#dfc27d",    # ready
"error":"#a6611a",     # error
"complete":"#018571",  # complete
"incomplete":"#80cdc1" # incomplete
}

data_stream_status_palette = {
"not_reported":"#bebada",  # Not Reported
"pass":"#8dd3c7",          # pass
"flag":"#ffffb3",          # flag
"fail":"#fb8072"           # fail
}

def plot_status_by_id_matrix(df:pd.DataFrame,
                            id_col:str='data_id',
                            color_mapping:dict = data_stream_status_palette,
                            save_path:str=None,
                            xlabel:str=None,
                            ylabel:str=None,
                            title:str=None,
                            figsize:tuple=(10,8)):
    """a generic function that will take a dataframe of a specific struture
    and create a colored grid plot based on the values in the dataframe
    y axis: ids 
    x axis: all other colums in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        a dataframe with a specific structure:
        - an id column (e.g. ophys_session_id, ophys_experiment_id)
        - all remaining columns contain string values that are keys in the color_mapping dictionary
        the strings represent the status of the given column

        example: df with columns: ophys_session_id, data_stream_1, data_stream_2, data_stream_3
        where all the string data_stream columns can be "pass", "flag", "fail"

    color_mapping : dict
        the color mapping dictionary that maps the string values in the dataframe to colors
    figsize : tuple, optional
        _description_, by default (10,8)
    save_path : str, optional
        _description_, by default None
    """
    # Create a color matrix based on the DataFrame values
    color_matrix = df.iloc[:, 1:].applymap(lambda x: color_mapping[x])

    # Plotting the colored grid
    fig, ax = plt.subplots(figsize=figsize)

    # Create a grid of colored boxes
    for (row_idx, col_idx), val in np.ndenumerate(df.iloc[:, 1:].values):
        ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, color=color_mapping[val]))

    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns[1:])) + 0.5)
    ax.set_yticks(np.arange(len(df)) + 0.5)
    ax.set_xticklabels(df.columns[1:], rotation=90)
    ax.set_yticklabels(df[id_col])

    # Add gridlines
    ax.hlines(np.arange(len(df) + 1), *ax.get_xlim(), color='gray')
    ax.vlines(np.arange(len(df.columns[1:]) + 1), *ax.get_ylim(), color='gray')

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

    # Save the plot if a save path is provided, otherwise show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    

    def plot_data_stream_outcomes_matrix(data_stream_outcomes_df, 
                                         id_col = "data_id",
                                         save_path=None):


    def plot_qc_status_by_id_matrix(mongo_connection, 
                                start_date, 
                                end_date, 
                                save_path=None):