import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path


def is_int(n):
    return isinstance(n, (int, np.integer))


def is_float(n):
    return isinstance(n, (float, np.float))


def is_uuid(n):
    return isinstance(n, uuid.UUID)


def is_bool(n):
    return isinstance(n, (bool, np.bool_))


def is_array(n):
    return isinstance(n, np.ndarray)


def correct_filepath(filepath:str)->str:
    """using the pathlib python module, takes in a filepath from an
    arbitrary operating system and returns a filepath that should work
    for the users operating system

    Parameters
    ----------
    filepath : string
        given filepath

    Returns
    -------
    string
        filepath adjusted for users operating system
    """
    if filepath is None or filepath=="NA" or filepath=="":
        corrected_path = filepath
    else:
        filepath = filepath.replace('/allen', '//allen')
        corrected_path = Path(filepath)
    return corrected_path


def correct_dataframe_filepath(df:pd.DataFrame, column: str) -> pd.DataFrame:
    """applies the correct_filepath function to a given dataframe
    column, replacing the filepath in that column in place


    Parameters
    ----------
    dataframe : table
        pandas dataframe with the column
    column_string : string
        the name of the column that contains the filepath to be
        replaced

    Returns
    -------
    dataframe
        returns the input dataframe with the filepath in the given
        column 'corrected' for the users operating system, in place
    """
    df[column] = df[column].apply(lambda x: correct_filepath(x))
    return df


def save_df_to_csv(df:pd.DataFrame, 
                   csv_name:str = None,
                   csv_path:str = None,
                   index:bool = False)-> None:
    """saves a dataframe as a csv file.
    Checks that filepath is proper format for os

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to save
    csv_name : str
        name of the csv file
    csv_path : str
        path to save the csv file
        defaults to current working directory if no path is provided
    """
    if csv_path:
        csv_path = correct_filepath(csv_path) # ensure filepath is proper format for operating system
    else:       
        csv_path = os.getcwd()  # Set path to current working directory if not provided

    # check if csv_name ends with ".csv"
    if csv_name.endswith(".csv"):
        full_path = os.path.join(csv_path, csv_name)
    else:
        csv_name = csv_name + ".csv"
        full_path = os.path.join(csv_path, csv_name)

    # save dat data!
    df.to_csv(full_path, index = index)
