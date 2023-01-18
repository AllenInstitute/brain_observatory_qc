import pandas as pd

MOUSE_NAMES = {"603892": "Gold",
               "608368": "Silicon",
               "612764": "Silver",
               "624942": "Bronze",
               "629294": "Copper",
               "632324": "Nickel",
               "633532": "Titanium",
               "636496": "Aluminum",
               "639224": "Mercury",
               "646883": "Iron",
               "623975": "Helium",
               "623972": "Neon",
               "631563": "Argon",
               "637848": "Xenon",
               "637851": "Radon"}


def experiment_table_extended(df: pd.DataFrame):
    """Adds extra columns to the expt_table #WFDF

    Parameters:
    -----------
    df : pandas.DataFrame
        experiment table

    Returns:
    --------
    df : pandas.DataFrame
        experiment table with additional columns

    Notes:
    ------
    Adds the following columns:
        - 'session_type_num'
        - 'bisect_layer'
        - 'depth_order'
        - 'n_exposure_session_type'
        - 'n_exposure_session_type_num'
        - 'n_exposure_session_type_num_layer'
        - 'reporter_line'
        - 'reporter'

    """
    df = add_n_exposure_session_type_column(df)
    df = add_session_type_num_column(df)
    df = add_bisect_layer_column(df)
    df = add_depth_order_column(df)
    df = add_fixed_reporter_line_column(df)
    df = add_fixed_reporter_line_column(df)
    df = add_mouse_names_columns(df)

    # TODO: PROJECT SPECIFIC (order matters)
    # df = add_short_session_name_column(df)
    # df = add_n_exposure_stimulus_column(df) #(epe)
    # df = add_short_session_name_num_column(df)

    return df


def add_mouse_names_columns(df: pd.DataFrame):
    """Adds a column called 'mouse_name' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "mouse_id" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    """

    df["mouse_name"] = df["mouse_id"].map(MOUSE_NAMES)

    return df


def add_session_type_num_column(df: pd.DataFrame):
    """Adds a column called 'session_type_num' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "session_type" (et) & "n_exposure_session_type" (ete)
        columns

    Returns:
    --------
    df : pandas.DataFrame

    Examples:
    --------

    "TRAINING_1_gratings" -> "TRAINING_1_gratings_1"
    "TRAINING_1_gratings" -> "TRAINING_1_gratings_2"
    "TRAINING_1_gratings" -> "TRAINING_1_gratings_3"
    "TRAINING_2_gratings_flashed" -> "TRAINING_2_gratings_flashed_1"
    ...

    """
    df["session_type_num"] = df["session_type"] + "_" + \
        df["n_exposure_session_type"].astype(str)

    return df


def add_bisect_layer_column(df, bisecting_depth=220):
    """Adds a column called 'bisect_layer' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "imaging_depth" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    """
    df.loc[:, 'bisect_layer'] = None

    indices = df[(df.imaging_depth < bisecting_depth)].index.values
    df.loc[indices, 'bisect_layer'] = 'upper'

    indices = df[(df.imaging_depth > bisecting_depth)].index.values
    df.loc[indices, 'bisect_layer'] = 'lower'

    return df


def add_depth_order_column(df):
    """Adds a column called 'depth_order' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "imaging_depth" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------
    Will rank the order of the imaging planes from 1 to N, th most dorsal plane
    will be 1. Generally the 1st planes are less than 220 um"""

    gb = ["ophys_session_id", "targeted_structure"]
    df["depth_order"] = (df.groupby(gb)["imaging_depth"]
                         .transform(lambda x: x.rank(method="dense",
                                    ascending=True)))

    return df


###############################################################################
def add_n_exposure_session_type_column(df):
    """Adds a column called 'n_exposure_session_type' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "mouse_id" (et) & "session_type" (et) columns

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """

    df["n_exposure_session_type"] = (df.groupby(["mouse_id", "session_type"])
                                     ["date_of_acquisition"]
                                     .rank(method="dense", ascending=True)
                                     .astype(int))

    return df


# TODO: this is an extended column
def add_n_exposure_stimulus_column(df):
    """stim exposure based on short session name (which groups stim)"""

    df["n_exposure_stimulus"] = (df.groupby(["mouse_id", "short_session_name"])["date_of_acquisition"]
                                   .rank(method="dense", ascending=True)
                                   .astype(int))

    return df


def add_abbreviated_reporter_line_column(df):

    def abbreviate_reporter_line(row):
        return row.reporter_line.split("(")[0]

    df["reporter_line"] = df.apply(abbreviate_reporter_line, axis=1)

    return df


def add_fixed_reporter_line_column(df):

    def fix_reporter_line(row):
        return row.full_genotype.split(';')[-1].split('/')[0]

    df["reporter"] = df.apply(fix_reporter_line, axis=1)

    return df


###############################################################################
# PROJECTS

def add_short_session_name_column(df):
    """Adds a column called 'short_session_name' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "session_type" (et) column

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """
    # import mjd_dev.lamf_task1a as constants
    # df["short_session_name"] = df["session_type"].map(constants.short_session_names)
    # issue warning, not implemented

    import warnings
    warnings.warn("Not implemented")

    return df


def add_short_session_name_num_column(df):
    """Adds a column called 'short_session_name_num' to expt_table #WKDF

    Parameters:
    -----------
    df : pandas.DataFrame
        expt_table with "short_session_name" (ete) & "n_exposure_stimulus" (ete)
        columns

    Returns:
    --------
    df : pandas.DataFrame

    Notes:
    ------

    """

    df["short_session_name_num"] = df["short_session_name"] + " - " + \
        df["n_exposure_stimulus"].astype(str)

    return df
