
import os
import warnings
import pandas as pd
from psycopg2 import extras


from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP

import mindscope_qc.data_access.utilities as utils
import mindscope_qc.utilities.pre_post_conditions as conditions

#####################################################################
#
#           CONNECT TO LIMS
#
#####################################################################
try:
    lims_dbname = os.environ["LIMS_DBNAME"]
    lims_user = os.environ["LIMS_USER"]
    lims_host = os.environ["LIMS_HOST"]
    lims_password = os.environ["LIMS_PASSWORD"]
    lims_port = os.environ["LIMS_PORT"]

    lims_engine = PostgresQueryMixin(
        dbname=lims_dbname,
        user=lims_user,
        host=lims_host,
        password=lims_password,
        port=lims_port
    )

    # building querys
    mixin = lims_engine

except Exception as e:
    warn_string = 'failed to set up LIMS/mtrain credentials\n{}\n\n \
        internal AIBS users should set up environment variables \
        appropriately\nfunctions requiring database access will fail'.format(e)
    warnings.warn(warn_string)


def get_psql_dict_cursor():
    """Set up a connection to a psql db server with a dict cursor

    Returns
    -------
    [type]
        [description]
    """
    api = (credential_injector(LIMS_DB_CREDENTIAL_MAP)(PostgresQueryMixin)())
    connction = api.get_connection()
    connction.set_session(readonly=True, autocommit=True)
    return connction.cursor(cursor_factory=extras.RealDictCursor)


#####################################################################
#
#           LIMS TABLE LOCATION DICTIONARIES
#
#####################################################################


ALL_ID_TYPES_DICT = {
    "donor_id":            {"lims_table": "specimens",                             "id_column": "donor_id"},                 # noqa: E241, E501
    "cell_roi_id":         {"lims_table": "cell_rois",                             "id_column": "id"},                       # noqa: E241, E501
    "specimen_id":         {"lims_table": "specimens",                             "id_column": "id"},                       # noqa: E241, E501
    "labtracks_id":        {"lims_table": "specimens",                             "id_column": "external_specimen_name"},   # noqa: E241, E501
    "cell_specimen_id":    {"lims_table": "cell_rois",                             "id_column": "cell_specimen_id"},         # noqa: E241, E501
    "ophys_experiment_id": {"lims_table": "ophys_experiments",                     "id_column": "id"},                       # noqa: E241, E501
    "ophys_session_id":    {"lims_table": "ophys_sessions",                        "id_column": "id"},                       # noqa: E241, E501
    "foraging_id":         {"lims_table": "ophys_sessions",                        "id_column": "foraging_id"},              # noqa: E241, E501
    "behavior_session_id": {"lims_table": "behavior_sessions",                     "id_column": "id"},                       # noqa: E241, E501
    "ophys_container_id":  {"lims_table": "visual_behavior_experiment_containers", "id_column": "id"},                       # noqa: E241, E501
    "isi_experiment_id":   {"lims_table": "isi_experiments",                       "id_column": "id"},                       # noqa: E241, E501
    "supercontainer_id":   {"lims_table": "visual_behavior_supercontainers",       "id_column": "id"},                       # noqa: E241, E501
}


OPHYS_ID_TYPES_DICT = {
    "specimen_id":         {"lims_table": "specimens",         "id_column": "id",          "query_abbrev": "specimens.id"},  # noqa: E241
    "ophys_experiment_id": {"lims_table": "ophys_experiments", "id_column": "id",          "query_abbrev": "oe.id"},            # noqa: E241, E501
    "ophys_session_id":    {"lims_table": "ophys_sessions",    "id_column": "id",          "query_abbrev": "os.id"},           # noqa: E241, E501
    "foraging_id":         {"lims_table": "ophys_sessions",    "id_column": "foraging_id", "query_abbrev": "os.foraging_id"},   # noqa: E241, E501
    "behavior_session_id": {"lims_table": "behavior_sessions", "id_column": "id",          "query_abbrev": "bs.id"},           # noqa: E241, E501
    "supercontainer_id":   {"lims_table": "visual_behavior_supercontainers", "id_column": "id", "query_abbrev": "vbs.id"},        # noqa: E241, E501
    "ophys_container_id":  {"lims_table": "ophys_experiments_visual_behavior_experiment_containers",  "id_column": "visual_behavior_experiment_container_id", "query_abbrev": "oevbec.visual_behavior_experiment_container_id"},   # noqa: E241, E501
}


MOUSE_IDS_DICT = {
    "donor_id":               {"lims_table": "specimens", "id_column": "donor_id"},                 # noqa: E241, E501
    "specimen_id":            {"lims_table": "specimens", "id_column": "id"},                       # noqa: E241, E501
    "labtracks_id":           {"lims_table": "specimens", "id_column": "external_specimen_name"},   # noqa: E241, E501
    "external_specimen_name": {"lims_table": "specimens", "id_column": "external_specimen_name"},   # noqa: E241, E501
    "external_donor_name":    {"lims_table": "donors",    "id_column": "external_donor_name"}       # noqa: E241, E501
}


MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT = {
    "Nikon":       ["CAM2P.1", "CAM2P.2"],                     # noqa: E241
    "Scientifica": ["CAM2P.3, CAM2P.4, CAM2P.5, CAM2P.6"],     # noqa: E241, E501
    "Mesoscope":   ["MESO.1", "MESO.2"],                       # noqa: E241, E501
    "Deepscope":   ["DS.1"]                                    # noqa: E241, E501
}

GEN_INFO_QUERY_DICT = {
    "ophys_experiment_id": {"query_abbrev": "oe.id"},          # noqa: E241
    "ophys_session_id":    {"query_abbrev": "os.id"},          # noqa: E241
    "behavior_session_id": {"query_abbrev": "bs.id"},
    "ophys_container_id":  {"query_abbrev": "vbec.id"},        # noqa: E241
    "supercontainer_id":   {"query_abbrev": "os.visual_behavior_supercontainer_id"}  # noqa: E241
}


#####################################################################
#
#           FLEXIBLE LIMS/SQL QUERY
#
#####################################################################

def generic_lims_query(query: str) -> pd.DataFrame:
    """
    execute a SQL query in LIMS

    Parameters
    ----------
    query : string
        the type of ID to search on. allowable id_types:
            donor_id
            specimen_id
            labtracks_id: Labtracks ID (6 digit ID on mouse cage)
            external_specimen_name: alternate name for labtracks_id
                (used in specimens table)
            external_donor_name: alternate name for labtracks_id
                (used in donors table)
    id_number : int,string, list of ints or list of strings
        the associated ID number(s)

    Returns
    -------
    dataframe
        * the result if the result is a single element
        * results in a pandas dataframe otherwise

    Examples
    --------
    >> generic_lims_query('select ophys_session_id from
                   ophys_experiments where id = 878358326')
        returns 877907546

    >> generic_lims_query('select * from ophys_experiments where id = 878358326')
        returns a single line dataframe with all columns from the
        ophys_experiments table for ophys_experiment_id =  878358326

    >> generic_lims_query('select * from ophys_sessions where id in (877907546,
                                                             876522267,
                                                             869118259)')
        returns a three line dataframe with all columns from the
            ophys_sessions table for ophys_session_id in the
            list [877907546, 876522267, 869118259]

    >> generic_lims_query('select * from ophys_sessions where specimen_id = 830901424')
        returns all rows and columns in the ophys_sessions table
            for specimen_id = 830901424
    """

    df = pd.read_sql(query)
    if df.shape == (1, 1):
        # if the result is a single element, return only that element
        return df.iloc[0][0]
    else:
        # otherwise return in dataframe format
        return df


def get_value_from_table(search_key, search_value, target_table, target_key):
    """a general function for getting a value from a LIMS table

    Parameters
    ----------
    search_key : _type_
        _description_
    search_value : _type_
        _description_
    target_table : _type_
        _description_
    target_key : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    query = '''
        select {}
        from {}
        where {} = '{}'
    '''
    result = pd.read_sql(query.format(target_key, target_table, search_key, search_value))
    if len(result) == 1:
        return result[target_key].iloc[0]
    else:
        return None

#####################################################################
#
#           GET IDS & ID TYPES
#
#####################################################################


def get_LIMS_id_type(id_number: int) -> str:
    """ A function to find the id type of the input id.

    This function can detect the following id types included in ALL_ID_TYPES_DICT
    "donor_id"
    "cell_roi_id"
    "specimen_id"
    "labtracks_id"
    "cell_specimen_id"
    "ophys_experiment_id"
    "ophys_session_id"
    "foraging_id"
    "behavior_session_id"
    "ophys_container_id"
    "isi_experiment_id"
    "supercontainer_id"

    Parameters
    ----------
    id_number : int
        A lims id

    Returns
    -------
    str
        The ID type of the input ID
    """
    found_id_types = []
    for id_type_key in ALL_ID_TYPES_DICT:
        if len(general_id_type_query(id_type_key, id_number)) > 0:
            found_id_types.append(id_type_key)

    # assert that no more than one ID type was found (they should be unique)
    assert len(found_id_types) <= 1, 'multiple id types found: {}'.format(found_id_types)

    if len(found_id_types) == 1:
        # if only one id type was found, return it
        id_type = found_id_types[0]
    else:
        # return 'unknown_id' if id was not found or more than one ID type was found
        id_type = "unknown_id"

    return id_type


def general_id_type_query(id_type: str, id_number: int):
    """ A pre-built dynamic query that utilizes the ALL_ID_TYPES_DICT
    to get all columns from the LIMS table related to a specific
    lims ID type and number.

    Parameters
    ----------
    id_type : str
        Type of ID. Used with ALL_ID_TYPES_DICT to determine
        appropriate lims2 table to query.
    id_number : int, str
       Generic lims ID.

    Returns
    -------
    pd.DataFrame
        Returns a 1xN dataframe that contains all columns
        from the appropriate lims table.
    """
    conditions.validate_value_in_dict_keys(id_type,
                                           ALL_ID_TYPES_DICT,
                                           "ALL_ID_TYPES_DICT")
    query = '''
    SELECT *
    FROM {}
    WHERE {} = '{}' limit 1
    '''.format(ALL_ID_TYPES_DICT[id_type]["lims_table"],
               ALL_ID_TYPES_DICT[id_type]["id_column"],
               id_number)

    table_row = mixin.select(query)
    return table_row


def get_all_imaging_ids_for_imaging_id(id_type: str, id_number: int) -> pd.DataFrame:
    """ Will get all other ophys & imaginge related LIMS id types
    when given a single ophys/imaging related id. See acceptable
    LIMS ID types in parameters.

    Parameters
    ----------
    id_type : str
        options are the keys in the OPHYS_ID_TYPES_DICT
        "specimen_id"
        "ophys_experiment_id"
        "ophys_session_id"
        "foraging_id"
        "behavior_session_id"
        "ophys_container_id"
        "supercontainer_id"

    id_number : int
        the actual ID number for LIMS to lookup

    Returns
    -------
    pd.DataFrame
        A table with all of the output IDS listed.
    """
    conditions.validate_value_in_dict_keys(id_type,
                                           OPHYS_ID_TYPES_DICT,
                                           "OPHYS_ID_TYPES_DICT")
    validate_LIMS_id_type(id_number, id_type)

    query = '''
    SELECT
    specimens.id as specimen_id,
    oe.id AS ophys_experiment_id,
    os.id AS ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id AS foraging_id,
    oevbec.visual_behavior_experiment_container_id AS ophys_container_id,
    vbs.id AS supercontainer_id

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN specimens
    ON os.specimen_id = specimens.id

    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id

    JOIN visual_behavior_supercontainers vbs
    ON os.visual_behavior_supercontainer_id = vbs.id

    WHERE
    {} = {}
    '''.format(OPHYS_ID_TYPES_DICT[id_type]["query_abbrev"], id_number)
    all_ids = mixin.select(query)

    return all_ids


def get_microscope_type(ophys_session_id: int) -> str:
    """maps a microscope's speciic 'equipment_name'
    onto a "type" of microscope. Utilizes on the
    MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT

    Mapping : "microscope_type"   [equipment_names]

    "Nikon":       ["CAM2P.1", "CAM2P.2"]
    "Scientifica": ["CAM2P.3, CAM2P.4, CAM2P.5, CAM2P.6"]
    "Mesoscope":   ["MESO.1", "MESO.2"]
    "Deepscope":   ["DS.1"]

    Parameters
    ----------
    ophys_session_id : int
        unique identifier for an ophys imaging session

    Returns
    -------
    str
        type of microscope.Options:
        "Nikon", "Scientifica", "Mesoscope", "Deepscope"
    """
    validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    equipment_name = get_general_info_for_LIMS_imaging_id("ophys_session_id", ophys_session_id)["equipment_name"][0]

    for key, value in MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT.items():
        if equipment_name in value:
            return key
    return "Cannot find microscope type for {}".format(equipment_name)


def get_mouse_ids_from_id(id_type: str, id_number: int) -> pd.DataFrame:
    """
    returns a dataframe of all variations (donor_id, labtracks_id,
    specimen_id) of mouse ID for a given input ID.

    Note: in rare cases, a single donor_id/labtracks_id was
    associated with multiple specimen_ids this occured for IDs used
    as test_mice (e.g. labtracks_id 900002) and should not have
    occured for real experimental mice

    Parameters
    ----------
    id_type : string
        the type of ID to search on. allowable id_types:
            donor_id
            specimen_id
            labtracks_id: Labtracks ID (6 digit ID on mouse cage)
            external_specimen_name: alternate name for labtracks_id
                (used in specimens table)
            external_donor_name: alternate name for labtracks_id
                (used in donors table)
    id_number : int,string, list of ints or list of strings
        the associated ID number(s)

    Returns
    -------
    pd.DataFrame
        a dataframe with columns for `donor_id`, `labtracks_id`, `specimen_id`
    """
    conditions.validate_value_in_dict_keys(id_type, MOUSE_IDS_DICT, "MOUSE_IDS_DICT")
    query = '''
    SELECT
    donors.id  AS donor_id,
    donors.external_donor_name AS labtracks_id,
    specimens.id AS specimen_id

    FROM
    donors
    JOIN specimens ON donors.external_donor_name = specimens.external_specimen_name

    WHERE {}.{} = '{}'
    '''.format(MOUSE_IDS_DICT[id_type]["lims_table"], MOUSE_IDS_DICT[id_type]["id_column"], id_number)
    mouse_ids = mixin.select(query)
    return mouse_ids


def get_general_info_for_LIMS_imaging_id(id_type: str, id_number: int) -> pd.DataFrame:
    """
    combines columns from several different lims tables to provide
    some basic overview information.

    Parameters
    ----------
    id_type : string
        the type of lims_id that is being entered. (i.e :ophys_experiment_id)
        Acceptable id_types are found in the keys in the GEN_INFO_QUERY_DICT.
        Current options are:
            "ophys_experiment_id"
            "ophys_session_id"
            "behavior_session_id"
            "ophys_container_id"
            "ophys_supercontainer_id"
    id_number : int
        the id number for the unique experiment, ophys_session,
        behavior_session etc. Usually a 9 digit number.


    Returns
    -------
    DataFrame
        dataframe with the following columns:
            ophys_experiment_id
            ophys_session_id
            behavior_session_id
            foraging_id
            ophys_container_id
            supercontainer_id
            experiment_workflow_state
            session_workflow_state
            container_workflow_state
            specimen_id
            donor_id
            specimen_name
            date_of_acquisition
            session_type
            targeted_structure
            depth
            equipment_name
            project
            experiment_storage_directory
            behavior_storage_directory
            session_storage_directory
            container_storage_directory
            supercontainer_storage_directory
            specimen_storage_directory
    """

    conditions.validate_value_in_dict_keys(id_type,
                                           GEN_INFO_QUERY_DICT,
                                           "GEN_INFO_QUERY_DICT")
    validate_LIMS_id_type(id_number, id_type)
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.ophys_session_id,
    bs.id AS behavior_session_id,
    os.foraging_id,
    vbec.id AS ophys_container_id,
    os.visual_behavior_supercontainer_id AS supercontainer_id,

    oe.workflow_state AS experiment_workflow_state,
    os.workflow_state AS session_workflow_state,
    vbec.workflow_state AS container_workflow_state,

    os.specimen_id,
    specimens.donor_id,
    specimens.name AS specimen_name,

    os.date_of_acquisition,
    os.stimulus_name AS session_type,
    structures.acronym AS targeted_structure,
    imaging_depths.depth,
    equipment.name AS equipment_name,
    projects.code AS project,

    oe.storage_directory AS experiment_storage_directory,
    bs.storage_directory AS behavior_storage_directory,
    os.storage_directory AS session_storage_directory,
    vbec.storage_directory AS container_storage_directory,
    vbs.storage_directory AS supercontainer_storage_directory,
    specimens.storage_directory AS specimen_storage_directory


    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id

    JOIN visual_behavior_experiment_containers vbec
    ON vbec.id = oevbec.visual_behavior_experiment_container_id

    JOIN visual_behavior_supercontainers vbs
    ON os.visual_behavior_supercontainer_id = vbs.id

    JOIN projects ON projects.id = os.project_id
    JOIN specimens ON specimens.id = os.specimen_id
    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id
    JOIN equipment ON equipment.id = os.equipment_id

    WHERE
    {} = {}
    '''.format(GEN_INFO_QUERY_DICT[id_type]["query_abbrev"],
               id_number)

    general_info = mixin.select(query)

    # ensure operating system compatible filepaths
    general_info = correct_LIMS_storage_directory_filepaths(general_info)

    return general_info


#####################################################################
#
#           FILEPATHS & STORAGE DIRECTORIES
#
#####################################################################


def correct_LIMS_storage_directory_filepaths(dataframe: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    dataframe : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    storage_directory_columns_list = ['specimen_storage_directory',
                                      'experiment_storage_directory',
                                      'behavior_storage_directory',
                                      'session_storage_directory',
                                      'container_storage_directory'
                                      'supercontainer_storage_directory']

    for column in storage_directory_columns_list:
        dataframe = utils.correct_dataframe_filepath(dataframe, column)
    return dataframe


def get_storage_directories_for_id(id_type: str, id_number: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    id_type : str
        options are the keys in the OPHYS_ID_TYPES_DICT
        "ophys_experiment_id"
        "ophys_session_id"
        "foraging_id"
        "behavior_session_id"
        "ophys_container_id"
        "supercontainer_id"
    id_number : int
        usually 9 digits, foraging IDs are the exception

    Returns
    -------
    pd.DataFrame
        dataframe with the following columns:
        "specimen_storage_directory"
        "experiment_storage_directory"
        "behavior_storage_directory"
        "session_storage_directory"
        "container_storage_directory"
        "supercontainer_storage_directory"
    """

    query = '''
    specimens.storage_directory AS specimen_storage_directory,
    oe.storage_directory AS experiment_storage_directory,
    bs.storage_directory AS behavior_storage_directory,
    os.storage_directory AS session_storage_directory,
    vbec.storage_directory AS container_storage_directory,
    vbs.storage_directory as supercontainer_storage_directory

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN specimens
    ON os.specimen_id = specimens.id

    JOIN behavior_sessions bs
    ON bs.foraging_id = os.foraging_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oe.id = oevbec.ophys_experiment_id

    JOIN visual_behavior_experiment_containers vbec
    ON vbec.id = oevbec.visual_behavior_experiment_container_id

    JOIN visual_behavior_supercontainers vbs
    ON os.visual_behavior_supercontainer_id = vbs.id

    WHERE
    {} = {}
    '''.format(OPHYS_ID_TYPES_DICT[id_type]["query_abbrev"], id_number)
    storage_directories_df = mixin.select(query)
    storage_directories_df = correct_LIMS_storage_directory_filepaths(storage_directories_df)
    return storage_directories_df


#####################################################################
#
#           VALIDATIONS
#
#####################################################################


def validate_microscope_type(correct_microscope_type: str, ophys_session_id: int):
    """gets the microscope id for a given ophys session id and maps
    it on to a "microscope_type" then compares that microscope type
    to the "correct" or desidered microscope type

    Mappinng. Microscope type and list of equiment names that match
    to that type
    "Nikon":       ["CAM2P.1", "CAM2P.2"],
    "Scientifica": ["CAM2P.3, CAM2P.4, CAM2P.5, CAM2P.6"],
    "Mesoscope":   ["MESO.1", "MESO.2"],
    "Deepscope":   ["DS.1"]

    Parameters
    ----------
    correct_microscope_type : str
        Enumerated string. Options are:
        "Nikon", "Scientifica", "Mesoscope", "Deepscope"
    ophys_session_id : int
        unique identifer for an ophys imaging session
    """

    session_microscope_type = get_microscope_type(ophys_session_id)
    assert session_microscope_type == correct_microscope_type, "Error: incorrect microscope type.\
        {} provided but {} necessary.".format(session_microscope_type, correct_microscope_type)


def validate_LIMS_id_type(desired_id_type: str, id_number: int):
    """takes an input id, looks up what type of id it is and then validates
       whether it is the same as the desired/correct id type

    Parameters
    ----------
    input_id : int
         the numeric any of the common types of ids associated or used with an optical physiology.
         Examples: ophys_experiment_id, ophys_session_id, cell_roi_id etc. See ID_TYPES_DICT in
         from_lims module for complete list of acceptable id types
    correct_id_type : string
        [description]
    """
    conditions.validate_value_in_dict_keys(desired_id_type, ALL_ID_TYPES_DICT, "ID_TYPES_DICT")
    id_number_type = get_LIMS_id_type(id_number)
    assert id_number_type == desired_id_type, "Incorrect id type. Entered Id type is {},\
        correct id type is {}".format(id_number_type, desired_id_type)


def validate_ophys_associated_with_behavior(behavior_session_id: int):
    """_summary_

    Parameters
    ----------
    behavior_session_id : _type_
        _description_
    """
    validate_LIMS_id_type(behavior_session_id, "behavior_session_id")
    ophys_session_id = get_all_imaging_ids_for_imaging_id('behavior_session_id', behavior_session_id)['ophys_session_id'][0]
    assert ophys_session_id is not None, "There is no ophys_session_id \
        associated with this behavior_session_id: {}".format(behavior_session_id)
