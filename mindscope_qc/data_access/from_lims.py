import os
import h5py
import warnings
import numpy as np
import pandas as pd
from array import array
from psycopg2 import extras


from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


import mindscope_qc.data_access.utilities as utils
import mindscope_qc.utilities.pre_post_conditions as conditions


### ACCESSING LIMS DATABASE ###      # noqa: E266
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


### GENERAL QUERIES ###      # noqa: E266


### ID TYPES ###      # noqa: E266


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


def get_mouse_ids_from_id(id_type: str, id_number: int):
    """_summary_

    Parameters
    ----------
    id_type : str
        _description_
    id_number : int
        _description_

    Returns
    -------
    _type_
        _description_
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

    WHERE {}.{} = {}
    '''.format(MOUSE_IDS_DICT[id_type]["lims_table"], MOUSE_IDS_DICT[id_type]["id_column"], id_number)
    mouse_ids = mixin.select(query)
    return mouse_ids


def get_mouse_ids(id_type: str, id_number: int) -> pd.DataFrame:
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
    dataframe
        a dataframe with columns for `donor_id`, `labtracks_id`, `specimen_id`

    Raises
    ------
    TypeError
        [description]
    """
    conditions.validate_value_in_dict_keys(id_type,
                                           MOUSE_IDS_DICT,
                                           "MOUSE_IDS_DICT")
    if id_type.lower() == 'donor_id':
        id_type_string = 'donors.id'
    elif id_type.lower() == 'specimen_id':
        id_type_string = 'specimens.id'
    elif id_type.lower() in ['labtracks_id', 'external_specimen_name', 'external_donor_name']:
        id_type_string = 'donors.external_donor_name'
    else:
        raise TypeError('invalid `id_type` {}'.format(id_type))

    if isinstance(id_number, (str, int, np.int64)):
        id_number = [id_number]
    id_number = [str(i) for i in id_number]

    query = '''
    SELECT
    donors.id donor_id,
    donors.external_donor_name AS labtracks_id,
    specimens.id AS specimen_id

    FROM donors
    JOIN specimens ON donors.external_donor_name = specimens.external_specimen_name
    where {} in {}'''.format(id_type_string, tuple(id_number)).replace(',)', ')')

    return lims_utils.lims_query(query)


def general_id_type_query(id_type: str, id_number: int):
    """_summary_

    Parameters
    ----------
    id_type : str
        _description_
    id_number : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    conditions.validate_value_in_dict_keys(id_type, ALL_ID_TYPES_DICT, "ALL_ID_TYPES_DICT")
    query = '''
    SELECT *
    FROM {}
    WHERE {} = {} limit 1
    '''.format(ALL_ID_TYPES_DICT[id_type]["lims_table"],
               ALL_ID_TYPES_DICT[id_type]["id_column"],
               id_number)

    table_row = mixin.select(query)
    return table_row


def get_id_type(id_number: int) -> str:
    """_summary_

    Parameters
    ----------
    id_number : int
        _description_

    Returns
    -------
    str
        _description_
    """
    found_id_types = []
    for id_type_key in ALL_ID_TYPES_DICT:
        if len(general_id_type_query(id_number, id_type_key)) > 0:
            found_id_types.append(id_type_key)

    # assert that no more than one ID type was found (they should be unique)
    assert len(found_id_types) <= 1, 'multiple id types found: {}'.format(found_id_types)

    if len(found_id_types) == 1:
        # if only one id type was found, return it
        id_type = found_id_types[0]
    else:
        # return 'unknown_id' if id was not found
        id_type = "unknown_id"

    return id_type


def get_microscope_type(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    equipment_name = get_general_info_for_ophys_session_id(ophys_session_id)["equipment_name"][0]

    for key, value in MICROSCOPE_TYPE_EQUIPMENT_NAMES_DICT.items():
        if equipment_name in value:
            return key
    return "Cannot find microscope type for {}".format(equipment_name)


### QUERIES USED FOR MULTIPLE FUNCTIONS ###      # noqa: E266

def correct_storage_directory_filepaths(dataframe: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    general_info_df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    storage_directory_columns_list = ['experiment_storage_directory',
                                      'session_storage_directory',
                                      'container_storage_directory']

    for column in storage_directory_columns_list:
        dataframe = utils.correct_dataframe_filepath(dataframe, column)
    return dataframe


def get_general_info_for_id(id_type: str, id_number: int) -> pd.DataFrame:
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
            session_storage_directory
    """

    conditions.validate_value_in_dict_keys(id_type,
                                           GEN_INFO_QUERY_DICT,
                                           "GEN_INFO_QUERY_DICT")
    conditions.validate_id_type(id_number, id_type)
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
    os.storage_directory AS session_storage_directory,
    vbec.storage_directory AS container_storage_directory

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
    general_info = correct_storage_directory_filepaths(general_info)

    return general_info


def get_all_ophys_ids_for_id(id_type: str, id_number: int) -> pd.DataFrame:
    """ Will get all other ophys related id types when given one
    of the the ids.

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
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    conditions.validate_value_in_dict_keys(id_type,
                                           OPHYS_ID_TYPES_DICT,
                                           "OPHYS_ID_TYPES_DICT")
    conditions.validate_id_type(id_number, id_type)

    query = '''
    SELECT
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


### ID TYPES ###      # noqa: E266

def get_donor_id_for_specimen_id(specimen_id: int) -> int:
    conditions.validate_id_type(specimen_id, "specimen_id")
    specimen_id = int(specimen_id)
    query = '''
    SELECT
    donor_id

    FROM
    specimens

    WHERE
    id = {}
    '''.format(specimen_id)
    donor_ids = mixin.select(query)
    return donor_ids


def get_specimen_id_for_donor_id(donor_id: int) -> int:
    """_summary_

    Parameters
    ----------
    donor_id : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    conditions.validate_id_type(donor_id, "donor_id")
    donor_id = int(donor_id)
    query = '''
    SELECT
    id

    FROM
    specimens

    WHERE
    donor_id = {}
    '''.format(donor_id)
    specimen_ids = mixin.select(query)
    return specimen_ids


# Cell / ROI related IDS
def get_ophys_experiment_id_for_cell_roi_id(cell_roi_id: int) -> int:
    '''
    returns the ophys experiment ID from which a given cell_roi_id
    was recorded

    Parameters:
    -----------
    cell_roi_id: int
        cell_roi_id of interest

    Returns:
    --------
    int
        ophys_experiment_id
    '''
    conditions.validate_id_type(cell_roi_id, "cell_roi_id")
    query = '''
    SELECT ophys_experiment_id
    FROM cell_rois
    WHERE id = {}
    '''.format(cell_roi_id)
    ophys_experiment_id = mixin.select(query)
    return ophys_experiment_id


# for ophys_experiment_id
def get_current_segmentation_run_id(ophys_experiment_id: int) -> int:
    """gets the id for the current cell segmentation run for a given experiment.
        Queries LIMS via AllenSDK PostgresQuery function.

    Parameters
    ----------
    ophys_experiment_id : int
        9 digit ophys experiment id

    Returns
    -------
    int
        current cell segmentation run id
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    query = '''
    SELECT
    id

    FROM
    ophys_cell_segmentation_runs

    WHERE
    current = true
    AND ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    current_run_id = mixin.select(query)
    current_run_id = int(current_run_id["id"][0])
    return current_run_id


def get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id: int) -> int:
    """uses an sqlite to query lims2 database. Gets the ophys_session_id
       for a given ophys_experiment_id from the ophys_experiments table
       in lims.
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    query = '''
    SELECT
    ophys_session_id

    FROM
    ophys_experiments

    WHERE
    id = {}
    '''.format(ophys_experiment_id)
    ophys_session_id = mixin.select(query)
    ophys_session_id = ophys_session_id["ophys_session_id"][0]
    return ophys_session_id


def get_behavior_session_id_for_ophys_experiment_id(ophys_experiment_id: int) -> int:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    query = '''
    SELECT
    bs.id AS behavior_session_id

    FROM
    behavior_sessions bs

    JOIN
    ophys_experiments oe ON oe.ophys_session_id = bs.ophys_session_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    behavior_session_id = mixin.select(query)
    behavior_session_id = behavior_session_id["behavior_session_id"][0]
    return behavior_session_id


def get_ophys_container_id_for_ophys_experiment_id(ophys_experiment_id: int) -> int:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    query = '''
    SELECT
    visual_behavior_experiment_container_id AS ophys_container_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers

    WHERE
    ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    ophys_container_id = mixin.select(query)
    ophys_container_id = ophys_container_id["ophys_container_id"][0]
    return ophys_container_id


def get_supercontainer_id_for_ophys_experiment_id(ophys_experiment_id: int) -> int:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    conditions.validate_microscope_type(ophys_session_id, "Mesoscope")
    query = '''
    SELECT
    sessions.visual_behavior_supercontainer_id as supercontainer_id

    FROM
    ophys_experiments experiments

    JOIN ophys_sessions sessions
    ON sessions.id = experiments.ophys_session_id

    WHERE
    experiments.id = {}
    '''.format(ophys_experiment_id)
    supercontainer_id = mixin.select(query)
    supercontainer_id = supercontainer_id["supercontainer_id"][0]
    return supercontainer_id


def get_all_ids_for_ophys_experiment_id(ophys_experiment_id: int) -> pd.DataFrame:
    """queries LIMS and gets all of the ids for a given ophys
    experiment id

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment
        (single FOV optical physiology session)

    Returns
    -------
    dataframe/table
        table with the following columns:
            ophys_experiment_id,
            ophys_session_id,
            behavior_session_id,
            foraging_id
            ophys_container_id,
            supercontainer_id

    """
    all_ids = get_all_ophys_ids_for_id("ophys_experiment_id", ophys_experiment_id)
    return all_ids


def get_general_info_for_ophys_experiment_id(ophys_experiment_id: int) -> pd.DataFrame:
    """
    combines columns from several different lims tables to provide
    some basic overview information.

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment (single FOV)

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
            session_storage_directory
    """
    general_info = get_general_info_for_id("ophys_experiment_id", ophys_experiment_id)
    return general_info


def get_genotype_for_ophys_experiment_id(ophys_experiment_id: int):
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    query = '''
    SELECT donors.full_genotype
    FROM ophys_experiments as OE

    JOIN ophys_sessions os ON  os.id = oe.ophys_session_id
    JOIN specimens ON specimens.id = os.specimen_id
    JOIN donors ON donors.id = specimens.donor_id

    WHERE oe.id = {}
    '''.format(ophys_experiment_id)
    genotype = mixin.select(query)
    return genotype


# for ophys_session_id
def get_ophys_experiment_ids_for_ophys_session_id(ophys_session_id):
    """uses an sqlite to query lims2 database. Gets the ophys_experiment_id
       for a given ophys_session_id from the ophys_experiments table in lims.

       note: number of experiments per session will vary depending on the
       rig it was collected on and the project it was collected for.
    """
    conditions.validate_id_type(ophys_session_id, 'ophys_session_id')
    query = '''
    SELECT
    id

    FROM ophys_experiments

    WHERE
    ophys_session_id = {}
    '''.format(ophys_session_id)
    ophys_experiment_ids = mixin.select(query)
    ophys_experiment_ids = ophys_experiment_ids.rename(columns={"id": "ophys_experiment_id"})
    return ophys_experiment_ids


def get_behavior_session_id_for_ophys_session_id(ophys_session_id: int) -> int:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    int
        _description_
    """
    conditions.validate_id_type(ophys_session_id, 'ophys_session_id')
    query = '''
    SELECT
    id AS behavior_session_id

    FROM behavior_sessions

    WHERE
    ophys_session_id = {}
    '''.format(ophys_session_id)
    behavior_session_id = mixin.select(query)
    behavior_session_id = behavior_session_id["behavior_session_id"][0]
    return behavior_session_id


def get_ophys_container_ids_for_ophys_session_id(ophys_session_id: int):
    conditions.validate_id_type(ophys_session_id, 'ophys_session_id')
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM ophys_experiments_visual_behavior_experiment_containers container
    JOIN ophys_experiments oe ON oe.id = container.ophys_experiment_id

    WHERE
    oe.ophys_session_id = {}
    '''.format(ophys_session_id)
    ophys_container_ids = mixin.select(query)
    return ophys_container_ids


def get_supercontainer_id_for_ophys_session_id(ophys_session_id: int) -> int:
    conditions.validate_id_type(ophys_session_id, 'ophys_session_id')
    conditions.validate_microscope_type(ophys_session_id, "Mesoscope")
    query = '''
    SELECT
    visual_behavior_supercontainer_id
    AS supercontainer_id

    FROM
    ophys_sessions

    WHERE id = {}
    '''.format(ophys_session_id)
    supercontainer_id = mixin.select(query)
    supercontainer_id = supercontainer_id["supercontainer_id"][0]
    return supercontainer_id


def get_all_ids_for_ophys_session_id(ophys_session_id: int) -> pd.DataFrame:
    """

    Parameters
    ----------
    ophys_session_id : int
        unique identifier for an ophys session

    Returns
    -------
    dataframe/table
        table with the following columns:
            ophys_experiment_id,
            ophys_session_id,
            behavior_session_id,
            foraging_id
            ophys_container_id,
            supercontainer_id
    """
    all_ids = get_all_ophys_ids_for_id("ophys_session_id", ophys_session_id)
    return all_ids


def get_general_info_for_ophys_session_id(ophys_session_id: int) -> pd.DataFrame:
    """combines columns from several different lims tables to provide
    basic overview information

    Parameters
    ----------
    ophys_session_id : int
        9 digit unique identifier for an ophys_session

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
            session_storage_directory
    """
    general_info = get_general_info_for_id("ophys_session_id", ophys_session_id)
    return general_info


# for behavior_session_id
def get_ophys_experiment_ids_for_behavior_session_id(behavior_session_id: int):
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")

    query = '''
    SELECT
    oe.id
    AS ophys_experiment_id

    FROM behavior_sessions behavior

    JOIN ophys_experiments oe
    ON oe.ophys_session_id = behavior.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)

    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_id_for_behavior_session_id(behavior_session_id: int) -> int:
    """uses an sqlite to query lims2 database. Gets the ophys_session_id
       for a given behavior_session_id from the behavior_sessions table in
       lims.

       note: not all behavior_sessions have associated ophys_sessions. Only
       behavior_sessions where ophys imaging took place will have
       ophys_session_id's
    """
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")

    query = '''
    SELECT
    ophys_session_id

    FROM behavior_sessions

    WHERE id = {}'''.format(behavior_session_id)

    ophys_session_id = mixin.select(query)
    ophys_session_id = ophys_session_id["ophys_session_id"][0]
    return ophys_session_id


def get_ophys_container_ids_for_behavior_session_id(behavior_session_id: int):
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")

    query = '''
    SELECT
    container.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)

    container_id = mixin.select(query)
    return container_id


def get_supercontainer_id_for_behavior_session_id(behavior_session_id: int) -> int:
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")
    ophys_session_id = get_ophys_session_id_for_behavior_session_id(behavior_session_id)
    conditions.validate_microscope_type(ophys_session_id, "Mesoscope")

    query = '''
    SELECT
    sessions.visual_behavior_supercontainer_id AS supercontainer_id

    FROM
    ophys_experiments experiments

    JOIN behavior_sessions behavior
    ON behavior.ophys_session_id = experiments.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers container
    ON container.ophys_experiment_id = experiments.id

    JOIN ophys_sessions sessions on sessions.id = experiments.ophys_session_id

    WHERE
    behavior.id = {}
    '''.format(behavior_session_id)

    supercontainer_id = mixin.select(query)
    supercontainer_id = supercontainer_id["supercontainer_id"][0]
    return supercontainer_id


def get_all_ids_for_behavior_session_id(behavior_session_id):
    """queries LIMS and gets all of the ids for a given behavior
    session id

    Parameters
    ----------
    behavior_session_id : int
        unique identifier for a behavior session

    Returns
    -------
    dataframe/table
        table with the following columns:
            ophys_experiment_id,
            ophys_session_id,
            behavior_session_id,
            ophys_container_id,
            supercontainer_id
    """
    all_ids = get_all_ophys_ids_for_id("behavior_session_id", behavior_session_id)
    return all_ids


def get_general_info_for_behavior_session_id(behavior_session_id: int) -> pd.DataFrame:
    """
    combines columns from several different lims tables to provide
    basic overview information

    Parameters
    ----------
    behavior_session_id : int
        [description]

    Returns
    -------
    dataframe
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
            session_storage_directory
    """
    general_info = get_general_info_for_id("behavior_session_id", behavior_session_id)
    return general_info


# for ophys_container_id


def get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id):
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")

    query = '''
    SELECT
    ophys_experiment_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers

    WHERE
    visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_ids_for_ophys_container_id(ophys_container_id):
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")
    query = '''
    SELECT
    oe.ophys_session_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers oevbec

    JOIN ophys_experiments oe
    ON oe.id = oevbec.ophys_experiment_id

    WHERE
    oevbec.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    ophys_session_ids = mixin.select(query)
    return ophys_session_ids


def get_behavior_session_ids_for_ophys_container_id(ophys_container_id):
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")

    query = '''
    SELECT
    bs.id
    AS behavior_session_id

    FROM
    ophys_experiments oe

    JOIN behavior_sessions bs
    ON bs.ophys_session_id = oe.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oevbec.ophys_experiment_id = oe.id

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    WHERE
    oevbec.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)

    behavior_session_id = mixin.select(query)
    return behavior_session_id


def get_current_container_run_id_for_ophys_container_id(ophys_container_id):
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")

    query = '''
    SELECT
    vbcr.id AS container_run_id

    FROM
    visual_behavior_container_runs as vbcr

    WHERE
    vbcr.current = true
    AND vbcr.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    current_container_run_id = mixin.select(query)
    return current_container_run_id


def get_supercontainer_id_for_ophys_container_id(ophys_container_id):
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")
    ophys_session_id = get_ophys_session_ids_for_ophys_container_id(ophys_container_id)["ophys_session_id"][0]
    conditions.validate_microscope_type(ophys_session_id, "Mesoscope")
    ophys_container_id = int(ophys_container_id)
    query = '''
    SELECT
    os.visual_behavior_supercontainer_id AS supercontainer_id

    FROM
    ophys_experiments oe

    JOIN behavior_sessions bs
    ON bs.ophys_session_id = oe.ophys_session_id

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oevbec.ophys_experiment_id = oe.id

    JOIN ophys_sessions os on os.id = oe.ophys_session_id

    WHERE
    oevbec.visual_behavior_experiment_container_id = {}
    '''.format(ophys_container_id)
    supercontainer_id = mixin.select(query)
    supercontainer_id = int(supercontainer_id['supercontainer_id'][0])
    return supercontainer_id


def get_all_ids_for_ophys_container_id(ophys_container_id):
    all_ids = get_all_ophys_ids_for_id("ophys_container_id", ophys_container_id)
    return all_ids


def get_general_info_for_ophys_container_id(ophys_container_id: int) -> pd.DataFrame:
    """combines columns from several different lims tables to provide
    basic overview information

    Parameters
    ----------
    ophys_container_id : int
        9 digit unique identifier

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
            session_storage_directory
    """
    general_info = get_general_info_for_id("ophys_container_id", ophys_container_id)
    return general_info


# for supercontainer_id


def get_ophys_experiment_ids_for_supercontainer_id(supercontainer_id: int):
    """_summary_

    Parameters
    ----------
    supercontainer_id : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    conditions.validate_id_type(supercontainer_id, "supercontainer_id")

    query = '''
    SELECT
    oe.id AS ophys_experiment_id

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    WHERE
    os.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)

    ophys_experiment_ids = mixin.select(query)
    return ophys_experiment_ids


def get_ophys_session_ids_for_supercontainer_id(supercontainer_id: int):
    """_summary_

    Parameters
    ----------
    supercontainer_id : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    conditions.validate_id_type(supercontainer_id, "supercontainer_id")

    query = '''
    SELECT
    id AS ophys_session_id

    FROM
    ophys_sessions

    WHERE
    visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)

    ophys_session_ids = mixin.select(query)
    return ophys_session_ids


def get_behavior_session_id_for_supercontainer_id(supercontainer_id: int) -> int:
    conditions.validate_id_type(supercontainer_id, "supercontainer_id")

    query = '''
    SELECT
    bs.id AS behavior_session_id

    FROM
    ophys_sessions os

    JOIN behavior_sessions bs
    ON bs.ophys_session_id = os.id

    WHERE
    os.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    behavior_session_ids = mixin.select(query)
    return behavior_session_ids


def get_ophys_container_ids_for_supercontainer_id(supercontainer_id: int):
    conditions.validate_id_type(supercontainer_id, "supercontainer_id")

    query = '''
    SELECT
    oevbec.visual_behavior_experiment_container_id
    AS ophys_container_id

    FROM
    ophys_experiments oe

    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec
    ON oevbec.ophys_experiment_id = oe.id

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    WHERE
    os.visual_behavior_supercontainer_id = {}
    '''.format(supercontainer_id)
    ophys_container_ids = mixin.select(query)
    return ophys_container_ids


def get_all_ids_for_supercontainer_id(supercontainer_id: int):
    """queries LIMS and gets all of the ids for a given
    supercontainer id

    Parameters
    ----------
    supercontainer_id : [type]
        [description]

    Returns
    -------
    dataframe/table
        table with the following columns:
            ophys_experiment_id,
            ophys_session_id,
            behavior_session_id,
            ophys_container_id,
            supercontainer_id
    """
    all_ids = get_all_ophys_ids_for_id("supercontainer_id", supercontainer_id)
    return all_ids


def get_general_info_for_supercontainer_id(supercontainer_id: int) -> pd.DataFrame:
    """combines columns from several different lims tables to provide
    basic overview information

    Parameters
    ----------
    supercontainer_id : int
        [description]

    Returns
    -------
    dataframe
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
            session_storage_directory
    """
    general_info = get_general_info_for_id("supercontainer_id", supercontainer_id)
    return general_info


### TABLES ###    # noqa: E266
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


def get_cell_segmentation_runs_table(ophys_experiment_id: int) -> pd.DataFrame:
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve
        information on all segmentations run in the
        ophys_cell_segmenatation_runs table for a given experiment

    Arguments:
         ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dataframe --  dataframe with the following columns:
                        id {int}:  9 digit segmentation run id
                        run_number {int}: segmentation run number
                        ophys_experiment_id{int}: 9 digit ophys experiment id
                        current{boolean}: True/False
                                    True: most current segmentation run
                                    False: not the most current segmentation run
                        created_at{timestamp}:
                        updated_at{timestamp}:
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")

    query = '''
    SELECT *

    FROM
    ophys_cell_segmentation_runs

    WHERE
    ophys_experiment_id = {}
    '''.format(ophys_experiment_id)
    return mixin.select(query)


def get_cell_rois_table(ophys_experiment_id: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    ophys_experiment_id : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_segmentation_run_id = get_current_segmentation_run_id(ophys_experiment_id)

    query = '''
    SELECT *

    FROM
    cell_rois

    WHERE
    cell_rois.ophys_cell_segmentation_run_id = {}
    '''.format(current_segmentation_run_id)

    cell_rois_table = mixin.select(query)
    cell_rois_table.rename(columns={"id": "cell_roi_id"})

    return cell_rois_table


def get_ophys_experiments_table(ophys_experiment_id: int) -> pd.DataFrame:
    conditions.validate_id_type(ophys_experiment_id, "ophys_expeirment_id")

    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.workflow_state AS experiment_workflow_state,
    oe.storage_directory AS experiment_storage_directory,
    oe.ophys_session_id,
    structures.acronym AS targeted_structure,
    imaging_depths.depth AS imaging_depth,
    oe.calculated_depth,
    oevbec.visual_behavior_experiment_container_id AS ophys_container_id,
    oe.raw_movie_number_of_frames,
    oe.raw_movie_width,
    oe.raw_movie_height,
    oe.movie_frame_rate_hz,
    oe.ophys_imaging_plane_group_id

    FROM
    ophys_experiments_visual_behavior_experiment_containers oevbec

    JOIN ophys_experiments oe
    ON oe.id = oevbec.ophys_experiment_id

    JOIN structures ON structures.id = oe.targeted_structure_id
    JOIN imaging_depths ON imaging_depths.id = oe.imaging_depth_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)
    ophys_experiments_table = mixin.select(query)
    return ophys_experiments_table


def get_ophys_sessions_table(ophys_session_id: int) -> pd.DataFrame:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")

    query = '''
    SELECT
    os.id AS ophys_session_id,
    os.storage_directory AS session_storage_directory,
    os.workflow_state AS session_workflow_state,
    os.specimen_id,
    os.isi_experiment_id,
    os.parent_session_id,
    os.date_of_acquisition,
    equipment.name AS equipment_name,
    projects.code AS project,
    os.stimulus_name AS session_type,
    os.foraging_id,
    os.stim_delay,
    os.visual_behavior_supercontainer_id AS supercontainer_id

    FROM
    ophys_sessions os

    JOIN equipment ON equipment.id = os.equipment_id

    JOIN projects ON projects.id = os.project_id

    WHERE
    os.id = {}
    '''.format(ophys_session_id)
    ophys_sessions_table = mixin.select(query)

    return ophys_sessions_table


def get_behavior_sessions_table(behavior_session_id: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    behavior_session_id : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")

    query = '''
    SELECT
    bs.id as behavior_session_id,
    bs.ophys_session_id,
    bs.behavior_training_id,
    bs.foraging_id,
    bs.donor_id,
    equipment.name as equipment_name,
    bs.created_at,
    bs.date_of_acquisition

    FROM
    behavior_sessions bs
    JOIN equipment ON equipment.id = bs.equipment_id

    WHERE
    bs.id = {}
    '''.format(behavior_session_id)
    behavior_sessions_table = mixin.select(query)
    return behavior_sessions_table


def get_visual_behavior_experiment_containers_table(ophys_container_id: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    ophys_container_id : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")

    query = '''
    SELECT
    vbec.id AS container_id,
    projects.code AS project,
    vbec.specimen_id,
    vbec.storage_directory AS container_storage_directory,
    vbec.workflow_state AS container_workflow_state

    FROM
    visual_behavior_experiment_containers as vbec
    JOIN projects on projects.id = vbec.project_id
    WHERE
    vbec.id = {}
    '''.format(ophys_container_id)

    visual_behavior_experiment_containers_table = mixin.select(query)
    return visual_behavior_experiment_containers_table


### ROI ###       # noqa: E266


def get_cell_exclusion_labels(ophys_experiment_id: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    ophys_experiment_id : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")

    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    cr.id AS cell_roi_id,
    rel.name AS exclusion_label

    FROM ophys_experiments oe

    JOIN ophys_cell_segmentation_runs ocsr
    ON ocsr.ophys_experiment_id=oe.id
    AND ocsr.current = 't'

    JOIN cell_rois cr ON cr.ophys_cell_segmentation_run_id=ocsr.id
    JOIN cell_rois_roi_exclusion_labels crrel ON crrel.cell_roi_id=cr.id
    JOIN roi_exclusion_labels rel ON rel.id=crrel.roi_exclusion_label_id

    WHERE
    oe.id = {}
    '''.format(ophys_experiment_id)

    return mixin.select(query)


### ________________LIMS__STORAGE__DIRECTORIES_____________________ # noqa: E266

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
        "experiment_storage_directory"
        "session_storage_directory"
        "container_storage_directory"
    """

    query = '''
    oe.storage_directory AS experiment_storage_directory,
    os.storage_directory AS session_storage_directory,
    vbec.storage_directory AS container_storage_directory

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

    WHERE
    {} = {}
    '''.format(OPHYS_ID_TYPES_DICT[id_type]["query_abbrev"], id_number)
    storage_directories_df = mixin.select(query)
    storage_directories_df = correct_storage_directory_filepaths(storage_directories_df)
    return storage_directories_df


def get_experiment_storage_directory(ophys_experiment_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_experiment_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    directories_df = get_storage_directories_for_id("ophys_experiment_id", ophys_experiment_id)
    experiment_path = directories_df["experiment_storage_directory"][0]
    return experiment_path


def get_session_storage_directory(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    directories_df = get_storage_directories_for_id("ophys_session_id", ophys_session_id)
    session_path = directories_df["session_storage_directory"][0]
    return session_path


def get_container_storage_directory(ophys_container_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_container_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    directories_df = get_storage_directories_for_id("ophys_container_id", ophys_container_id)
    container_path = directories_df["container_storage_directory"][0]
    return container_path


def get_crosstalk_storage_directory(ophys_session_id: int) -> str:
    """

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        filepath string to the "crosstalk" folder in the 
    """
    session_directory = get_session_storage_directory(ophys_session_id)
    crosstalk_directory = os.path.join(session_directory, "crosstalk")
    return crosstalk_directory


def get_eye_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    eye_tracking_directory = os.path.join(session_directory, "eye_tracking")
    return eye_tracking_directory


def get_face_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    face_tracking_directory = os.path.join(session_directory, "face_tracking")
    return face_tracking_directory


def get_side_tracking_storage_directory(ophys_session_id: int) -> str:
    session_directory = get_session_storage_directory(ophys_session_id)
    side_tracking_directory = os.path.join(session_directory, "side_tracking")
    return side_tracking_directory


def get_demix_storage_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    demix_directory = os.path.join(experiment_directory, "demix")
    return demix_directory


def get_demix_plots_storage_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    demix_plots_directory = os.path.join(experiment_directory, "demix", "demix_plots")
    return demix_plots_directory


def get_neuropil_subtraction_plots_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    np_subtraction_directory = os.path.join(experiment_directory, "demix", "demix_plots")
    return np_subtraction_directory


def get_experiment_processed_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    processed_directory = os.path.join(experiment_directory, "processed")
    return processed_directory


def get_current_cell_segmentation_run_directory(ophys_experiment_id: int) -> str:
    experiment_directory = get_experiment_storage_directory(ophys_experiment_id)
    segmentation_run_id = get_current_segmentation_run_id(ophys_experiment_id)
    cell_segmentation_run_directory = os.path.join(experiment_directory, "processed", "ophys_cell_segmentation_run_{}".format(segmentation_run_id))
    return cell_segmentation_run_directory



### FILEPATHS FOR WELL KNOWN FILES###      # noqa: E266


VISUAL_BEHAVIOR_WELL_KNOWN_FILE_ATTACHABLE_ID_TYPES = ["'IsiExperiment'",
                                                       "'OphysExperiment'",
                                                       "'OphysCellSegmentationRun'",
                                                       "'OphysSession'",
                                                       "'BehaviorSession'",
                                                       "'VisualBehaviorContainerRun'"]


def get_well_known_file_names_for_attachable_id_type(attachable_id_type: str):
    query = '''
    SELECT DISTINCT
    wkft.name AS well_known_file_name

    FROM
    well_known_files wkf

    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id

    WHERE
    wkf.attachable_type = {}
    '''.format(attachable_id_type)
    well_known_file_names = mixin.select(query)
    well_known_file_names["attachable_id_type"] = attachable_id_type
    return well_known_file_names


def get_well_known_file_names_for_isi_experiments():
    return get_well_known_file_names_for_attachable_id_type("'IsiExperiment'")


def get_well_known_file_names_for_ophys_experiments():
    return get_well_known_file_names_for_attachable_id_type("'OphysExperiment'")


def get_well_known_file_names_for_ophys_cell_segmentation_runs():
    return get_well_known_file_names_for_attachable_id_type("'OphysCellsegmentationRun'")


def get_well_known_file_names_for_ophys_sessions():
    return get_well_known_file_names_for_attachable_id_type("'OphysSession'")


def get_well_known_file_names_for_behavior_sessions():
    return get_well_known_file_names_for_attachable_id_type("'BehaviorSession'")


def get_well_known_file_names_for_ophys_containers():
    return get_well_known_file_names_for_attachable_id_type("'VisualBehaviorContainerRun'")


def get_all_well_known_file_names_for_visual_behavior():
    well_known_file_names = pd.DataFrame()
    for attachable_id_type in VISUAL_BEHAVIOR_WELL_KNOWN_FILE_ATTACHABLE_ID_TYPES:
        well_known_file_names = well_known_file_names.append(get_well_known_file_names_for_attachable_id_type(attachable_id_type))
    return well_known_file_names


def get_well_known_file_realdict(wellKnownFileName, attachable_id):
    """a generalized function to get the filepath for well known files when
    given the wellKnownFile name and the correct attchable_id

    Parameters
    ----------
    wellKnownFileName : string
        well known file name, e.g. "BehaviorOphysNwb",'DemixedTracesFile' etc.
    attachable_id : int
        the id that the well known file can be identified by. Potential attachable_id
        types include:
            ophys_experiment_id
            ophys_session_id
            OphysCellSegmentationRun etc.

        refer to the wellKnownFilesDict and see "attachable_id_type" key to get the attachable
        id type for each wellKnownFile.

    Returns
    -------
    filepath string
        the filepath for the well known file
    """

    query = '''
    SELECT
    wkf.storage_directory || wkf.filename
    AS filepath

    FROM
    well_known_files wkf

    JOIN well_known_file_types wkft
    ON wkft.id = wkf.well_known_file_type_id

    WHERE
    wkft.name = {}
    AND wkf.attachable_id = {}
    '''.format(wellKnownFileName, attachable_id)

    RealDict_object = mixin.select(query)
    return RealDict_object


def get_filepath_from_realdict_object(realdict_object):
    """takes a RealDictRow object returned when loading well known files
       from lims and parses it to return the filepath to the well known file.

    Args:
        wkf_realdict_object ([type]): [description]

    Returns:
        filepath: [description]
    """
    filepath = realdict_object['filepath'][0]
    filepath = utils.correct_filepath(filepath)
    return filepath


def get_well_known_file_path(wellKnownFileName: str, attachable_id: int) -> str:
    RealDict_object = get_well_known_file_realdict(wellKnownFileName, attachable_id)
    # filepath works for all operating systems
    filepath = get_filepath_from_realdict_object(RealDict_object)
    return filepath


# FOR ISI EXPERIMENT ID
def get_isi_experiment_filepath(isi_experiment_id: int) -> str:
    conditions.validate_id_type(isi_experiment_id, "isi_experiment_id")
    filepath = get_well_known_file_path("'IsiExperiment'", isi_experiment_id)
    return filepath


def get_processed_isi_experiment_filepath(isi_experiment_id: int) -> str:
    conditions.validate_id_type(isi_experiment_id, "isi_experiment_id")
    filepath = get_well_known_file_path("'IsiProcessed'", isi_experiment_id)
    return filepath


def get_isi_NWB_filepath(isi_experiment_id: int) -> str:
    conditions.validate_id_type(isi_experiment_id, "isi_experiment_id")
    filepath = get_well_known_file_path("'NWBISI'", isi_experiment_id)
    return filepath


# FOR OPHYS EXPERIMENT ID


def get_ophys_NWB_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id)
    filepath = get_well_known_file_path("'NWBOphys'", ophys_experiment_id)
    return filepath


def get_ophys_extracted_traces_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id)
    filepath = get_well_known_file_path("'NWBOphys'", ophys_experiment_id)
    return filepath


def get_BehaviorOphys_NWB_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'BehaviorOphysNwb'", ophys_experiment_id)
    return filepath


def get_demixed_traces_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'DemixedTracesFile'", ophys_experiment_id)
    return filepath


def get_motion_corrected_movie_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'MotionCorrectedImageStack'", ophys_experiment_id)
    return filepath


def get_neuropil_correction_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'NeuropilCorrection'", ophys_experiment_id)
    return filepath


def get_average_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysAverageIntensityProjectionImage'", ophys_experiment_id)
    return filepath


def get_dff_traces_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysDffTraceFile'", ophys_experiment_id)
    return filepath


def get_event_trace_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysEventTraceFile'", ophys_experiment_id)
    return filepath


def get_extracted_traces_input_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysExtractedTracesInputJson'", ophys_experiment_id)
    return filepath


def get_extracted_traces_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysExtractedTraces'", ophys_experiment_id)
    return filepath


def get_motion_preview_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysMotionPreview'", ophys_experiment_id)
    return filepath


def get_motion_xy_offset_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysMotionXyOffsetData'", ophys_experiment_id)
    return filepath


def get_neuropil_traces_filepath(ophys_experiment_id: int) -> str:
    """uses well known file system to query lims and get the directory
    and filename for the neuropil_traces.h5 for a given ophys experiment

    Parameters
    ----------
    ophys_experiment_id : int
        a unique identifier for an ophys experiment

    Returns
    -------
    filepath string
        AI network filepath string for neuropil_traces.h5 for a
        given ophys experiment.
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysNeuropilTraces'", ophys_experiment_id)
    return filepath


def get_ophys_registration_summary_image_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysRegistrationSummaryImage'", ophys_experiment_id)
    return filepath


def get_roi_traces_filepath(ophys_experiment_id: int) -> str:
    """uses well known file system to query lims and get the directory
    and filename for the roi_traces.h5 for a given ophys experiment

    Parameters
    ----------
    ophys_experiment_id : int
        unique identifier for an ophys experiment

    Returns
    -------
    filepath
        filepath to the roi_traces.h5 for an ophys experiment
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysRoiTraces'", ophys_experiment_id)
    return filepath


def get_cell_roi_metrics_file_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'OphysExperimentCellRoiMetricsFile'", ophys_experiment_id)
    return filepath


def get_time_syncronization_filepath(ophys_experiment_id: int) -> str:
    """for scientifica experiments only (CAM2P.3, CAM2P.4, CAM2P.5)

    Parameters
    ----------
    ophys_experiment_id : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    conditions.validate_microscope_type(ophys_session_id, "Scientifica")
    filepath = get_well_known_file_path("'OphysTimeSynchronization'", ophys_experiment_id)
    return filepath


def get_observatory_events_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    filepath = get_well_known_file_path("'ObservatoryEventsFile'", ophys_experiment_id)
    return filepath


## FOR ophys_cell_segmentation_run_id via ophys_experiment_id             # noqa: E266


def get_ophys_suite2p_rois_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSuite2pRois'", current_seg_id)
    return filepath


def get_segmentation_objects_filepath(ophys_experiment_id: int) -> str:
    """use SQL and the LIMS well known file system to get the
       location information for the objectlist.txt file for a
       given cell segmentation run

    Arguments:
        segmentation_run_id {int} -- 9 digit segmentation run id

    Returns:
        list -- list with storage directory and filename
    """
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationObjects'", current_seg_id)
    return filepath


def get_lo_segmentation_mask_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysLoSegmentationMaskData'", current_seg_id)
    return filepath


def get_segmentation_mask_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationMaskData'", current_seg_id)
    return filepath


def get_segmentation_mask_image_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationMaskImage'", current_seg_id)
    return filepath


def get_ave_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysAverageIntensityProjectionImage'", current_seg_id)
    return filepath


def get_max_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    conditions.validate_id_type(ophys_experiment_id, "ophys_experiment_id")
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysMaxIntImage'", current_seg_id)
    return filepath


## for ophys_session_id ##              # noqa: E266


def get_timeseries_ini_filepath(ophys_session_id: int) -> str:
    """use SQL and the LIMS well known file system to get the
       timeseries_XYT.ini file for a given ophys session.
       Notes: only Scientifca microscopes produce timeseries.ini files

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    conditions.validate_microscope_type(ophys_session_id, "Scientifica")
    filepath = get_well_known_file_path("'SciVivoMetadata'", ophys_session_id)
    return filepath


def get_stimulus_pkl_filepath_for_ophys_session(ophys_session_id: int) -> str:
    """use SQL and the LIMS well known file system to get the
        session pkl file information for a given
        ophys_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'StimulusPickle'", ophys_session_id)
    return filepath


def get_session_h5_filepath(ophys_session_id: int) -> str:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'OphysRigSync'", ophys_session_id)
    return filepath


def get_behavior_avi_filepath(ophys_session_id: int) -> str:
    """use SQL and the LIMS well known file system to get the network
    video-0.avi (behavior video) filepath for a given ophys_session_id

    Parameters
    ----------
    ophys_session_id : int
        unique identifier for ophys session

    Returns
    -------
    string
        network filepath as a string
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'RawBehaviorTrackingVideo'", ophys_session_id)
    return filepath


def get_behavior_h5_filepath(ophys_session_id: int) -> str:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    RealDict_object = get_well_known_file_realdict("'RawBehaviorTrackingVideo'", ophys_session_id)
    filepath = RealDict_object['filepath'][0]
    filepath = filepath.replace('avi', 'h5')
    filepath = utils.correct_filepath(filepath)
    return filepath


def get_eye_tracking_avi_filepath(ophys_session_id: int) -> str:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'RawEyeTrackingVideo'", ophys_session_id)
    return filepath


def get_eye_tracking_h5_filepath(ophys_session_id: int) -> str:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    RealDict_object = get_well_known_file_realdict("'RawEyeTrackingVideo'", ophys_session_id)
    filepath = RealDict_object['filepath'][0]
    filepath = filepath.replace('avi', 'h5')
    filepath = utils.correct_filepath(filepath)
    return filepath


def get_eye_screen_mapping_filepath(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeScreenMapping'", ophys_session_id)
    return filepath


def get_eyetracking_corneal_reflection(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeTracking Corneal Reflection'", ophys_session_id)
    return filepath


def get_eyetracking_pupil_filepath(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeTracking Pupil'", ophys_session_id)
    return filepath


def get_ellipse_filepath(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeTracking Ellipses'", ophys_session_id)
    return filepath


def get_platform_json_filepath(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'OphysPlatformJson'", ophys_session_id)
    return filepath


def get_screen_mapping_h5_filepath(ophys_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeDlcScreenMapping'", ophys_session_id)
    return filepath


def get_deepcut_h5_filepath(ophys_session_id: int) -> str:
    conditions.validate_id_type(ophys_session_id, "ophys_session_id")
    filepath = get_well_known_file_path("'EyeDlcOutputFile'", ophys_session_id)
    return filepath


## for behavior_session_id ##              # noqa: E266


def get_behavior_NWB_filepath(behavior_session_id: int) -> str:
    """_summary_

    Parameters
    ----------
    behavior_session_id : int
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")
    filepath = get_well_known_file_path("'BehaviorNwb'", behavior_session_id)
    return filepath


def get_stimulus_pkl_filepath_for_behavior_session(behavior_session_id: int) -> str:
    """use SQL and the LIMS well known file system to get the
        session pkl file information for a given
        behavior_session_id

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session ID

    Returns:
        [type] -- [description]
    """
    conditions.validate_id_type(behavior_session_id, "behavior_session_id")
    filepath = get_well_known_file_path("'StimulusPickle'", behavior_session_id)
    return filepath


## for ophys_container_id ##              # noqa: E266


def get_nway_cell_matching_output_filepath(ophys_container_id: int) -> str:
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")
    current_container_run_id = get_current_container_run_id_for_ophys_container_id(ophys_container_id)
    filepath = get_well_known_file_path("'OphysNwayCellMatchingOutput'", current_container_run_id)
    return filepath


def get_cell_matching_output_filepath(ophys_container_id: int) -> str:
    conditions.validate_id_type(ophys_container_id, "ophys_container_id")
    current_container_run_id = get_current_container_run_id_for_ophys_container_id(ophys_container_id)
    filepath = get_well_known_file_path("'OphysCellMatchingOutput'", current_container_run_id)
    return filepath


### WELL KNOWN FILES ###      # noqa: E303, E266


def load_demixed_traces_array(ophys_experiment_id: int) -> array:
    """use SQL and the LIMS well known file system to find and load
            "demixed_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            demixed_traces_array -- mxn array where m = rois and n = time
        """
    filepath = get_demixed_traces_filepath(ophys_experiment_id)
    demix_file = h5py.File(filepath, 'r')
    demixed_traces_array = np.asarray(demix_file['data'])
    demix_file.close()
    return demixed_traces_array


def load_neuropil_traces_array(ophys_experiment_id: int) -> array:
    """use SQL and the LIMS well known file system to find and load
            "neuropil_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            neuropil_traces_array -- mxn array where m = rois and n = time
        """
    filepath = get_neuropil_traces_filepath(ophys_experiment_id)
    f = h5py.File(filepath, 'r')
    neuropil_traces_array = np.asarray(f['data'])
    f.close()
    return neuropil_traces_array


def load_motion_corrected_movie(ophys_experiment_id, frame_limit=None):
    """uses well known file system to get motion_corrected_movie.h5
        filepath and then loads the h5 file with h5py function.
        Gets the motion corrected movie array in the h5 from the only
        datastream/key 'data' and returns it.

    Parameters
    ----------
    ophys_experiment_id : [type]
        [description]

    Returns
    -------
    HDF5 dataset
        3D array-like (z, y, x) dimensions
                        z: timeseries/frame number
                        y: single frame y axis
                        x: single frame x axis
    """
    filepath = get_motion_corrected_movie_filepath(ophys_experiment_id)
    motion_corrected_movie_file = h5py.File(filepath, 'r')
    if not frame_limit:
        motion_corrected_movie = motion_corrected_movie_file['data']
    else:
        motion_corrected_movie = motion_corrected_movie_file['data'][0:frame_limit]

    return motion_corrected_movie


def load_rigid_motion_transform(ophys_experiment_id):
    """use SQL and the LIMS well known file system to locate
        and load the rigid_motion_transform.csv file for
        a given ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dataframe -- dataframe with the following columns:
                        "framenumber":
                                  "x":
                                  "y":
                        "correlation":
                           "kalman_x":
                           "kalman_y":
    """
    filepath = get_motion_xy_offset_filepath(ophys_experiment_id)
    rigid_motion_transform_df = pd.read_csv(filepath)
    return rigid_motion_transform_df


def load_objectlist(ophys_experiment_id):
    """loads the objectlist.txt file for the current segmentation run,
       then "cleans" the column names and returns a dataframe

    Arguments:
        experiment_id {[int]} -- 9 digit unique identifier for the experiment

    Returns:
        dataframe -- dataframe with the following columns:
                    (from http://confluence.corp.alleninstitute.org/display/IT/Ophys+Segmentation)
            trace_index:The index to the corresponding trace plot computed
                        (order of the computed traces in file _somaneuropiltraces.h5)
            center_x: The x coordinate of the centroid of the object in image pixels
            center_y:The y coordinate of the centroid of the object in image pixels
            frame_of_max_intensity_masks_file: The frame the object mask is in maxInt_masks2.tif
            frame_of_enhanced_movie: The frame in the movie enhimgseq.tif that best shows the object
            layer_of_max_intensity_file: The layer of the maxInt file where the object can be seen
            bbox_min_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_min_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_max_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            bbox_max_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            area: Total area of the segmented object
            ellipseness: The "ellipticalness" of the object, i.e. length of
                         long axis divided by length of short axis
            compactness: Compactness :  perimeter^2 divided by area
            exclude_code: A non-zero value indicates the object should be excluded from
                         further analysis.  Based on measurements in objectlist.txt
                        0 = not excluded
                        1 = doublet cell
                        2 = boundary cell
                        Others = classified as not complete soma, apical dendrite, ....
            mean_intensity: Correlates with delta F/F.  Mean brightness of the object
            mean_enhanced_intensity: Mean enhanced brightness of the object
            max_intensity: Max brightness of the object
            max_enhanced_intensity: Max enhanced brightness of the object
            intensity_ratio: (max_enhanced_intensity - mean_enhanced_intensity)
                             / mean_enhanced_intensity, for detecting dendrite objects
            soma_minus_np_mean: mean of (soma trace – its neuropil trace)
            soma_minus_np_std: 1-sided stdv of (soma trace – its neuropil trace)
            sig_active_frames_2_5:# frames with significant detected activity (spiking):
                                  Sum ( soma_trace > (np_trace + Snpoffsetmean+ 2.5 * Snpoffsetstdv)
                                  trace_38_soma.png  See example traces attached.
            sig_active_frames_4: # frames with significant detected activity (spiking):
                                 Sum ( soma_trace > (np_trace + Snpoffsetmean+ 4.0 * Snpoffsetstdv)
            overlap_count: 	Number of other objects the object overlaps with
            percent_area_overlap: the percentage of total object area that overlaps with other objects
            overlap_obj0_index: The index of the first object with which this object overlaps
            overlap_obj1_index: The index of the second object with which this object overlaps
            soma_obj0_overlap_trace_corr: trace correlation coefficient
                                          between soma and overlap soma0
                                          (-1.0:  excluded cell,  0.0 : NA)
            soma_obj1_overlap_trace_corr: trace correlation coefficient
                                          between soma and overlap soma1
    """
    filepath = get_segmentation_objects_filepath(ophys_experiment_id)
    objectlist_dataframe = pd.read_csv(filepath)
    objectlist_dataframe = update_objectlist_column_labels(objectlist_dataframe)
    return objectlist_dataframe


def update_objectlist_column_labels(objectlist_df: pd.DataFrame) -> pd.DataFrame:
    """take the roi metrics from the objectlist.txt file and renames
       them to be more explicit and descriptive.
        -removes single blank space at the beginning of column names
        -enforced naming scheme(no camel case, added _)
        -renamed columns to be more descriptive/reflect contents of
         column

    Parameters
    ----------
    objectlist_df : pd.DataFrame
        dataframe directly from the load_objectlist function

    Returns
    -------
    pd.DataFrame
    """
    objectlist_df = objectlist_df.rename(index=str,
                                         columns={' traceindex': 'trace_index',
                                                  ' cx': 'center_x',
                                                  ' cy': 'center_y',
                                                  ' mask2Frame': 'frame_of_max_intensity_masks_file',
                                                  ' frame': 'frame_of_enhanced_movie',
                                                  ' object': 'layer_of_max_intensity_file',
                                                  ' minx': 'bbox_min_x',
                                                  ' miny': 'bbox_min_y',
                                                  ' maxx': 'bbox_max_x',
                                                  ' maxy': 'bbox_max_y',
                                                  ' area': 'area',
                                                  ' shape0': 'ellipseness',
                                                  ' shape1': 'compactness',
                                                  ' eXcluded': 'exclude_code',
                                                  ' meanInt0': 'mean_intensity',
                                                  ' meanInt1': 'mean_enhanced_intensity',
                                                  ' maxInt0': 'max_intensity',
                                                  ' maxInt1': 'max_enhanced_intensity',
                                                  ' maxMeanRatio': 'intensity_ratio',
                                                  ' snpoffsetmean': 'soma_minus_np_mean',
                                                  ' snpoffsetstdv': 'soma_minus_np_std',
                                                  ' act2': 'sig_active_frames_2_5',
                                                  ' act3': 'sig_active_frames_4',
                                                  ' OvlpCount': 'overlap_count',
                                                  ' OvlpAreaPer': 'percent_area_overlap',
                                                  ' OvlpObj0': 'overlap_obj0_index',
                                                  ' OvlpObj1': 'overlap_obj1_index',
                                                  ' corcoef0': 'soma_obj0_overlap_trace_corr',
                                                  ' corcoef1': 'soma_obj1_overlap_trace_corr'})
    return objectlist_df


def get_dff_traces_for_roi(cell_roi_id):
    '''
    gets dff trace for desired cell_roi_id
    gets data directly from well_known_file h5 file, which is faster than opening the BehaviorOphysExperiment

    Parameters:
    -----------
    cell_roi_id: int
        desired cell_roi_id

    Returns:
    --------
    array
        1D array of dff values for the desired cell_roi_id
    '''
    # get associated experiment_id
    ophys_experiment_id = get_ophys_experiment_id_for_cell_roi_id(cell_roi_id)

    # get roi_traces filepath
    roi_traces_filename = get_well_known_file_path(ophys_experiment_id, 'OphysExperiment', 'OphysDffTraceFile')

    # open file for reading
    with h5py.File(roi_traces_filename, "r") as f:

        # get index for associated roi
        roi_ids = [roi_name.decode("utf-8") for roi_name in f.get('roi_names')]
        roi_index = roi_ids.index(str(cell_roi_id))

        # get corresponding data
        dff_data = f.get('data')
        dff = dff_data[roi_index]

    return dff
