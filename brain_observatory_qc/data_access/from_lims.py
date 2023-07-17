import os
import h5py
import warnings
import numpy as np
import pandas as pd
from array import array
from psycopg2 import extras
import json


from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP


import brain_observatory_qc.data_access.utilities as utils
import brain_observatory_qc.data_access.from_lims_utilities as lims_utils


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


def _get_psql_dict_cursor():
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
#           GET ASSOCIATED LIMS IDS
#
#####################################################################


############################
#     mouse ids
############################

def get_donor_id_for_specimen_id(specimen_id: int) -> int:
    lims_utils.validate_LIMS_id_type("specimen_id", specimen_id)
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
    lims_utils.validate_LIMS_id_type("donor_id", donor_id)
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


def get_imaging_ids_for_mouse_id(mouse_id: int) -> pd.DataFrame:
    """get imaging information for a given mouse

    Parameters
    ----------
    mouse_id : int
        6 digit mouse-id, also know as labtracks id,
        or the external_specimen_name in the lims
        specimens table

    Returns
    -------
    pd.DataFrame
        table with experiment and session information. Table columns:
            "ophys_experiment_id"
            "experiment_workflow_state"
            "experiment_storage_directory"
            "ophys_session_id"
            "session_workflow_state"
            "session_storage_directory"

    """
    query = '''
    SELECT
    oe.id AS ophys_experiment_id,
    oe.workflow_state AS experiment_workflow_state,
    oe.storage_directory AS experiment_storage_directory,
    os.id AS ophys_session_id,
    os.workflow_state AS session_workflow_state,
    os.storage_directory AS session_storage_directory

    FROM
    ophys_experiments oe

    JOIN ophys_sessions os
    ON os.id = oe.ophys_session_id

    JOIN specimens
    ON specimens.id = os.specimen_id

    WHERE specimens.external_specimen_name = '{}'
    '''.format(mouse_id)
    mouse_imaging_table = mixin.select(query)
    return mouse_imaging_table


############################
#     cell/ROI IDS
############################


def get_ophys_experiment_id_for_cell_roi_id(cell_roi_id: int) -> int:
    """
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
    """
    lims_utils.validate_LIMS_id_type("cell_roi_id", cell_roi_id)
    query = '''
    SELECT ophys_experiment_id
    FROM cell_rois
    WHERE id = {}
    '''.format(cell_roi_id)
    ophys_experiment_id = mixin.select(query)
    return ophys_experiment_id


############################
#     ophys experiment ID
############################

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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    lims_utils.validate_microscope_type("Mesoscope", ophys_session_id)
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
    all_ids = lims_utils.get_all_imaging_ids_for_imaging_id("ophys_experiment_id", ophys_experiment_id)
    return all_ids


def get_genotype_for_ophys_experiment_id(ophys_experiment_id: int):
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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


############################
#    ophys session ID
############################

def get_ophys_experiment_ids_for_ophys_session_id(ophys_session_id: int) -> pd.DataFrame:
    """uses an sqlite to query lims2 database. Gets the ophys_experiment_id
       for a given ophys_session_id from the ophys_experiments table in lims.

       note: number of experiments per session will vary depending on the
       rig it was collected on and the project it was collected for.
    """
    lims_utils.validate_LIMS_id_type('ophys_session_id', ophys_session_id)
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
    lims_utils.validate_LIMS_id_type('ophys_session_id', ophys_session_id)
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
    lims_utils.validate_LIMS_id_type('ophys_session_id', ophys_session_id)
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
    lims_utils.validate_LIMS_id_type('ophys_session_id', ophys_session_id)
    lims_utils.validate_microscope_type("Mesoscope", ophys_session_id)
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
    all_ids = lims_utils.get_all_imaging_ids_for_imaging_id("ophys_session_id", ophys_session_id)
    return all_ids


############################
#    behavior session id
############################


def get_ophys_experiment_ids_for_behavior_session_id(behavior_session_id: int):
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)

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
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)

    query = '''
    SELECT
    ophys_session_id

    FROM behavior_sessions

    WHERE id = {}'''.format(behavior_session_id)

    ophys_session_id = mixin.select(query)
    ophys_session_id = ophys_session_id["ophys_session_id"][0]
    return ophys_session_id


def get_ophys_container_ids_for_behavior_session_id(behavior_session_id: int):
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)

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
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)
    ophys_session_id = get_ophys_session_id_for_behavior_session_id(behavior_session_id)
    lims_utils.validate_microscope_type("Mesoscope", ophys_session_id)

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
    all_ids = lims_utils.get_all_imaging_ids_for_imaging_id("behavior_session_id", behavior_session_id)
    return all_ids


############################
#     ophys container id
############################

def get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id: int) -> pd.DataFrame:
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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)

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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)
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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)

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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)

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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)
    ophys_session_id = get_ophys_session_ids_for_ophys_container_id(ophys_container_id)["ophys_session_id"][0]
    lims_utils.validate_microscope_type("Mesoscope", ophys_session_id)
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
    all_ids = lims_utils.get_all_imaging_ids_for_imaging_id("ophys_container_id", ophys_container_id)
    return all_ids


############################
#     supercontainer id
############################


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
    lims_utils.validate_LIMS_id_type("supercontainer_id", supercontainer_id)

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
    lims_utils.validate_LIMS_id_type("supercontainer_id", supercontainer_id)
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
    """_summary_

    Parameters
    ----------
    supercontainer_id : int
        _description_

    Returns
    -------
    int
        _description_
    """
    lims_utils.validate_LIMS_id_type("supercontainer_id", supercontainer_id)

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
    lims_utils.validate_LIMS_id_type("supercontainer_id", supercontainer_id)
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
    all_ids = lims_utils.get_all_imaging_ids_for_imaging_id("supercontainer_id", supercontainer_id)
    return all_ids


#####################################################################
#
#           GENERAL INFO FOR IMAGING ID
#
#####################################################################


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
    general_info = lims_utils.get_general_info_for_LIMS_imaging_id("ophys_experiment_id", ophys_experiment_id)
    return general_info


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
    general_info = lims_utils.get_general_info_for_LIMS_imaging_id("ophys_session_id", ophys_session_id)
    return general_info


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
    general_info = lims_utils.get_general_info_for_LIMS_imgaing_id("behavior_session_id", behavior_session_id)
    return general_info


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
    general_info = lims_utils.get_general_info_for_LIMS_imaging_id("ophys_container_id", ophys_container_id)
    return general_info


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
    general_info = lims_utils.get_general_info_for_LIMS_imaging_id("supercontainer_id", supercontainer_id)
    return general_info


#####################################################################
#
#           LIMS TABLES
#
#####################################################################

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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)

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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)

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
    """_summary_

    Parameters
    ----------
    ophys_session_id : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)

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
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)

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
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)

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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)

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


#####################################################################
#
#           FILEPATHS FOR WELL KNOWN FILES
#
#####################################################################

VISUAL_BEHAVIOR_WELL_KNOWN_FILE_ATTACHABLE_ID_TYPES = ["'IsiExperiment'",
                                                       "'OphysExperiment'",
                                                       "'OphysCellSegmentationRun'",
                                                       "'OphysSession'",
                                                       "'BehaviorSession'",
                                                       "'VisualBehaviorContainerRun'",
                                                       "'Specimen'"]


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


def get_well_known_file_names_for_specimen_id():
    return get_well_known_file_names_for_attachable_id_type("'Specimen")


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


############################
#     for isi_experiment_id
############################

def get_isi_experiment_filepath(isi_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("isi_experiment_id", isi_experiment_id)
    filepath = get_well_known_file_path("'IsiExperiment'", isi_experiment_id)
    return filepath


def get_processed_isi_experiment_filepath(isi_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("isi_experiment_id", isi_experiment_id)
    filepath = get_well_known_file_path("'IsiProcessed'", isi_experiment_id)
    return filepath


def get_isi_NWB_filepath(isi_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("isi_experiment_id", isi_experiment_id)
    filepath = get_well_known_file_path("'NWBISI'", isi_experiment_id)
    return filepath


############################
#     for specimen_id
############################

def get_cortical_zstack_filepaths(specimen_id: int) -> pd.DataFrame:
    lims_utils.validate_LIMS_id_type("specimen_id", specimen_id)
    filepaths = get_well_known_file_realdict("'cortical_z_stacks'", specimen_id)
    utils.correct_dataframe_filepath(filepaths, "filepath")
    return filepaths


############################
#     for ophys_experiment_id
############################


def get_ophys_NWB_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'NWBOphys'", ophys_experiment_id)
    return filepath


def get_ophys_extracted_traces_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'NWBOphys'", ophys_experiment_id)
    return filepath


def get_BehaviorOphys_NWB_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'BehaviorOphysNwb'", ophys_experiment_id)
    return filepath


def get_demixed_traces_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'DemixedTracesFile'", ophys_experiment_id)
    return filepath


def get_motion_corrected_movie_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'MotionCorrectedImageStack'", ophys_experiment_id)
    return filepath


def get_neuropil_correction_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'NeuropilCorrection'", ophys_experiment_id)
    return filepath


def get_average_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysAverageIntensityProjectionImage'", ophys_experiment_id)
    return filepath


def get_dff_traces_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysDffTraceFile'", ophys_experiment_id)
    return filepath


def get_event_trace_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysEventTraceFile'", ophys_experiment_id)
    return filepath


def get_extracted_traces_input_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysExtractedTracesInputJson'", ophys_experiment_id)
    return filepath


def get_extracted_traces_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysExtractedTraces'", ophys_experiment_id)
    return filepath


def get_motion_preview_filepath(ophys_experiment_id):
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysMotionPreview'", ophys_experiment_id)
    return filepath


def get_motion_xy_offset_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysNeuropilTraces'", ophys_experiment_id)
    return filepath


def get_ophys_registration_summary_image_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysRoiTraces'", ophys_experiment_id)
    return filepath


def get_cell_roi_metrics_file_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'OphysExperimentCellRoiMetricsFile'", ophys_experiment_id)
    return filepath


def get_time_syncronization_filepath(ophys_experiment_id: int) -> str:
    """for scientifica experiments only (CAM2P.3, CAM2P.4, CAM2P.5)

    Parameters
    ----------
    ophys_experiment_id : int
        [description]

    Returns
    -------
    str filepath
        [description]
    """
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    lims_utils.validate_microscope_type("Scientifica", ophys_session_id)
    filepath = get_well_known_file_path("'OphysTimeSynchronization'", ophys_experiment_id)
    return filepath


def get_observatory_events_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    filepath = get_well_known_file_path("'ObservatoryEventsFile'", ophys_experiment_id)
    return filepath


############################
#     for ophys_cell_segmentation_run_id via ophys_experiment_id
############################


def get_ophys_suite2p_rois_filepath(ophys_experiment_id: int) -> str:
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
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
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationObjects'", current_seg_id)
    return filepath


def get_lo_segmentation_mask_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysLoSegmentationMaskData'", current_seg_id)
    return filepath


def get_segmentation_mask_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationMaskData'", current_seg_id)
    return filepath


def get_segmentation_mask_image_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysSegmentationMaskImage'", current_seg_id)
    return filepath


def get_ave_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysAverageIntensityProjectionImage'", current_seg_id)
    return filepath


def get_max_intensity_projection_filepath(ophys_experiment_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_experiment_id", ophys_experiment_id)
    current_seg_id = int(get_current_segmentation_run_id(ophys_experiment_id))
    filepath = get_well_known_file_path("'OphysMaxIntImage'", current_seg_id)
    return filepath


############################
#     for ophys_session_id
############################


def get_timeseries_ini_filepath(ophys_session_id: int) -> str:
    """use SQL and the LIMS well known file system to get the
       timeseries_XYT.ini file for a given ophys session.
       Notes: only Scientifca microscopes produce timeseries.ini files

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """

    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    lims_utils.validate_microscope_type("Scientifica", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    filepath = get_well_known_file_path("'StimulusPickle'", ophys_session_id)
    return filepath


def get_session_h5_filepath(ophys_session_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    filepath = get_well_known_file_path("'RawBehaviorTrackingVideo'", ophys_session_id)
    return filepath


def get_behavior_h5_filepath(ophys_session_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    RealDict_object = get_well_known_file_realdict("'RawBehaviorTrackingVideo'", ophys_session_id)
    filepath = RealDict_object['filepath'][0]
    filepath = filepath.replace('avi', 'h5')
    filepath = utils.correct_filepath(filepath)
    return filepath


def get_eye_tracking_avi_filepath(ophys_session_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    filepath = get_well_known_file_path("'RawEyeTrackingVideo'", ophys_session_id)
    return filepath


def get_eye_tracking_h5_filepath(ophys_session_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
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
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    filepath = get_well_known_file_path("'EyeDlcScreenMapping'", ophys_session_id)
    return filepath


def get_deepcut_h5_filepath(ophys_session_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_session_id", ophys_session_id)
    filepath = get_well_known_file_path("'EyeDlcOutputFile'", ophys_session_id)
    return filepath


############################
#     for behavior_session_id
############################


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
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)
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
    lims_utils.validate_LIMS_id_type("behavior_session_id", behavior_session_id)
    filepath = get_well_known_file_path("'StimulusPickle'", behavior_session_id)
    return filepath


############################
#     for ophys_container_id
############################


def get_nway_cell_matching_output_filepath(ophys_container_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)
    current_container_run_id = get_current_container_run_id_for_ophys_container_id(ophys_container_id)
    filepath = get_well_known_file_path("'OphysNwayCellMatchingOutput'", current_container_run_id)
    return filepath


def get_cell_matching_output_filepath(ophys_container_id: int) -> str:
    lims_utils.validate_LIMS_id_type("ophys_container_id", ophys_container_id)
    current_container_run_id = get_current_container_run_id_for_ophys_container_id(ophys_container_id)
    filepath = get_well_known_file_path("'OphysCellMatchingOutput'", current_container_run_id)
    return filepath


#####################################################################
#
#          LOAD WELL KNOWN FILES
#
#####################################################################


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


#####################################################################
#
#           2p Imaging Info
#
#####################################################################


def get_platform_frame_rate_for_oeid(oeid):
    """get the frame rate for a given oeid
    Parameters
    ----------
    oeid : int
        unique ophys experiment id

    Returns
    -------
    float
        frame rate for the oeid
    """
    osid = get_ophys_session_id_for_ophys_experiment_id(oeid)
    return get_platform_frame_rate_for_osid(osid)


def get_platform_frame_rate_for_osid(osid):
    """get the frame rate for a given osid
    Parameters
    ----------
    osid : int
        unique ophys session id

    Returns
    -------
    float
        frame rate for the osid
    """
    json_fn = get_session_h5_filepath(osid).parent / f'{osid}_platform.json'
    with open(json_fn, 'r') as f:
        platform = json.load(f)
    frame_rates = []
    for plane in platform['imaging_plane_groups']:
        frame_rates.append(plane['acquisition_framerate_Hz'])
    assert (np.array(frame_rates) - frame_rates[0]).any() == 0
    frame_rate = frame_rates[0]
    return frame_rate
