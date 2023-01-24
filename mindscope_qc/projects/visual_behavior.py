import numpy as np
import pandas as pd
import warnings

import mindscope_qc.utilities.pre_post_conditions as conditions

#####################################################################
#
#           		PROJECT CONSTANTS
#
#####################################################################

SESSION_TYPES = [
    "OPHYS_0_images_A_habituation",
    "OPHYS_0_images_B_habituation",
    "OPHYS_0_images_G_habituation",
    "OPHYS_1_images_A",
    "OPHYS_1_images_B",
    "OPHYS_1_images_G",
    "OPHYS_2_images_A_passive",
    "OPHYS_2_images_B_passive",
    "OPHYS_2_images_G_passive",
    "OPHYS_3_images_A",
    "OPHYS_3_images_B",
    "OPHYS_3_images_G",
    "OPHYS_4_images_A",
    "OPHYS_4_images_B",
    "OPHYS_4_images_H",
    "OPHYS_5_images_A_passive",
    "OPHYS_5_images_B_passive",
    "OPHYS_5_images_H_passive",
    "OPHYS_6_images_A",
    "OPHYS_6_images_B",
    "OPHYS_6_images_H",
    "TRAINING_0_gratings_autorewards_15min",
    "TRAINING_1_gratings",
    "TRAINING_2_gratings_flashed",
    "TRAINING_3_images_A_10uL_reward",
    "TRAINING_3_images_B_10uL_reward",
    "TRAINING_3_images_G_10uL_reward",
    "TRAINING_4_images_A_handoff_lapsed",
    "TRAINING_4_images_A_handoff_ready",
    "TRAINING_4_images_A_training",
    "TRAINING_4_images_B_training",
    "TRAINING_4_images_G_training",
    "TRAINING_5_images_A_epilogue",
    "TRAINING_5_images_A_handoff_lapsed",
    "TRAINING_5_images_A_handoff_ready",
    "TRAINING_5_images_B_epilogue",
    "TRAINING_5_images_B_handoff_lapsed",
    "TRAINING_5_images_B_handoff_ready",
    "TRAINING_5_images_G_epilogue",
    "TRAINING_5_images_G_handoff_lapsed",
    "TRAINING_5_images_G_handoff_ready"
]

PROJECT_CODES = [
    'VisualBehavior',
    'VisualBehaviorTask1B',
    'VisualBehaviorMultiscope',
    'VisualBehaviorMultiscope4areasx2d']

NUM_STRUCTURES_DICT = {
    'VisualBehavior': 1,
    'VisualBehaviorTask1B': 1,
    'VisualBehaviorMultiscope': 2,
    'VisualBehaviorMultiscope4areasx2d': 4}


NUM_DEPTHS_DICT = {
    'VisualBehavior': 1,
    'VisualBehaviorTask1B': 1,
    'VisualBehaviorMultiscope':4,
    'VisualBehaviorMultiscope4areasx2d':2}

OPHYS_BEHAV_TYPE_DICT = {
    'active_behavior': ["0", "1", "3", "4", "6"],
    'passive_viewing': ["2", "5", "7"]
}

#####################################################################
#
#           		PARSE METADATA
#
#####################################################################


############################
#
#     from session type
#
############################

def split_session_type_elements(session_type: str) ->tuple:
    element_0: session_type.split('_')[0]
    element_1: session_type.split('_')[1]
    element_2: session_type.split('_')[2]
    element_3: session_type.split('_')[3]
    return element_0, element_1, element_2, element_3


def parse_session_category(session_type: str) ->str:
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")

    if "OPHYS" in session_type:
        if "0" in session_type:
            session_category = "ophys_habituation"
        else: 
            session_category = "ophys_data_collection"
    elif "TRAINING" in session_type:
        session_category = "task_training"
    return session_category


def parse_stimulus_type(session_type: str) ->str:
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")
    if "images" in session_type:
        stimulus_type = "natural_images"
    elif "gratings" in session_type:
        stimulus_type = "gratings"
    return stimulus_type


def parse_stimulus_presentation_type(session_type: str) -> str:
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")
    if "0" in session_type or "1" in session_type and "TRAINING" in session_type:
        stimulus_presentation_type = "static"
    elif "7" in session_type:
        stimulus_presentation_type = "repeated_movie_clips"
    else:
        stimulus_presentation_type = "flashed"
    return stimulus_presentation_type


def parse_behavior_context(session_type: str) ->str:
    """_summary_

    Parameters
    ----------
    session_type : str
        _description_

    Returns
    -------
    str
        _description_
    """
    passive_ophys_session_nums = ["2", "5", "7"]

    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")
    is_passive = any(passive_num in session_type for passive_num in passive_ophys_session_nums) and "OPHYS" in session_type
    if is_passive:
        behavior_type = "passive_viewing"
    else:
        behavior_type = "active_behavior"
    return behavior_type


def parse_session_subcategory(session_type: str) ->str:
    """_summary_

    Parameters
    ----------
    session_type : str
        _description_

    Returns
    -------
    str
        _description_
    """
    if "habituation" in session_type:
        subtype = "habituation"
    elif "gratings" in session_type: 
        subtype = "gratings"
    elif "images" in session_type: 
        subtype = "images"

    return subtype


def parse_session_primary_category(session_type: str) ->str:
    """_summary_

    Parameters
    ----------
    session_type : str
        _description_

    Returns
    -------
    str
        _description_
    """
    if "OPHYS" in session_type:
        category = "ophys"
    elif "TRAINING" in session_type:
        category = "training"
    return category


def parse_session_category(session_type: str) -> str:
    """_summary_

    Parameters
    ----------
    session_type : str
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")
    primary_category = parse_session_primary_category(session_type)
    subcategory = parse_session_subcategory(session_type)
    full_category = primary_category + "_" + subcategory
    return full_category


def parse_stimulus_set(session_type: str) ->str:
    """_summary_

    Parameters
    ----------
    session_type : str
        _description_

    Returns
    -------
    str
        _description_
    """
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")
    
    split_str = session_type.split('_')
    if split_str[2] == 'images':
        stim_set = split_str[2] + "_" + split_str[3]
    elif split_str[2] == 'gratings':
        stim_set = 'gratings'
    return stim_set


def parse_stimulus_set_novelty(session_type: str) ->str:
    familiar_ophys_sessions = ["0", "1", "2", "3"]
    
    conditions.validate_value_in_list(session_type, SESSION_TYPES, "SESSION_TYPES")

    if "TRAINING" in session_type:
        stimulus_set_novelty = "training"
    elif "7" in session_type:
        stimulus_set_novelty = "NA"
    else:
        is_familiar = any(familiar_num in session_type for familiar_num in familiar_ophys_sessions) 
        if is_familiar:
            stimulus_set_novelty = "familiar"
        else:
            stimulus_set_novelty = "novel"
    return stimulus_set_novelty


def parse_task_type(session_type: str) ->str:
    pass


############################
#
#     from project code
#
############################

def parse_num_cortical_structures(project_code: str) ->int:
    """_summary_

    Parameters
    ----------
    project_code : str
        _description_

    Returns
    -------
    int
        _description_
    """
    conditions.validate_value_in_list(project_code, PROJECT_CODES, "PROJECT_CODES")
    num_structures = NUM_STRUCTURES_DICT[project_code]
    return num_structures

   
def parse_num_depths(project_code: str) ->int:
    """_summary_

    Parameters
    ----------
    project_code : str
        _description_

    Returns
    -------
    int
        _description_
    """
    conditions.validate_value_in_list(project_code, PROJECT_CODES, "PROJECT_CODES")
    num_depths = NUM_DEPTHS_DICT[project_code]
    return num_depths

