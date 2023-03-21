import brain_observatory_qc.utilities.pre_post_conditions as conditions




#####################################################################
#
#           		PROJECT CONSTANTS
#
#####################################################################

ACTIVE_BEHAVIOR_STIMULI = [
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
    "TRAINING_5_images_G_handoff_ready",
    "OPHYS_1_images_A",
    "OPHYS_1_images_B",
    "OPHYS_1_images_G",
    "OPHYS_3_images_A",
    "OPHYS_3_images_B",
    "OPHYS_3_images_G",
    "OPHYS_4_images_A",
    "OPHYS_4_images_B",
    "OPHYS_4_images_H",
    "OPHYS_6_images_A",
    "OPHYS_6_images_B",
    "OPHYS_6_images_H",
]


def parse_behavior_type(session_type: str) -> str:
    """assigns a behavior type to a specific session based on session_type

    Parameters
    ----------
    session_type : str
        the visual stimulus that was run during the session

    Returns
    -------
    str
        "passive_viewing" or "active_behavior"
    """
    if session_type in ACTIVE_BEHAVIOR_STIMULI:
         behavior_type = "active_behavior"
    else: 
        behavior_type = "passive_viewing"
    return behavior_type