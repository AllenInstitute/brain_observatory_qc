def add_behavior_record(behavior_session_uuid=None, pkl_path=None, overwrite=False, db_connection=None, db_name='behavior_data', data_type='foraging2'):
    '''
    for a given behavior_session_uuid:
      - opens the data with VBA
      - adds a row to the visual_behavior_data database summary
      - adds an entry to each of:
        - running
        - licks
        - time
        - rewards
        - visual_stimuli
        - omitted_stimuli
        - metadata
        - log
    if the session fails to open with VBA:
        - adds a row to summary with 'error_on_load' = True
        - saves traceback and time to 'error_log' table
    '''

    from .translator.foraging2 import data_to_change_detection_core as foraging2_translator
    from .translator.foraging import data_to_change_detection_core as foraging1_translator
    from .translator.core import create_extended_dataframe
    from .change_detection.trials import summarize

    if data_type.lower() == 'foraging2':
        data_to_change_detection_core = foraging2_translator
    elif data_type.lower() == 'foraging1':
        data_to_change_detection_core = foraging1_translator
    else:
        raise NameError('data_type must be either `foraging1` or `foraging2`')