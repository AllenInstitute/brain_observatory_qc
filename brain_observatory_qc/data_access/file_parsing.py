# import json
# import h5py
import logging
# import pandas as pd

logger = logging.getLogger(__name__)


# def get_sync_data(lims_data, use_acq_trigger):
#     logger.info('getting sync data')
#     sync_path = get_sync_path(lims_data)
#     sync_dataset = SyncDataset(sync_path)
#     # Handle mesoscope missing labels
#     try:
#         sync_dataset.get_rising_edges('2p_vsync')
#     except ValueError:
#         sync_dataset.line_labels = ['2p_vsync', '', 'stim_vsync', '', 'photodiode', 'acq_trigger', '', '',
#                                     'behavior_monitoring', 'eye_tracking', '', '', '', '', '', '', '', '', '', '', '',
#                                     '', '', '', '', '', '', '', '', '', '', 'lick_sensor']
#         sync_dataset.meta_data['line_labels'] = sync_dataset.line_labels

#     meta_data = sync_dataset.meta_data
#     sample_freq = meta_data['ni_daq']['counter_output_freq']
#     # 2P vsyncs
#     vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
#     vs2p_f = sync_dataset.get_falling_edges(
#         '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
#     vs2p_rsec = vs2p_r / sample_freq
#     vs2p_fsec = vs2p_f / sample_freq
#     if use_acq_trigger:  # if 2P6, filter out solenoid artifacts
#         vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
#         frames_2p = vs2p_r_filtered
#     else:  # dont need to filter out artifacts in pipeline data
#         frames_2p = vs2p_rsec
#     # use rising edge for Scientifica, falling edge for Nikon http://confluence.corp.alleninstitute.org/display/IT/Ophys+Time+Sync
#     # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
#     # vs2p_rsec = vs2p_r / sample_freq
#     # frames_2p = vs2p_rsec
#     # stimulus vsyncs
#     # vs_r = d.get_rising_edges('stim_vsync')
#     vs_f = sync_dataset.get_falling_edges('stim_vsync')
#     # convert to seconds
#     # vs_r_sec = vs_r / sample_freq
#     vs_f_sec = vs_f / sample_freq
#     # vsyncs = vs_f_sec
#     # add display lag
#     monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
#     vsyncs = vs_f_sec + monitor_delay  # this should be added, right!?
#     # line labels are different on 2P6 and production rigs - need options for both
#     if 'lick_times' in meta_data['line_labels']:
#         lick_times = sync_dataset.get_rising_edges('lick_1') / sample_freq
#     elif 'lick_sensor' in meta_data['line_labels']:
#         lick_times = sync_dataset.get_rising_edges('lick_sensor') / sample_freq
#     else:
#         lick_times = None
#     if '2p_trigger' in meta_data['line_labels']:
#         trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
#     elif 'acq_trigger' in meta_data['line_labels']:
#         trigger = sync_dataset.get_rising_edges('acq_trigger') / sample_freq
#     if 'stim_photodiode' in meta_data['line_labels']:
#         stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
#     elif 'photodiode' in meta_data['line_labels']:
#         stim_photodiode = sync_dataset.get_rising_edges('photodiode') / sample_freq
#     if 'cam2_exposure' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
#     elif 'cam2' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('cam2') / sample_freq
#     elif 'eye_tracking' in meta_data['line_labels']:
#         eye_tracking = sync_dataset.get_rising_edges('eye_tracking') / sample_freq
#     if 'cam1_exposure' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
#     elif 'cam1' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('cam1') / sample_freq
#     elif 'behavior_monitoring' in meta_data['line_labels']:
#         behavior_monitoring = sync_dataset.get_rising_edges('behavior_monitoring') / sample_freq
#     # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger for 2P6 only
#     if use_acq_trigger:
#         frames_2p = frames_2p[frames_2p > trigger[0]]
#     # print(len(frames_2p))
#     if lims_data.rig.values[0][0] == 'M':  # if Mesoscope
#         roi_group = get_roi_group(lims_data)  # get roi_group order
#         frames_2p = frames_2p[roi_group::4]  # resample sync times
#     # print(len(frames_2p))
#     logger.info('stimulus frames detected in sync: {}'.format(len(vsyncs)))
#     logger.info('ophys frames detected in sync: {}'.format(len(frames_2p)))
#     # put sync data in format to be compatible with downstream analysis
#     times_2p = {'timestamps': frames_2p}
#     times_vsync = {'timestamps': vsyncs}
#     times_lick = {'timestamps': lick_times}
#     times_trigger = {'timestamps': trigger}
#     times_eye_tracking = {'timestamps': eye_tracking}
#     times_behavior_monitoring = {'timestamps': behavior_monitoring}
#     times_stim_photodiode = {'timestamps': stim_photodiode}
#     sync_data = {'ophys_frames': times_2p,
#                  'stimulus_frames': times_vsync,
#                  'lick_times': times_lick,
#                  'eye_tracking': times_eye_tracking,
#                  'behavior_monitoring': times_behavior_monitoring,
#                  'stim_photodiode': times_stim_photodiode,
#                  'ophys_trigger': times_trigger,
#                  }
#     return sync_data


# def get_timestamps(lims_data):
#     if '2P6' in lims_data.rig.values[0]:
#         use_acq_trigger = True
#     else:
#         use_acq_trigger = False
#     sync_data = get_sync_data(lims_data, use_acq_trigger)
#     timestamps = pd.DataFrame(sync_data)
#     return timestamps
