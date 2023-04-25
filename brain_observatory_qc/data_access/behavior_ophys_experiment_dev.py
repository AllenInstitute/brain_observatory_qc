from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment

from brain_observatory_qc.pipeline_dev import calculate_new_dff
from brain_observatory_qc.data_access import utilities

from allensdk.brain_observatory.behavior.event_detection import \
    filter_events_array

DFF_PATH = Path("//allen/programs/mindscope/workgroups/learning/pipeline_validation/dff")
GH_DFF_PATH = Path("//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/GH_data/dff")
EVENTS_PATH = Path("//allen/programs/mindscope/workgroups/learning/pipeline_validation/events/")


class BehaviorOphysExperimentDev:
    """Wrapper class for BehaviorOphysExperiment that adds custom
     methods, loads from_lims() only

    Parameters
    ----------
    ophys_experiment_id : int
        The ophys_experiment_id of the experiment to be loaded.
    kwargs : dict
        Keyword arguments to be passed to the BehaviorOphysExperiment

    Returns
    -------
    BehaviorOphysExperimentDev
        An instance of the BehaviorOphysExperimentDev class.

    Notes
    -----

    Uses "duck typing" to override the behavior of the
    BehaviorOphysExperiment class. All uknown requests are passed to the
    BehaviorOphysExperiment class. One issue wit this approach:
    that uses isinstance or issubclass will not work as expected.
    see: https://stackoverflow.com/a/60509130

    _get_new_dff looks for new_dffs in pipeline_dev folder, will throw
    error if not found.

    Example usage:
    expt_id = 1191477425
    expt = BehaviorOphysExperimentDev(expt_id, skip_eye_tracking=True)

    """

    def __init__(self,
                 ophys_experiment_id,
                 events_version: int = 1,
                 filter_params: dict = None,
                 **kwargs):
        self.inner = BehaviorOphysExperiment.from_lims(ophys_experiment_id,
                                                       **kwargs)
        # self.dff_traces = self._get_new_dff() # not relevant for now
        self.ophys_experiment_id = ophys_experiment_id
        self.metadata = self._update_metadata()
        self.cell_specimen_table = self._update_cell_specimen_table()

        try:
            self.events = self._get_new_events(events_version, filter_params)
        except FileNotFoundError:
            # warn new_events not loaded
            # TODO: should we create one?
            print(f"No new_events file for ophys_experiment_id: "
                  f"{self.ophys_experiment_id}")

    def _get_new_dff(self):
        """Get new dff traces from pipeline_dev folder"""

        # TODO: not hardcoded
        pipeline_dev_paths = [DFF_PATH, GH_DFF_PATH]

        # check if file exits, matching pattern "ophys_experiment_id_dff_*.h5"
        dff_fn = f"{self.ophys_experiment_id}_new_dff.h5"
        dff_file = []
        for path in pipeline_dev_paths:
            dff_file += list(path.glob(dff_fn))
        if len(dff_file) == 0:
            # warn and create new dff
            print(f"No dff file for ophys_experiment_id: "
                  f"{self.ophys_experiment_id}, creating new one")

            dff_file = self._create_new_dff()

        elif len(dff_file) > 1:
            raise FileNotFoundError((f">1 dff files for ophys_experiment_id"
                                     f"{self.ophys_experiment_id}"))
        else:
            dff_file = dff_file[0]

        # load dff traces, hacky because of dff trace is list in DataFrame
        # TODO: make into function
        with h5py.File(dff_file, "r") as f:
            # bit of code from dff_file.py in SDK
            traces = np.asarray(f['new_dff'], dtype=np.float64)
            roi_names = np.asarray(f['cell_roi_id'])
            idx = pd.Index(roi_names, name='cell_roi_id').astype('int64')
            new_dff = pd.DataFrame({'dff': [x for x in traces]}, index=idx)
        old_dff = self.inner.dff_traces.copy().reset_index()

        # check if rois are the same
        same_rois = np.intersect1d(old_dff.cell_roi_id.values, new_dff.index.values)
        if len(same_rois) != len(old_dff):
            print('rois in dff traces df are different')

        # if there are nans in index, check if cell_specimen_id is in table
        if old_dff['cell_specimen_id'].isna().sum() == len(old_dff):
            print('found nan cell specimen ids in dff traces df')
            cell_specimen_table = utilities.replace_cell_specimen_ids(old_dff.cell_roi_id.values)
            old_dff.drop(['cell_specimen_id'], axis=1, inplace=True)
            old_dff = pd.merge(old_dff, cell_specimen_table, on='cell_roi_id', how='inner')

        # merge on cell_roi_id
        updated_dff = (pd.merge(new_dff.reset_index(),
                                old_dff.drop(columns=["dff"]),
                                on="cell_roi_id", how="inner")
                       .set_index("cell_specimen_id"))
        # if there are nans in index, check if cell_specimen_id is in table
        # if updated_dff.index.isna().sum() > 0:
        #     cell_specimen_table = utilities.replace_cell_specimen_ids(updated_dff.cell_roi_ids.values)
        #     updated_dff = updated_dff.reset_index()
        #     updated_dff.drop(['cell_specimen_id'], axis=1, inplace=True)
        #     updated_dff = pd.merge(updated_dff, cell_specimen_table, on='cell_roi_id', how='inner')

        return updated_dff

    def _get_new_events(self, events_version: int = 1,
                        filter_params: dict = None):
        """Get new events from pipeline_dev folder"""

        events_folder = "oasis_nrsac_v1"  # f"oasis_v{events_version}"
        version_folder = EVENTS_PATH / events_folder

        # check version folder exists
        if not version_folder.exists():
            version_folder = EVENTS_PATH / "oasis_v1"
            print(f"Events version folder not found: {events_folder}, "
                  f"defaulting to {version_folder}")

        events_file = version_folder / f"{self.ophys_experiment_id}.h5"

        if not events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        events_df = self._load_oasis_events_h5_to_df(events_file, filter_params)

        # if there are nans in index, check if cell_specimen_id is in table
        if events_df.index.isna().sum() == len(events_df):
            print('found nan cell specimen ids in events df')
            cell_specimen_table = utilities.replace_cell_specimen_ids(events_df.cell_roi_id.values)
            events_df = events_df.reset_index()
            events_df.drop(['cell_specimen_id'], axis=1, inplace=True)
            events_df = pd.merge(events_df, cell_specimen_table, on='cell_roi_id', how='inner')
            events_df = events_df.set_index('cell_specimen_id')

        return events_df

    def _load_oasis_events_h5_to_df(self,
                                    events_h5: str,
                                    filter_params: dict = None) -> pd.DataFrame:
        """Load h5 file from new_dff module

        Parameters
        ----------
        events_h5 : Path
            Path to h5 file
        filter_params : dict
            Keyword arguments to be passed to filter_events_array, if None
            use default values. See filter_events_array for details.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns "cell_roi_id" and "events" and filtered events
        """
        # default filter params
        filter_scale_seconds = 2
        frame_rate_hz = 10.7
        filter_n_time_steps = 20

        # check filter params for each key, if not present, use default
        if filter_params is not None:
            for key in ["filter_scale_seconds", "frame_rate_hz", "filter_n_time_steps"]:
                if key in filter_params:
                    locals()[key] = filter_params[key]

        with h5py.File(events_h5, 'r') as f:
            h5 = {}
            for key in f.keys():
                h5[key] = f[key][()]

        events = h5['spikes']

        filtered_events = \
            filter_events_array(arr=events,
                                scale=filter_scale_seconds * frame_rate_hz,
                                n_time_steps=filter_n_time_steps)

        dl = [[d] for d in events]  # already numpy arrays
        fe = [np.array(fe) for fe in filtered_events]
        df = pd.DataFrame(dl).rename(columns={0: 'events'})
        df['cell_roi_id'] = h5['cell_roi_id']
        df['filtered_events'] = fe

        # columns order
        df = df[['cell_roi_id', 'events', 'filtered_events']]

        # get dff trace for cell_specimen_id mapping to cell_roi_id
        dff = self.inner.dff_traces.copy().reset_index()
        df = (pd.merge(df, dff[["cell_roi_id", "cell_specimen_id"]],
                       on="cell_roi_id", how="inner").set_index("cell_specimen_id"))

        return df

    def _update_metadata(self):
        """Update metadata, specifically correct ophsy_frame_rate"""
        metadata = self.inner.metadata.copy()
        dt = np.median(np.diff(self.ophys_timestamps))
        metadata["ophys_frame_rate"] = 1 / dt
        return metadata

    def _update_cell_specimen_table(self):
        """Update cell_specimen_table with new cell_specimen_ids if they exist"""
        cst = self.inner.cell_specimen_table.copy()
        if cst.index.isna().sum() == len(cst): # if all nans
            cst = cst.reset_index().drop(['cell_specimen_id'], axis=1)
            cell_roi_ids = cst.cell_roi_id.values
            cell_specimen_table = utilities.replace_cell_specimen_ids(cell_roi_ids)
            cst = cst.join(cell_specimen_table, on='cell_roi_id', how='inner')
            cst = cst.set_index('cell_specimen_id')
        
        return cst

    def _create_new_dff(self):
        """Create new dff traces"""
        try:
            # get new dff DataFrame
            new_dff_df, timestamps = calculate_new_dff.get_new_dff_df(self.ophys_experiment_id, use_valid_rois=True)

            # Save as h5 file, because of the timestamps
            dff_file = calculate_new_dff.save_new_dff_h5(DFF_PATH, new_dff_df, timestamps, self.ophys_experiment_id)

            print(f"Created new_dff file at: {dff_file}")

            return dff_file
        except AssertionError:
            print('Could not save new dff file. Encountered Assertion Error.')

    # Delegate all else to the "inherited" BehaviorOphysExperiment object
    # Need attribute error to pickle/multiprocessing
    # see: https://stackoverflow.com/a/49380669
    def __getattr__(self, attr):
        if 'inner' not in vars(self):
            raise AttributeError
        return getattr(self.inner, attr)

    # # getstate/setstate for multiprocessing
    # # see: https://stackoverflow.com/a/50158865
    # def __getstate__(self):
    #     return self.inner

    # def __setstate__(self, state):
    #     print(state)
    #     self.inner = state
