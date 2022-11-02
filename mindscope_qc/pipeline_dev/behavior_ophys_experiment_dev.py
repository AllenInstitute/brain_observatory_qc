from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment


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
    def __init__(self, ophys_experiment_id, **kwargs):
        self.inner = BehaviorOphysExperiment.from_lims(ophys_experiment_id,
                                                       **kwargs)
        self.dff_traces = self._get_new_dff()
        self.ophys_experiment_id = ophys_experiment_id
        self.metadata = self._update_metadata()

    def _get_new_dff(self):
        """Get new dff traces from pipeline_dev folder"""

        # TODO: not hardcoded
        pipeline_dev_path = Path("/allen/programs/mindscope/workgroups"
                                 "/learning/pipeline_validation/dff")

        # check if file exits, matching pattern "ophys_experiment_id_dff_*.h5"
        dff_fn = f"{self.ophys_experiment_id}_new_dff.h5"
        dff_file = list(pipeline_dev_path.glob(dff_fn))
        if len(dff_file) == 0:
            raise FileNotFoundError((f"No dff file for ophys_experiment_id"
                                     f"{self.ophys_experiment_id}"))
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

        # merge on cell_roi_id
        updated_dff = (pd.merge(new_dff.reset_index(),
                                old_dff.drop(columns=["dff"]),
                                on="cell_roi_id", how="inner")
                       .set_index("cell_specimen_id"))

        return updated_dff

    def _update_metadata(self):
        """Update metadata, specifically correct ophsy_frame_rate"""
        metadata = self.inner.metadata.copy()
        dt = np.median(np.diff(self.ophys_timestamps))
        metadata["ophys_frame_rate"] = 1 / dt
        return metadata

    # delegate all else to the "inherited" BehaviorOphysExperiment object
    def __getattr__(self, attr):
        return getattr(self.inner, attr)
