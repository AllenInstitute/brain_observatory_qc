import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mindscope_qc.pipeline_dev.behavior_ophys_experiment_dev \
    import BehaviorOphysExperimentDev

########################
# CHANGE OASIS REPO PATH
########################
# oasis needs to be imported manually (not sure why)
oasis_repo_path = Path("/home/matt.davids/code/OASIS")
if not oasis_repo_path.exists():
    raise UserWarning("OASIS repo not found. Please clone from"
                      "github.com/j-friedrich/OASIS, or change path"
                      "in this script")
sys.path.insert(0, os.path.abspath(oasis_repo_path))
from oasis.functions import deconvolve  # noqa: E402
from oasis.oasis_methods import oasisAR1  # noqa: E402


def get_h5s_from_pipe_dev(pipe_dev_path, sub_folder="dff"):
    """Get all h5 files from a given subfolder of pipe_dev_path

    Parameters
    ----------
    pipe_dev_path : Path
        Path to the pipe_dev folder
    sub_folder : str, optional
        Subfolder to search for h5 files, by default "dff"

    Returns
    -------
    h5s : list
        List of h5 files

    Notes
    -----
    + Specific for pipeline_dev work, with expt_table in the parent folder
    + Assumes H5 file startes with ophys_exerperiment_id
    """
    # load a table with experiments
    expt_table = pd.read_pickle(pipe_dev_path / 'three_mice_ids_df.pkl')
    ids = expt_table.ophys_experiment_id.values

    # load h5s with ids in fn
    # keys : ['cell_roi_id',
    #         'cell_specimen_id',
    #         'new_dff',
    #         'np_corrected',
    #         'old_dff',
    #         'timestamps']
    h5s = []
    for id in ids:
        h5s.extend(glob.glob(str(pipe_dev_path / sub_folder / f"{id}_*.h5")))

    return h5s


def oasis_deconvolve(traces, tau=1.0, rate=11.0, s_min=0,
                     **kwargs) -> np.array:
    """Deconvolve traces for for all cells in a trace array

    Parameters
    ----------
    traces : np.array
        Array of traces, shape (n_cells, n_frames)
    tau : float, optional
        Time constant used to calculate 'g' for AR1 process
    rate : float, optional
        Sampling rate of traces
    s_min : float, optional
        spike minimum size
    **kwargs : dict
        Additional arguments to pass to oasisAR1

    Returns
    -------
    spikes : np.array
        Array of spikes, shape (n_cells, n_frames)

    Notes
    -----
    + Uses oasisAR1 (unconstrained OASIS)
    + See OASIS repo for more parameters

    """
    g = np.exp(-1/(tau * rate))
    lam = 0

    # run oasis on each trace
    spikes = []
    for t in traces:

        # check if trace has any nans, if so return all nans
        if np.isnan(t).any():
            spikes.append(np.full_like(t, np.nan))
        else:
            # c: inferred calcium, s: inferred spikes
            c, s = oasisAR1(t, g, s_min=s_min, lam=lam)
            spikes.append(s)
    # smoothing
    # y = np.convolve(y, np.ones(3)/3, mode='same')
    return np.array(spikes)


def generate_oasis_events_for_trace_types(tau: float = 1.0,
                                          qc_plot: bool = True,
                                          **kwargs) -> None:
    """Generate oasis events for all traces in pipe_dev dff folder

    Parameters
    ----------
    tau : float
        Time constant used to calculate 'g' for AR1 process
    **kwargs : dict
        Keyword arguments to pass to oasis_deconvolve

    Returns
    -------
    None

    Notes
    -----
    This function is specific to pipeline_dev work, and depends on
    an experiment table ('three_mice_ids_df.pkl'), which describes the
    experiments to be processed.
    """
    pipe_dev_path = Path("/allen/programs/mindscope/workgroups/"
                         "learning/pipeline_validation")
    oasis_path_root = pipe_dev_path / "events_oasis"
    oasis_path = oasis_path_root / f"tau_{tau}"
    oasis_path.mkdir(exist_ok=True)
    plots_path = oasis_path / "plots"
    plots_path.mkdir(exist_ok=True)

    h5s = get_h5s_from_pipe_dev(pipe_dev_path)

    nan_expt_ids = []
    for h5 in h5s:

        expt_id = int(Path(h5).name.split("_")[0])
        oasis_h5 = oasis_path / f"{expt_id}_oasis.h5"
        print(f"Processing {expt_id}")
        # cehck if oasis h5 exists
        if not oasis_h5.exists():
            trace_dict = {}
            with h5py.File(h5, 'r') as f:
                for key in f.keys():
                    trace_dict[key] = f[key][()]
            trace_type = 'new_dff'  # can select old_dff, np_corrected
            traces = trace_dict[trace_type]
            roi_ids = trace_dict['cell_roi_id']

            # check if trace has nans, skip if so
            # if np.isnan(traces).any():
            #     nan_expt_ids.append(expt_id)
            #     continue

            # TODO replace with from_network/direct
            experiment = BehaviorOphysExperimentDev(expt_id)
            timestamps = trace_dict['timestamps']
            rate = experiment.metadata['ophys_frame_rate']

            spikes = oasis_deconvolve(traces, tau, rate, **kwargs)

            # get index with nan traces
            nan_traces = np.isnan(spikes).all(axis=1)
            nan_trace_ids = roi_ids[nan_traces]

            # save to h5
            with h5py.File(oasis_h5, 'w') as file:
                file.create_dataset("cell_roi_id", data=roi_ids)
                file.create_dataset("spikes", data=spikes)

            if qc_plot:
                sns.set_context('talk')
                pdf = PdfPages(plots_path / f"{expt_id}_oasis.pdf")
                for spike, trace, cell in zip(spikes, traces, roi_ids):
                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    ax.plot(timestamps, trace, color='g', label='new_dff')
                    ax.plot(timestamps, spike * -1, color='orange',
                            label=f"events, tau={tau}")
                    ax.legend()
                    pdf.savefig(fig)
                pdf.close()

        else:
            print(f"{oasis_h5} already exists")

        # write nan expts to text file
        if nan_expt_ids:
            with open(oasis_path / f"{expt_id}_nan_rois.txt", "w") as f:
                for roi in nan_trace_ids:
                    f.write(f"{roi}")


def oasis_deconvolve_per_cell(y, tau, rate, s_min, opt_g):
    """Deconvolve traces for a single cell, with optimize= option
    Note: not documented as it is not used in the pipeline
    """
    g = np.exp(-1/(tau * rate))

    if opt_g:
        c, s, b, g, lam = deconvolve(y, optimize_g=5)
    else:
        lam = 0
        y = (y - np.mean(y))
        c, s = oasisAR1(y, g, s_min=s_min, lam=lam)
    y = np.convolve(y, np.ones(3) / 3, mode='same')

    return y, c, s, g, lam


if __name__ == "__main__":
    generate_oasis_events_for_trace_types(tau=1.0)
