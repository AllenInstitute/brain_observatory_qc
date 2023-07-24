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
import json

from allensdk.brain_observatory.behavior.behavior_ophys_experiment \
    import BehaviorOphysExperiment
from brain_observatory_qc.pipeline_dev.calculate_new_dff import get_correct_frame_rate

import logging

# argparser
import argparse
parser = argparse.ArgumentParser(description='')

# provide one of these required args, not both
parser.add_argument('--h5_path', type=str, default=None,
                    metavar='h5 file path')
parser.add_argument('--oeid', type=int, default=None,
                    metavar='ophys_experiment_id')

# multi only works for h5_path currently (not oeid)
parser.add_argument('--multiprocessing', action='store_true',
                    default=False, help='use multiprocessing')


########################
# CHANGE OASIS REPO PATH
########################
# oasis needs to be imported manually (not sure why)
# oasis_repo_path = Path(r"C:\Users\ariellel\repos\brain_observatory\OASIS")
# if not oasis_repo_path.exists():
#     raise UserWarning("OASIS repo not found. Please clone from"
#                       "github.com/j-friedrich/OASIS, or change path"
#                       "in this script")
# sys.path.insert(0, os.path.abspath(oasis_repo_path))
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


def get_h5s(path):
    """Get all h5 files from a given path

    Parameters
    ----------
    path : Path
        Path to search for h5 files

    Returns
    -------
    h5s : list
        List of h5 files
    """
    h5s = glob.glob(str(path / "*.h5"))
    return h5s


def oasis_deconvolve(traces,
                     params: dict,
                     estimate_parameters: bool = True,
                     **kwargs) -> np.array:
    """Deconvolve traces for for all cells in a trace array

    Parameters
    ----------
    traces : np.array
        Array of traces, shape (n_cells, n_frames)
    params : dict
        Dictionary of parameters for oasis
    estimate_parameters : bool, optional
        Whether to estimate parameters, by default True
    **kwargs : dict
        Additional arguments to pass to oasisAR1

    Returns
    -------
    spikes : np.array
        Array of spikes, shape (n_cells, n_frames)

    Notes
    -----
    + See OASIS repo for more parameters

    estimate_parameters=True
    ------------------------
    + penalty: (sparsity penalty) 1: min |s|_1  0: min |s|_0
    + g
    + sn
    + b
    + b_nonneg
    + optimize_g: number of large, isolated events to FURTHER optimize g
    + kwargs

    estimate_parameters=False
    -------------------------
    + tau
    + rate
    + s_min

    """
    if estimate_parameters:
        # check params for required keys
        required_keys = ['g', 'sn', 'b', 'b_nonneg', 'optimize_g', 'penalty']
        for key in required_keys:
            if key not in params:
                raise UserWarning(f"params must contain {key}")

    elif not estimate_parameters:
        # check params for required keys
        required_keys = ['tau', 'rate', 's_min']
        for key in required_keys:
            if key not in params:
                raise UserWarning(f"params must contain {key}")

        g = np.exp(-1 / (params['tau'] * params['rate']))
        lam = 0

        params['g'] = g
        params['lam'] = lam

    # run oasis on each trace
    spikes = []
    calcium = []
    baseline = []
    g_hat = []
    lam_hat = []
    for t in traces:

        # check if trace has any nans, if so return all nans
        if np.isnan(t).any():
            spikes.append(np.full_like(t, np.nan))
        else:
            if estimate_parameters:
                c, s, b, g, lam = deconvolve(t, g=params['g'],
                                             sn=params['sn'],
                                             b=params['b'],
                                             b_nonneg=params['b_nonneg'],
                                             optimize_g=params['optimize_g'],
                                             penalty=params['penalty'],
                                             **kwargs)
                g_hat.append(g)  # Note using AR1, so only need the first param
                lam_hat.append(lam)

                spikes.append(s)
                calcium.append(c)
                baseline.append(b)
            else:
                # c: inferred calcium, s: inferred spikes
                c, s = oasisAR1(t, params['g'],
                                s_min=params['s_min'],
                                lam=params['lam'])
                spikes.append(s)
                calcium.append(c)
    if estimate_parameters:
        # AR1, so just list of g, if need Ar2, need to account for tuple in json dump
        params["g_hat"] = g_hat
        params["lam_hat"] = lam_hat

    # smoothing
    # y = np.convolve(y, np.ones(3)/3, mode='same')

    return np.array(spikes), params


def quick_fix_nans(traces: np.array,
                   max_per_nan: float = 0.2) -> np.array:
    """Quick fix for nans in spikes

    Parameters
    ----------
    traces : np.array
        Array of spikes, shape (n_cells, n_frames)
    max_per_nan : float, optional
        Maximum percent of nans allowed, by default 0.2
        sets whole trace to 0 if nans > max_per_nan, so OASIS works

    Returns
    -------
    traces : np.array
        Array of tracews, shape (n_cells, n_frames)
    n_nans_list : list
        List of number of nans in each trace
    """

    n_nans_list = []
    for i, t in enumerate(traces):

        # check is n nans greater than 20% of trace
        n_nans = np.sum(np.isnan(t))
        if n_nans > max_per_nan * len(t):
            traces[i] = np.full_like(t, 0)
            continue

        # Set to 0 all nans at start of trace
        start = np.where(~np.isnan(t))[0][0]
        traces[i, :start] = 0

        # linear interolate nans
        traces[i] = np.interp(np.arange(len(t)), np.where(
            ~np.isnan(t))[0], t[~np.isnan(t)])

        n_nans_list.append(int(n_nans))

    return traces, n_nans_list


def load_new_dff_h5(h5_path: Path) -> dict:
    """Load h5 file from new_dff module

    Parameters
    ----------
    h5_path : Path
        Path to h5 file

    Returns
    -------
    h5 : dict
        Dictionary of h5 file
    """
    with h5py.File(h5_path, 'r') as f:
        h5 = {}
        for key in f.keys():
            h5[key] = f[key][()]
    return h5


# TODO: make params as input
def generate_oasis_events_for_h5_path(h5_path: Path,
                                      out_path: Path,
                                      trace_type: list = 'new_dff',
                                      estimate_parameters: bool = True,
                                      qc_plot: bool = True,
                                      **kwargs) -> None:
    """Generate oasis events for all traces in pipe_dev dff folder

    Parameters
    ----------
    h5_path : Path
        Path to h5 file, expected new_dff h5 file
    out_path : Path
        Path to save oasis events
    trace_type : list, optional
        Trace type to use for deconvolution, can be 'new_dff', 'old_dff',
        or'np_corrected', as these are outputs of the pipeline_dev new_dff module
    estimate_parameters : bool, optional
        Whether to estimate parameters (CONSTRAINED AR1)
        or use provided parameters (UNCONSTRAINED AR1)
    trace_type : str
        Trace type to use for deconvolution
        can be 'new_dff', 'old_dff', or'np_corrected', as these are outputs
        of the pipeline_dev new_dff module
    **kwargs : dict
        UNCONSTRAINED AR1 kwargs
        + tau
        + rate
        + s_min

        CONSTRAINED AR1 kwargs
        + optimize_g
        + penalty
        + g (optimized)
        + sn (optimized)
        + b (optimized)

    Returns
    -------
    None

    Notes
    -----
    This function is specific to pipeline_dev work, and depends on
    an experiment table ('three_mice_ids_df.pkl'), which describes the
    experiments to be processed.


    """

    # DEFAULT PARAMS
    # TODO: make these as inputs
    params = {}
    params['g'] = (None,)
    params['b'] = None
    params['sn'] = None
    params['optimize_g'] = 0
    params['penalty'] = 1
    params['b_nonneg'] = True
    params['estimate_parameters'] = estimate_parameters
    params['method'] = 'constrained_oasisAR1' if estimate_parameters else 'unconstrained_oasisAR1'
    try:
        # create json file to see all files and params

        # root_dir = Path("/allen/programs/mindscope/workgroups/"
        #                      "learning/pipeline_validation")
        # oasis_path_root = root_dir / "events_oasis"
        out_path.mkdir(exist_ok=True, parents=True)

        # h5s = get_h5s_from_pipe_dev(root_dir)
        expt_id = int(Path(h5_path).name.split("_")[0])
        oasis_h5 = out_path / f"{expt_id}.h5"
        print(f"Processing {expt_id}")

        # cehck if oasis h5 exists
        if oasis_h5.exists():
            print(f"{oasis_h5} already exists")
            return

        trace_dict = load_new_dff_h5(h5_path)
        traces = trace_dict[trace_type]
        roi_ids = trace_dict['cell_roi_id']

        # check if trace has nans, skip if so
        # if np.isnan(traces).any():
        #     nan_expt_ids.append(expt_id)
        #     continue

        # just get expt for frame rate
        # experiment = BehaviorOphysExperimentDev(expt_id)

        # timestamps = trace_dict['timestamps']
        # rate = experiment.metadata['ophys_frame_rate']

        rate, timestamp_df = get_correct_frame_rate(expt_id)
        timestamps = timestamp_df.ophys_frames.values[0]

        params['rate'] = rate

        # traces, n_nan_list = quick_fix_nans(traces)
        # # warn that some traces have nans
        # if len(n_nan_list) > 0:
        #     print(f"WARNING: {len(n_nan_list)} traces have nans")

        nans = np.where(np.isnan(traces))[0]
        if len(nans) > 0:
            raise ValueError(f"Traces have nans: {len(nans)} in {expt_id}")

        spikes, params = oasis_deconvolve(traces, params, estimate_parameters)

        # # get index with nan traces
        # nan_traces = np.isnan(spikes).all(axis=1)
        # nan_trace_ids = roi_ids[nan_traces]

        # save to h5
        with h5py.File(oasis_h5, 'w') as file:
            file.create_dataset("cell_roi_id", data=roi_ids)
            file.create_dataset("spikes", data=spikes)

        if qc_plot:
            plots_path = out_path / "plots"
            plots_path.mkdir(exist_ok=True, parents=True)
            plot_trace_and_events(traces, spikes, timestamps,
                                  roi_ids, params, expt_id, plots_path)

        params['events_path'] = str(oasis_h5)
        params['trace_type'] = trace_type
        # params['n_nans'] = n_nan_list

        # dump params to json
        with open(out_path / f"{expt_id}_params.json", 'w') as file:
            json.dump(params, file)

        logging.info(f"SUCCESS: {expt_id}")
    except Exception as e:
        logging.error(f"FAILED: {expt_id}")
        raise e
    return oasis_h5

def generate_oasis_events_for_oeid(oeid: int,
                                   out_path: Path,
                                   trace_type: list = 'dff',
                                   estimate_parameters: bool = True,
                                   qc_plot: bool = False,
                                   **kwargs) -> None:
    """Generate oasis events for all traces in pipe_dev dff folder

    Parameters
    ----------
    oeid : int
        ophys experiment id
    out_path : Path
        Path to save oasis events
    trace_type : list, optional
        (not implemented, defalt is dff)
    estimate_parameters : bool, optional
        Whether to estimate parameters (CONSTRAINED AR1)
        or use provided parameters (UNCONSTRAINED AR1)
    trace_type : str
        Trace type to use for deconvolution
        can be 'new_dff', 'old_dff', or'np_corrected', as these are outputs
        of the pipeline_dev new_dff module
    **kwargs : dict
        UNCONSTRAINED AR1 kwargs
        + tau
        + rate
        + s_min

        CONSTRAINED AR1 kwargs
        + optimize_g
        + penalty
        + g (optimized)
        + sn (optimized)
        + b (optimized)

    Returns
    -------
    None

    """

    # DEFAULT PARAMS
    # TODO: make these as inputs

    params = {}
    params['g'] = (None,)
    params['b'] = None
    params['sn'] = None
    params['optimize_g'] = 0
    params['penalty'] = 1
    params['b_nonneg'] = True
    params['estimate_parameters'] = estimate_parameters
    params['method'] = 'constrained_oasisAR1' if estimate_parameters else 'unconstrained_oasisAR1'

    try:
        expt_id = oeid
        oasis_h5 = out_path / f"{expt_id}.h5"

        # check if oasis h5 exists
        if oasis_h5.exists():
            logging.info(f"SKIPPED {expt_id} already exists")
            return

        experiment = BehaviorOphysExperiment.from_lims(expt_id)
        dff = experiment.dff_traces
        traces = np.array([np.array(d) for d in dff.dff])
        roi_ids = experiment.dff_traces.cell_roi_id.values
        timestamps = experiment.ophys_timestamps
        rate = experiment.metadata['ophys_frame_rate']

        traces, n_nan_list = quick_fix_nans(traces)

        events, params = oasis_deconvolve(traces, params, estimate_parameters)

        # # get index with nan traces
        # nan_traces = np.isnan(spikes).all(axis=1)
        # nan_trace_ids = roi_ids[nan_traces]

        # save to h5
        with h5py.File(oasis_h5, 'w') as file:
            file.create_dataset("cell_roi_id", data=roi_ids)
            # KEEP "spikes" to be consitent
            file.create_dataset("spikes", data=events)

        if qc_plot:
            plots_path = out_path / "plots"
            plots_path.mkdir(exist_ok=True, parents=True)
            plot_trace_and_events(traces, events, timestamps,
                                  roi_ids, params, expt_id, plots_path)

        # output params
        params['rate'] = rate
        params['events_path'] = str(oasis_h5)
        params['trace_type'] = trace_type
        params['n_nans'] = n_nan_list

        # dump params to json
        with open(out_path / f"{expt_id}_params.json", 'w') as file:
            json.dump(params, file)

        logging.info(f"SUCCESS: {expt_id}")
    except Exception as e:
        logging.error(f"FAILED: {expt_id}")
        raise e


# load h5 events to dataframe
def load_oasis_events_h5(h5_path: Path) -> pd.DataFrame:
    """Load h5 file from new_dff module

    Parameters
    ----------
    h5_path : Path
        Path to h5 file

    Returns
    -------
    """
    with h5py.File(h5_path, 'r') as f:
        h5 = {}
        for key in f.keys():
            h5[key] = f[key][()]

    # datframe
        df = pd.DataFrame(h5['cell_roi_id'], columns=h5['spikes'])
    return df


def plot_trace_and_events(traces, spikes, timestamps, roi_ids, params, expt_id, plots_path, show_fig=False):
    sns.set_context('talk')
    with PdfPages(plots_path / f"{expt_id}_oasis.pdf") as pdf:
        for i, (spike, trace, cell) in enumerate(zip(spikes, traces, roi_ids)):
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            ax.plot(timestamps, trace, color='g', label='new_dff')

            if params['estimate_parameters']:
                g = params['g_hat'][i]
            else:
                g = params['g']
            ax2 = ax.twinx()
            ax2.plot(timestamps, spike * 1, color='orange',
                     label=f"events, g={g}")

            # xlim
            ax.set_xlim(400, 580)
            ax2.set_xlim(400, 580)  # arbitrary time period to check
            ax.legend()
            ax2.legend()
            ax.set_title(f"cell_roi_id: {cell}")
            pdf.savefig(fig)
        if not show_fig:
            plt.close(fig)


def oasis_deconvolve_per_cell(y, tau, rate, s_min, opt_g):
    """Deconvolve traces for a single cell, with optimize= option
    Note: not documented as it is not used in the pipeline
    """
    g = np.exp(-1 / (tau * rate))

    if opt_g:
        c, s, b, g, lam = deconvolve(y, optimize_g=5)
    else:
        lam = 0
        y = (y - np.mean(y))
        c, s = oasisAR1(y, g, s_min=s_min, lam=lam)
    y = np.convolve(y, np.ones(3) / 3, mode='same')

    return y, c, s, g, lam


if __name__ == "__main__":
    args = args = parser.parse_args()
    h5_path = args.h5_path
    multiprocessing = args.multiprocessing
    oeid = args.oeid

    assert (h5_path is None) != (
        oeid is None), "Must provide h5_path or oeid, but not both"

    root_dir = Path(
        "/allen/programs/mindscope/workgroups/learning/pipeline_validation")
    out_path = root_dir / "events" / "oasis_nrsac_v1"

    # make output dir
    out_path.mkdir(exist_ok=True, parents=True)

    # start logger
    log_path = out_path / "events_processing.log"
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(f"SAVING: oasis events to: {out_path}")

    if multiprocessing:
        # import multiprocessing as mp
        # from functools import partial
        # from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

        # expt_ids = start_lamf_analysis().index.values

        # h5_list = []
        # for expt_id in expt_ids:
        #     h5_list.append(root_dir / "dff" / f"{expt_id}_new_dff.h5")

        # func = partial(generate_oasis_events_for_trace_types,
        #                out_path=out_path,
        #                trace_type='new_dff',
        #                estimate_parameters=True,
        #                qc_plot=True)

        # with mp.Pool(10) as p:
        #     p.map(func, h5_list)
        print('not implemented')

    if h5_path is not None:
        generate_oasis_events_for_h5_path(h5_path,
                                          out_path,
                                          trace_type='new_dff',
                                          estimate_parameters=True,
                                          qc_plot=True)

    if oeid is not None:
        generate_oasis_events_for_oeid(oeid,
                                       out_path,
                                       # not implemented, uses dff_traces from expt object (dev or SDK)
                                       trace_type='dff',
                                       estimate_parameters=True,
                                       qc_plot=False)
