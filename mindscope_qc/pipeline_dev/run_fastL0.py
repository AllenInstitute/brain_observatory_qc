# %% NOTES

# need to install FastlL0ZeroSpikeInference
# https://github.com/AllenInstitute/FastLZeroSpikeInference or seanjewell/FastLZeroSpikeInference

# follow these instructions, worked on centos
# https://linuxhint.com/install_llvm_centos7/

# %%
import seaborn as sns
from pathlib import Path

import ophys_etl.modules.event_detection.__main__ as events_module
from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt, start_lamf_analysis
import mindscope_qc.data_access.from_lims as from_lims

from tqdm import tqdm
import h5py
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from typing import Union
from scipy import stats
import logging

# LZERO_OUTPUT_PATH = Path("/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/events_fast10")
LZERO_OUTPUT_PATH = Path("/allen/programs/mindscope/workgroups/learning/pipeline_validation/events_FastLZero")
NEW_DFF_PATH = Path("/allen/programs/mindscope/workgroups/learning/pipeline_validation/dff")

# %% MAIN FUNCTION


def run_fastl0_wrapper(rate, tau, fpath, opath, rois, nworkers):
    """Wrapper for running FastL0 from ophys_etl_pipeline."""
    args = {'movie_frame_rate_hz': rate,
            'ophysdfftracefile': str(fpath),
            'valid_roi_ids': rois,
            'output_event_file': opath,
            'decay_time': tau,
            'n_parallel_workers': nworkers}

    ed = events_module.EventDetection(input_data=args, args=[])
    ed.run()


def generate_fastl0_events_for_trace_types(trace_keys: list = ['new_dff'],
                                           taus: list = [.715],
                                           output_path_root: str = LZERO_OUTPUT_PATH,
                                           input_path_root: str = NEW_DFF_PATH,
                                           frame_rate: float = 10.7,
                                           nworkers: int = -1,
                                           log: bool = True):
    """
    Generate FastL0 events for all experiments in new_dff folder of pipeline dev.


    Parameters
    ----------
    trace_keys : list, optional
        DESCRIPTION. The default is ['new_dff']. Options are ['new_dff', 'old_dff', 'np_corrected'],
        which are keys in the h5s from jinhos new_dff module.
    taus : list, optional
        DESCRIPTION. The default is [.715].
    output_path_root : str, optional
        DESCRIPTION. The default is LZERO_OUTPUT_PATH.
    input_path_root : str, optional
        Where the "new_dff" h5s are stored.
    frame_rate : float, optional
        Input to FastL0. The default is 10.7. TODO: get this from the experiment, sorry for hardcode :(
    nworkers : int, optional
        Input to FastL0. The default is -1 (all workers).
    log : bool, optional
        To log errors. The default is True.

    Returns
    -------
    None.

    """
    # keep track of experiments with nans, cause they error out
    nan_expt_ids = []

    if log:
        logger = logging.getLogger('generate_fastl0_events_for_trace_types')
        logger.setLevel(logging.DEBUG)
        log_file = output_path_root / 'generate_fastl0_events_for_trace_types.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.ERROR)
        logger.addHandler(fh)

    # where JK saved new_dff h5s
    h5s = get_h5s_from_pipe_dev(input_path_root)

    # load H5s, run FastL0 on each trace type
    for i, h5 in tqdm(enumerate(h5s)):

        expt_id = Path(h5).name.split("_")[0]

        trace_dict = {}
        with h5py.File(h5, 'r') as f:
            for key in f.keys():
                trace_dict[key] = f[key][()]

        # zscore np_corrected traces
        trace_dict['np_corrected'] = stats.zscore(trace_dict['np_corrected'], axis=1)

        # make input dict with only trace keys
        input_dict = {key: trace_dict[key] for key in trace_keys}

        for k, trace in input_dict.items():

            # check if trace has nans
            if np.isnan(trace).any():
                logger.error(f"{expt_id} has nans in {k}")
                nan_expt_ids.append(expt_id)
                continue

            # need to create tmp files with input format for the event detection
            # module in ophys_etl_pipelines
            tmp_fn = f"{expt_id}_{k}_TMP.h5"
            tmp_path = output_path_root / tmp_fn

            with h5py.File(tmp_path, 'w') as f2:
                f2.create_dataset('data', data=trace)
                f2.create_dataset('roi_names', data=trace_dict["cell_roi_id"])
                print(f"created tmp file: {f2.filename}")

            # run fastl0
            for tau in taus:
                try:
                    opath = output_path_root / f"{expt_id}_trace={k}_tau={tau}.h5"
                    if opath.exists():
                        print(f"skipping {opath}")
                        continue

                    run_fastl0_wrapper(frame_rate, tau, tmp_path, str(opath), trace_dict["cell_roi_id"], nworkers)
                except Exception as e:
                    if log:
                        logger.error(f"error with {expt_id}, {k},{tau}, {e}")
                    print(f"error with {expt_id}, {k}, {tau}, {e}")
            # log error

            # delete tmp files
            os.remove(tmp_path)

    # write nan expts to text file
    with open(output_path_root / "nan_expts.txt", "w") as f:
        for expt in nan_expt_ids:
            f.write(f"{expt}")

# %% RUN FASTl0 single testing


def run_Fastl0_experiment(experiment,
                          output_path_root: Union[Path, str] = None,
                          subfolder: str = None,
                          taus: list = [.715],
                          dff: str = "old") -> None:
    """Run FastL0 on an experiment, saving the events to a file.

    Parameters
    ----------
    experiment : BehaviorOphysExperiment
        The experiment to run FastL0 on.
    output_path_root : pathlib.Path
        The root directory to save the events to.
    subfolder : str, optional
        The subfolder to save the events to, by default None
    taus : list, optional
        The decay times to run FastL0 with, by default [.715]

    Returns
    -------
    None
    """

    # assert path provided
    if output_path_root is None:
        raise ValueError("Must provide a path to save events to.")

    # i think input function needs a string, so convert from Path if needed
    if isinstance(output_path_root, Path):
        output_path_root = str(output_path_root)

    expt_id = experiment.ophys_experiment_id
    rate = experiment.metadata["ophys_frame_rate"]
    rois = experiment.cell_specimen_table.cell_roi_id.values

    if dff == "old":
        fpath = from_lims.get_dff_traces_filepath(expt_id)
    elif dff == "new":
        fpath = NEW_DFF_PATH / f"{expt_id}_new_dff.hdf5"
        # fpath = ???
        # TODO: implement new dff

    # create subfolder if needed
    if subfolder is not None:
        output_path = Path(output_path_root) / subfolder
    else:
        output_path = Path(output_path_root)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    # create logger file to write to later
    logger = open(output_path / f"{expt_id}_log.txt", "w")

    # write header to logger in csv format
    logger.write("action, expt_id, tau, frame_rate, n_rois, input_dff_filepath, output_filepath")

    for tau in taus:
        output_file_path = str(output_path / f"{expt_id}_dff=new_tau={tau}.hdf5")

        run_fastl0_wrapper(rate, tau, fpath, output_file_path, rois)

        logger.write(f"Ran FastL0, {expt_id}, {tau}, {rate}, {len(rois)}, {fpath}, {output_file_path}\n")

    logger.close()


# %% EXAMPLE

def run_example():

    expt_id = 1188653069
    experiment = get_ophys_expt(expt_id)

    # taus = [.1, .35, .75, 1.5, 2.5]
    # time run

    import time
    start = time.time()

    taus = [.715]
    run_Fastl0_experiment(experiment, LZERO_OUTPUT_PATH, taus=taus)

    end = time.time()
    print(end - start)

# %%


def plot_compare_events_tau(expt_id):
    # TODO needs to be more generic
    h5 = "/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/events_fast10/1188653069.hdf5"

    # load h5
    h5s = glob.glob("/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/events_fast10/*.hdf5")
    h5s.sort()
    experiment = get_ophys_expt(expt_id)
    dff = experiment.dff_traces.loc[1189582700, "dff"]  # check if the right dff

    fig, ax = plt.subplots(5, 1, figsize=(40, 10))
    for i, h5 in enumerate(h5s):
        # load and plot h5,
        with h5py.File(h5, 'r') as f:
            events = f['events'][()]

        # plot on twinx
        ax[i].plot(events[0, :], label=os.path.basename(h5))
        ax[i].legend()

        # plot dff on twinx
        ax[i].twinx().plot(dff, color='r', alpha=.5)

        ax[i].set_xlim([35000, len(dff)])


# %% GENERATE EVENTS FOR ALL EXPERIMENTS (NEW_DFF, OLD_DFF, NP_CORRECTED)


def get_h5s_from_pipe_dev(pipe_dev_path):
    # load a table with experiments
    # expt_table = pd.read_pickle(pipe_dev_path / 'three_mice_ids_df.pkl')
    # ids = expt_table.ophys_experiment_id.values

    expt_table = start_lamf_analysis()
    ids = expt_table.index.values

    # load h5s with ids in fn
    # keys : ['cell_roi_id', 'cell_specimen_id', 'new_dff', 'np_corrected',
    #         'old_dff', 'timestamps']
    h5s = []
    for id in ids:
        h5s.extend(glob.glob(str(pipe_dev_path / f"{id}_new_dff.h5")))

    return h5s


# %% PLOT EVENT COMPARISON (NEW DFF, OLD DFF, NP-CORRECTED)

def plot_compare_events_from_trace_types(expt_id: int, cell_ind: int, dff_path: Path,
                                         events_path: Path, xlim: list = None):
    """Plot dff and events for a given experiment and cell index,
     for each trace type (new_dff, old_dff, np_corrected)

    Parameters
    ----------
    expt_id : int
        ophys_experiment_id
    cell_ind : int
        index of cell in dff array
    dff_path : str
        path to dff_h5, dff_h5 should be a dict with keys:
        ['cell_roi_id', 'cell_specimen_id', 'new_dff', 'np_corrected']
        generated by Jinho code (TODO: add link)
    events_path : str
        path to events h5, events h5 should be a dict with keys:
        ['events', 'lambdas', 'noise_stds', 'roi_names']
        Generated by ophys_etl_pipelines.modules.event_detection.schemas
    xlim : tuple, optional
        x limits for plot. The default is None.

    Returns
    -------
    None.

    Notes
    -----
    new_dff: see Jinho's code (TODO: JK add link)
    old_dff: pipeline dff used for visual behavior
    np_corrected: demixed traces with neuropil correction applied
    """

    # load h5s for expt_id (1 h5 for )
    events_h5s = glob.glob(str(events_path / f"{expt_id}_*.h5"))

    if len(events_h5s) == 0:
        print("no events found for this expt_id")
        return

    # load events_h5s into dict with key_name
    events_dict = {}
    for h5 in events_h5s:
        key_name = '_'.join(Path(h5).name.split("_")[1:3])
        # key_name = Path(h5).name.split("_")[1].replace("-", "_")
        with h5py.File(h5, 'r') as f:
            events_dict[key_name] = f['events'][()]

    # load dff h5
    dff_h5 = glob.glob(str(dff_path / f"{expt_id}_*.h5"))[0]

    # load h5 as dictionary
    trace_dict = {}
    with h5py.File(dff_h5, 'r') as f:
        for key in f.keys():
            trace_dict[key] = f[key][()]

    if trace_dict is None:
        print("no traces found for this expt_id")
        return

    trace_types = ["np_corrected", "new_dff", "old_dff"]
    # get 3 colors froms seaborn

    trace_dict['np_corrected'] = stats.zscore(trace_dict['np_corrected'], axis=1)

    trace_colors = sns.color_palette("husl", 3)
    # trace_types = ["np_corrected"]
    # plot traces and events on twin axes, for each trace type
    fig, axs = plt.subplots(3, 1, figsize=(20, 10))
    for i, (trace_type, color) in enumerate(zip(trace_types, trace_colors)):
        ax = axs[i]
        ax.plot(trace_dict["timestamps"], trace_dict[trace_type][cell_ind], color=color)
        # ax.plot(trace_dict["timestamps"], trace_dict[trace_type].mean(axis=0), color="black")
        ax.set_title(trace_type)
        ax.set_ylabel(f"{trace_type} (a.u.)")
        ax.set_xlabel("time (s)")

        ax2 = ax.twinx()
        # use same timestamps
        ax2.plot(trace_dict["timestamps"], events_dict[trace_type][cell_ind], color="orange")
        ax2.set_ylabel("events FastL0, tau=0.715")
        # ax2.set_yticks([])
        # ax2.set_xlim([3000, int(np.max(trace_dict["timestamps"]))])
        ax2.set_xlim([1000, 1300])

    plt.tight_layout()


if __name__ == "__main__":
    generate_fastl0_events_for_trace_types(trace_keys=["new_dff"],
                                           taus=[0.715],
                                           output_path_root=LZERO_OUTPUT_PATH,
                                           input_path_root=NEW_DFF_PATH,
                                           frame_rate=10.7,
                                           nworkers=-1)
