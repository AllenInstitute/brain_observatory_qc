import mindscope_qc.data_access.from_lims as from_lims
from suite2p import run_s2p, default_ops
from pathlib import Path
import os
import argparse

# set your options for running
NONRIGID_OUTPUT_PATH = Path('/allen/programs/mindscope/workgroups/learning/pipeline_validation/nonrigid_registration_parameter_search')
SCRATCH_PATH = Path('')

# argparse ophys_experiment_id
parser = argparse.ArgumentParser()
parser.add_argument("ophys_experiment_id", type=int, help="ophys_experiment_id")
parser.add_argument("job_id", type=int, help="path to scratch space")
parser.add_argument("--test_run", "-t", action="store_true", help="test_run")
parser.add_argument("--grid_seach", "-g", action="store_true", help="grid seach")


def run_s2p_with_params(ophys_experiment_id, scratch_path, params):
    """Run suite2p on a set of input parameters
 
    Note: Need to be on mjd_dev of Suite2p bracat nch, which contains h5 file handling
    """
    ops = default_ops()

    scratch_path = Path("/scratch/fast") / str(scratch_path) 

    print(f"SCRATCH PATH = {scratch_path}")
    # create string out of params key/value pairs
    param_string = ""
    for key, value in params.items():
        param_string += f"{key}={value}_"
    save_folder = f"{ophys_experiment_id}_{param_string[:-1]}"

    fast_disk_path = scratch_path / save_folder

    # get the h5 file
    h5_file = str(from_lims.get_motion_corrected_movie_filepath(ophys_experiment_id))

    # ana = "ana" if params["anatomical_only"] else ''

    # if not os.path.exists(SAVE_PATH):
    #    os.makedirs(SAVE_PATH)

    if not os.path.exists(fast_disk_path):
        os.makedirs(fast_disk_path)

    # # check if bin file exists (when resuing same bin files)
    # fast_bin_path = os.path.join(fast_path_final, "plane0/data.bin")
    # if not os.path.exists(fast_bin_path):
    #     fast_bin_path = None
    # else:
    #     fast_bin_path = fast_path_final
    fast_bin_path = None  # for resuing bin files
    db = {
        'h5py': h5_file,
        'h5py_key': 'data',
        'look_one_level_down': False,
        'data_path': [],
        'do_registration': 1,
        'save_path0': str(NONRIGID_OUTPUT_PATH),
        'save_folder': save_folder,
        'fast_disk': str(fast_disk_path),  # CHANGE TO FAST AFTER TESTING
        'delete_bin': False,

        # data settings
        'fs': 11,
        'tau': params['tau'],
        # 'frames_include':250

        # reg
        'nonrigid': True,
        'block_size': params['bs'],
        'smooth_time_sigma': params['sts'],
        'maxregshiftNR': params['mnrs'],

        # ROI
        'diameter': params["dia"],
        'roidetect': 1,
        'connected': True,  # set to 0 for dendrites/axons
        'sparse_mode': False,
        'spatial_scale': 0,  # default 0
        'threshold_scaling': params['ts'],  # default: 5.0 (high less ROIs, low more ROIs)
        'max_overlap': 1.0,  # default: .75, 1.0 = no overlap allowed
        'high_pass': 100,  # default: 100 (less than 10 for 1P data)
        'smooth_masks': True,
        'denoise': True,  # default: False, use for sparse mode
        'nbinned': 500,  # defulat: 5000

        # extraction
        'spikedetect': True,

        # cell pose
        'anatomical_only': params["ana"],  # use cellpose masks from mean image (no functional segmentation)
        'cellprob_threshold': 0.0,  # cellprob_threshold for cellpose (if anatomical_only > 1)
        'flow_threshold': 1.5,  # flow_threshold for cellpose (if anatomical_only > 1)

        # extra setting added by MJD
        "h5_file_list": None,
        "fast_bin_path": fast_bin_path,  # None, #ast_path,
        'extraction': True
    }

    opsEnd = run_s2p(ops=ops, db=db)

    return opsEnd


if __name__ == "__main__":

    args = parser.parse_args()
    expt_id = args.ophys_experiment_id
    job_id = args.job_id
    test = args.test_run
    grid_search = args.grid_seach

    assert not (test and grid_search), "Cannot specify both test and grid_search"
    assert expt_id is not None, "Must specify ophys_experiment_id"

    bss = [[8, 8], [32, 32], [64, 64], [96, 96], [128, 128]]
    maxnrs = [3, 6, 9, 12]
    sts = [0, 1, 2, 5, 10]

    if test:
        print("TEST RUN")
        params = {"sts": 0,
                  "bs": [64, 64],
                  "mnrs": 5,
                  "ts": 2.0,
                  "dia": 15,
                  "tau": 0.7,
                  "ana": 0}

        finals_ops = run_s2p_with_params(expt_id, job_id, params)

    if grid_search:
        for bs in bss:
            for maxnr in maxnrs:
                for st in sts:
                    params = {"sts": st,
                              "bs": bs,
                              "mnrs": maxnr,
                              "ts": 2.0,
                              "dia": 15,
                              "tau": 0.7,
                              "ana": 0}

                    finals_ops = run_s2p_with_params(expt_id, job_id, params)
