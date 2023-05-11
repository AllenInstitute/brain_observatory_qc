from pathlib import Path
from glob import glob
from brain_observatory_qc.pipeline_dev.scripts.oasis_module import generate_oasis_events_for_h5_path

import multiprocessing as mp
from functools import partial

### IMPORTANT  # noqa: E266
# Change OASIS path in oasis_module.py
###

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_core', type=int, default=None,
                    metavar='number of cores for multiprocessing')

if __name__ == "__main__":
    args = parser.parse_args()
    num_core = args.num_core

    ### IMPORTANT  # noqa: E266
    # Change directory paths
    ###
    h5_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\dff')
    h5_files = glob(str(h5_dir / '*.h5'))

    ### IMPORTANT  # noqa: E266
    # Change directory paths
    ###
    out_path = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\event_oasis')
    out_files = glob(str(out_path / '*.h5'))

    # dff_oeids = [int(Path(f).stem.split('_')[0]) for f in h5_files]
    # event_oeids = [int(Path(f).stem.split('_')[0]) for f in out_files]

    # # Get the oeid that has dff but no event
    # run_oeids = np.setdiff1d(dff_oeids, event_oeids)
    # run_files = [f for f in h5_files if int(Path(f).stem.split('_')[0]) in run_oeids]
    run_files = h5_files

    func = partial(generate_oasis_events_for_h5_path,
                   out_path=out_path,
                   trace_type='new_dff',
                   estimate_parameters=True,
                   qc_plot=True)

    if num_core is None:
        with mp.Pool() as p:
            p.map(func, run_files)
    else:
        with mp.Pool(num_core) as p:
            p.map(func, run_files)
