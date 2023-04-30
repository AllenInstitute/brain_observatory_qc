from pathlib import Path
from glob import glob
from brain_observatory_qc.pipeline_dev.scripts.oasis_module import generate_oasis_events_for_h5_path
import numpy as np

import multiprocessing as mp
from functools import partial

### IMPORTANT
# Change OASIS path in oasis_module.py
###

if __name__ == "__main__":
    h5_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\dff')
    h5_files = glob(str(h5_dir / '*.h5'))
    out_path = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\event_oasis')
    out_files = glob(str(out_path / '*.h5'))

    out_files = glob(str(out_path / '*.h5'))

    dff_oeids = [int(Path(f).stem.split('_')[0]) for f in h5_files]
    event_oeids = [int(Path(f).stem.split('_')[0]) for f in out_files]

    # Get the oeid that has dff but no event
    run_oeids = np.setdiff1d(dff_oeids, event_oeids)
    run_files = [f for f in h5_files if int(Path(f).stem.split('_')[0]) in run_oeids]

    func = partial(generate_oasis_events_for_h5_path,
                   out_path=out_path,
                   trace_type='new_dff',
                   estimate_parameters=True,
                   qc_plot=True)

    with mp.Pool(10) as p:
        p.map(func, run_files)