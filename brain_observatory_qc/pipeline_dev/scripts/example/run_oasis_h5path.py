from pathlib import Path
from glob import glob
from brain_observatory_qc.pipeline_dev.scripts.oasis_module import generate_oasis_events_for_h5_path

import multiprocessing as mp
from functools import partial

### IMPORTANT
# Change OASIS path in oasis_module.py
###

if __name__ == "__main__":
    h5_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\dff')
    h5_files = glob(str(h5_dir / '*.h5'))
    out_path = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\event_oasis')

    func = partial(generate_oasis_events_for_h5_path,
                   out_path=out_path,
                   trace_type='new_dff',
                   estimate_parameters=True,
                   qc_plot=True)

    with mp.Pool(10) as p:
        p.map(func, h5_files)