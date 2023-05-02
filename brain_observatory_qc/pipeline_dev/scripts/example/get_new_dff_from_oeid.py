
import argparse
import time
import os
from pathlib import Path
from brain_observatory_qc.pipeline_dev import calculate_new_dff

parser = argparse.ArgumentParser(
    description='dff calculation process across experiments')
parser.add_argument(
    'ophys_experiment_id',
    type=int,
    default=0,
    metavar='experiment',
    help='ophys_experiment_id'
)
parser.add_argument(
    'save_dir',
    type=str,
    default='//allen/programs/braintv/workgroups/nc-ophys/Jinho/hpc_temp',
    metavar='save_directory',
    help='directory to save the results, in str'
)

if __name__ == '__main__':
    args = parser.parse_args()
    oeid = args.ophys_experiment_id
    print(f'Processing ophys_experiment_id {oeid}')
    t0 = time.time()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = save_dir / f'tmp_{oeid}'
    # save_dir = Path(r'\\allen\programs\mindscope\workgroups\learning\pipeline_validation\dff'.replace('\\','/'))
    # Get new dff DataFrame
    new_dff_df, timestamps = calculate_new_dff.get_new_dff_df(oeid, tmp_dir=tmp_dir)
    # Save as h5 file, because of the timestamps
    calculate_new_dff.save_new_dff_h5(save_dir, new_dff_df, timestamps, oeid)
    # Draw figures and save them
    calculate_new_dff.draw_fig_new_dff(save_dir, new_dff_df, timestamps, oeid)
    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.1f} min.')
