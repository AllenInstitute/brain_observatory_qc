import time, os
from pathlib import Path
import depth_estimation_module as dem

import argparse
parser = argparse.ArgumentParser(description='estimate fov matched-plane')
parser.add_argument(
    'ophys_experiment_id',
    type=int,
    default=0,
    metavar='experiment',
    help='ophys_experiment_id'
)

if __name__ == '__main__':
    args = parser.parse_args()
    oeid = args.ophys_experiment_id
    print(f'Processing ophys_experiment_id {oeid}')
    t0 = time.time()
    oeid_ref = dem.find_first_experiment_id_from_ophys_experiment_id(oeid)
    if oeid_ref == oeid:
        dem.estimate_matched_plane_from_same_exp(oeid)
    else:
        dem.estimate_matched_plane_from_ref_exp(oeid, oeid_ref)
        dem.estimate_matched_plane_from_same_exp(oeid)
    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.1f} min.')
