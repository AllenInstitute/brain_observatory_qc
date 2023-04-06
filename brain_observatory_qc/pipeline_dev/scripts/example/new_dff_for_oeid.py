from brain_observatory_analysis.ophys.experiment_loading import get_ophys_expt
import time

import argparse

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='generate new dff using dev object loading')
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
    t0 = time.time()
    print(f'Processing ophys_experiment_id {oeid}')
    get_ophys_expt(oeid, dev=True)
    t1 = time.time()
    print(f'Done in {(t1-t0)/60:.1f} min.')
