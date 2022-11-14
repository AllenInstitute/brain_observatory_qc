import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc

parser = argparse.ArgumentParser(description='running sbatch for estimate_fov_plane_each_exp.py')
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc', metavar='path to conda environment to use')

if __name__=='__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    base_dir = Path('/home/jinho.kim/Github/jinho_data_analysis/QC/FOV_match/script/')
    python_file = base_dir / 'estimate_fov_plane_each_exp.py'
    job_dir = Path('//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/QC/FOV_matching/LAMF_Ai195_3/')
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    # Get ophys_experiment_ids from a txt file
    txt_fn = 'ophys_experiment_ids.txt'
    txt_fp = job_dir / txt_fn
    with open(txt_fp, 'r') as file:
        lines = file.readlines()
    ophys_experiment_ids = []
    for line in lines:
        ophys_experiment_ids.append(int(line.strip()))
    # mouse_ids = ['612764', '629294', '608368']
    # cache = bpc.from_lims()
    # ophys_experiment_table = cache.get_ophys_experiment_table(passed_only=False)
    # lamf_ai195_table = ophys_experiment_table.query("mouse_id in @mouse_ids")
    # ophys_experiment_ids = lamf_ai195_table.index.values

    job_count = 0

    job_string = "{}"

    rerun = False
    for oeid in ophys_experiment_ids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(oeid, job_count))
        job_title = 'ophys_experiment_id_{}'.format(oeid)
        walltime = '0:25:00'
        mem = '200gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
    
        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=1,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output= output,
            partition="braintv"
        )

        args_string = job_string.format(oeid)
        slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
        )
        time.sleep(0.01)