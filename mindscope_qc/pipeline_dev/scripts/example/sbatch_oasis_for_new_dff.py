import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry_run', action='store_true', default=False, help='dry run')
parser.add_argument('--test_run', action='store_true', default=False, help='test one parameter set')

ROOT_DIR = Path("/allen/programs/mindscope/workgroups/learning/pipeline_validation")

if __name__ == '__main__':
    args = parser.parse_args()
    dry_run = args.dry_run
    test_run = args.test_run
    python_executable = f"{args.env_path}/bin/python"

    # py file
    python_file = Path('//home/matt.davis/code/mindscope_qc/mindscope_qc/pipeline_dev/scripts/oasis_module.py')

    # job directory
    job_dir = ROOT_DIR / "events" / "oasis_v1"
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    # get h5 files
    expt_ids = start_lamf_analysis().index.values
    h5_list = []
    for expt_id in expt_ids:
        h5_list.append(ROOT_DIR / "dff" / f"{expt_id}_new_dff.h5")

    if dry_run:
        print('dry run, exiting')
        exit()

    job_count = 0
    print('number of jobs = {}'.format(len(h5_list)))

    if test_run:
        h5_list = h5_list[0:1]
    
    for h5 in h5_list:
        oeid = h5.stem.split('_')[0]
        print(h5)
        job_count += 1
        print(f'starting cluster job for {oeid}, job count = {job_count}')
        
        job_title = f'{oeid}_oasis_v1'
        walltime = '1:00:00'
        mem = '2G'
        #tmp = '3G',
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
        cpus_per_task = 1
        print(output)

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=cpus_per_task,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            #tmp=tmp,
            partition="braintv"
        )

        args_string = f"--h5_path {h5} --hpc"
        print(args_string)

        sbatch_string = f"{python_executable} {python_file} {args_string}"
        print(sbatch_string)
        slurm.sbatch(sbatch_string)
        time.sleep(0.01)
# 1171605650 $SLURM_JOB_ID -t