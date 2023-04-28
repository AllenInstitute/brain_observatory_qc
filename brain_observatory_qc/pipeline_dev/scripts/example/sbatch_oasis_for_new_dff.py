import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
import pandas as pd
from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry_run', action='store_true', default=False, help='dry run')
parser.add_argument('--test_run', action='store_true', default=False, help='test one parameter set')

ROOT_DIR = Path("/allen/programs/mindscope/workgroups/learning/pipeline_validation")
# bash check file count in a directory
# ls -l | wc -l
if __name__ == '__main__':
    args = parser.parse_args()
    dry_run = args.dry_run
    test_run = args.test_run
    python_executable = f"{args.env_path}/bin/python"

    # py file
    python_file = Path('//home/matt.davis/code/mindscope_qc/mindscope_qc/pipeline_dev/scripts/oasis_module.py')  # update to brain_observatory_qc

    # job directory
    # job_dir = ROOT_DIR / "events" / "oasis_v1"
    job_dir = ROOT_DIR / "events" / "oasis_nrsac_v1"
    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    expts = start_lamf_analysis()

    mouse_names = ["Copper", "Silicon", "Titanium", "Bronze", "Gold", "Silver", "Mercury", "Aluminum", "Iron", "Cobalt"]
    expts.mouse_name = pd.Categorical(expts.mouse_name, categories=mouse_names, ordered=True)
    expts = expts.sort_values(by="mouse_name")

    expt_ids = expts.index.values

    # OLD way
    # h5_list = []
    # for expt_id in expt_ids:
    #     h5_list.append(ROOT_DIR / "dff" / f"{expt_id}_new_dff.h5")

    if dry_run:
        print('dry run, exiting')
        exit()

    job_count = 0
    print('number of jobs = {}'.format(len(expt_ids)))

    if test_run:
        expt_ids = expt_ids[0:2]

    for oeid in expt_ids:
        job_count += 1
        print(f'starting cluster job for {oeid}, job count = {job_count}')

        job_title = f'{oeid}_oasis_v1'
        walltime = '00:15:00'
        mem = '1G'
        # tmp = '3G',
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
            # tmp=tmp,
            partition="braintv"
        )

        # args_string = f"--h5_path {h5} --hpc"
        args_string = f"--oeid {oeid} --hpc"
        print(args_string)

        sbatch_string = f"{python_executable} {python_file} {args_string}"
        print(sbatch_string)
        slurm.sbatch(sbatch_string)
        time.sleep(0.01)
