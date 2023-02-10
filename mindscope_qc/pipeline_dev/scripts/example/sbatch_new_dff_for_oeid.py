import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry-run', action='store_true', default=False, help='dry run')

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    dry_run = args.dry_run
    base_dir = Path('//home/matt.davis/code/mindscope_qc/mindscope_qc/pipeline_dev/scripts/example/')
    python_file = base_dir / 'new_dff_for_oeid.py'
    job_dir = Path('//allen/programs/mindscope/workgroups/learning/pipeline_validation/dff')
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)

    # Get ophys_experiment_ids from a txt file
    # txt_fn = 'ophys_experiment_ids.txt'
    # txt_fp = job_dir / txt_fn
    # with open(txt_fp, 'r') as file:
    #     lines = file.readlines()
    # ophys_experiment_ids = []
    # for line in lines:
    #     ophys_experiment_ids.append(int(line.strip()))

    expt_table = start_lamf_analysis()

    # Get ophys_experiment_ids from a table
    ophys_experiment_ids = expt_table.index.values

    # find ids in job dir already prcoessed {id}_new_dff.h5
    processed_ids = []
    for fn in os.listdir(job_dir):
        if fn.endswith('_new_dff.h5'):
            processed_ids.append(int(fn.split('_')[0]))

    # remove processed ids from ophys_experiment_ids
    ophys_experiment_ids = [oeid for oeid in ophys_experiment_ids if oeid not in processed_ids]

    # print number of jobs
    print('number of jobs = {}'.format(len(ophys_experiment_ids)))

    job_count = 0

    job_string = "{}"

    rerun = False

    # set dry_run to True to test the script
    dry_run = False

    if dry_run:
        print('dry run, exiting')
        exit()

    # select 10 ids to run
    # ophys_experiment_ids = ophys_experiment_ids[:10]

    for oeid in ophys_experiment_ids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(oeid, job_count))
        job_title = 'ophys_experiment_id_{}'.format(oeid)
        walltime = '0:25:00'
        mem = '10gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=1,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            partition="braintv"
        )

        args_string = job_string.format(oeid)
        slurm.sbatch('{} {} {}'.format(
                     python_executable,
                     python_file,
                     args_string,))
        time.sleep(0.01)
