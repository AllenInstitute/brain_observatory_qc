import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

parser = argparse.ArgumentParser(description='running sbatch for run_oasis_h5path.py')
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc', metavar='path to conda environment to use')

if __name__ == '__main__':
    args = parser.parse_args()
    ### IMPORTANT  # noqa: E266
    # Check OASIS path in oasis_module.py (brain_observatory_qc/pipeline_dev/script)
    ###
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))

    ###
    ### Change the paths  # noqa: E266
    ###
    base_dir = Path('/home/jinho.kim/Github/mindscope_qc/brain_observatory_qc/pipeline_dev/script/example/')
    python_file = base_dir / 'run_oasis_h5path.py'
    job_dir = Path('//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho/data/VB_data/event_oasis/')
    stdout_location = job_dir / 'job_records'
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)

    job_title = 'event_oasis'
    walltime = '3:00:00'
    mem = '200gb'
    cpus_per_task = 100
    job_id = Slurm.JOB_ARRAY_ID
    job_array_id = Slurm.JOB_ARRAY_MASTER_ID
    output = stdout_location / f'{job_array_id}_{job_id}.out'

    # instantiate a SLURM object
    slurm = Slurm(
        cpus_per_task=cpus_per_task,
        job_name=job_title,
        time=walltime,
        mem=mem,
        output=output,
        partition="braintv"
    )

    args_string = f'--num_core {cpus_per_task}'
    slurm.sbatch('{} {} {}'.format(
                 python_executable,
                 python_file,
                 args_string,))
    time.sleep(0.01)
