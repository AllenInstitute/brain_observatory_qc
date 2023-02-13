import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path

from brain_observatory_analysis.ophys.experiment_loading import start_lamf_analysis

# import ..suite2p_param_search

parser = argparse.ArgumentParser(description='running sbatch for new_dff_for_oeid.py')
parser.add_argument('--env-path', type=str, default='//data/learning/mattd/miniconda3/envs/dev', metavar='path to conda environment to use')
parser.add_argument('--dry-run', action='store_true', default=False, help='dry run')

NONRIGID_OUTPUT_PATH = Path('/allen/programs/mindscope/workgroups/learning/pipeline_validation/nonrigid_registration_parameter_search')


if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    dry_run = args.dry_run
    base_dir = Path('//home/matt.davis/code/mindscope_qc/mindscope_qc/pipeline_dev/scripts')
    python_file = base_dir / 'suite2p_param_search.py'

    job_dir = NONRIGID_OUTPUT_PATH
    stdout_location = job_dir / 'job_records'

    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)

    # expt_table = start_lamf_analysis()

    # # Get ophys_experiment_ids from a table
    # ophys_experiment_ids = expt_table.index.values

    # # find ids in job dir already prcoessed {id}_new_dff.h5
    # processed_ids = []
    # for fn in os.listdir(job_dir):
    #     if fn.endswith('_new_dff.h5'):
    #         processed_ids.append(int(fn.split('_')[0]))

    # # remove processed ids from ophys_experiment_ids
    # ophys_experiment_ids = [oeid for oeid in ophys_experiment_ids if oeid not in processed_ids]


    ophys_experiment_ids = [1227116668] # unstable JK manual
    ophys_experiment_ids = [1171605650] # jk stable
    # print number of jobs
    print('number of jobs = {}'.format(len(ophys_experiment_ids)))

    job_count = 0

    # rerun = False

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

        job_title = 'oeid_{}'.format(oeid)
        walltime = '2:00:00'
        mem = '100G'
        tmp = '100G',
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
        cpus_per_task = 32

        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=cpus_per_task,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output=output,
            tmp=tmp,
            partition="braintv"
        )

        # get job id from slurm
        job_id = slurm.JOB_ARRAY_MASTER_ID
        scratch_dir = f"/scratch/fast/{job_id}"

        # WAWRNING: this is inaccurate when interative mode on hpc
        job_id_str = '$SLURM_JOB_ID'  # get the job id from the environment variable
        #job_id_str = str(7745718)
        args_string = f"{oeid} {job_id_str} -t"

        print('JOB_ID = {}'.format(job_id))

        slurm.sbatch('{} {} {}'.format(
                     python_executable,
                     python_file,
                     args_string,))
        time.sleep(0.01)
