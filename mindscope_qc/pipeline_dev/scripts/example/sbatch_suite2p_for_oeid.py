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

NONRIGID_OUTPUT_PATH = Path('//allen/programs/mindscope/workgroups/learning/pipeline_validation/nonrigid_registration_parameter_search')


# 48.3 k frames, NR128, c32, 2300 secs 

if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    dry_run = args.dry_run
    test_run = args.test_run
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

    ophys_experiment_ids = [1227116668]  # unstable JK manual
    # ophys_experiment_ids = [1171605650]  # jk stable
    oeid = ophys_experiment_ids[0]
    print(oeid)

    # print number of jobs
    print('number of jobs = {}'.format(len(ophys_experiment_ids)))

    job_count = 0

    if dry_run:
        print('dry run, exiting')
        exit()

    # select 10 ids to run
    # ophys_experiment_ids = ophys_experiment_ids[:10]

    bss = [128, 64, 32, 8]
    maxnrs = [3, 6, 9, 12]
    sts = [0, 1, 2, 5, 10]

    # make a list of all combinations of bs, maxnr, and st
    params = []
    for bs in bss:
        for maxnr in maxnrs:
            for st in sts:
                params.append((bs, maxnr, st))

    if test_run:
        params = params[:1]

    for param in params:
        print(param)
        job_count += 1
        print(f'starting cluster job for {oeid}, job count = {job_count}')
        
        bs, mnrs, sts = param
        job_title = f'oeid_{oeid}_param_bs_{bs}_mnrs_{mnrs}_sts_{sts}'
        walltime = '1:30:00'
        mem = '100G'
        tmp = '75G',
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'
        cpus_per_task = 32
        print(output)

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
        # job_id_str = str(7745718)

        bs, mnrs, sts = param
        args_string = f"{oeid} {job_id_str} {bs} {mnrs} {sts}"
        print(args_string)

        sbatch_string = f"{python_executable} {python_file} {args_string}"
        print(sbatch_string)
        slurm.sbatch(sbatch_string)
        time.sleep(0.01)
# 1171605650 $SLURM_JOB_ID -t