import os
import argparse
import time
from simple_slurm import Slurm
from pathlib import Path
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc

parser = argparse.ArgumentParser(
    description='running sbatch for get_new_dff_from_oeid.py')
parser.add_argument('--env-path', type=str, default='/home/jinho.kim/anaconda3/envs/allenhpc',
                    metavar='path to conda environment to use')


def read_oeid_list(save_dir, oeid_list_txt_fn):
    with open(save_dir / oeid_list_txt_fn) as f:
        lines = f.readlines()
    oeid_list = []
    for line in lines:
        oeid_list.append(int(line.split('\\')[0]))
    return oeid_list


def already_processed(ophys_experiment_id, job_dir, rerun=False):
    new_dff_fn = job_dir / f'{ophys_experiment_id}_new_dff.h5'

    processed_flag = False
    if os.path.isfile(new_dff_fn):
        if rerun is False:
            print(f'{oeid} processed already. Skipping.')
            processed_flag = True

    return processed_flag


if __name__ == '__main__':
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    base_dir = Path(
        '/home/jinho.kim/Github/mindscope_qc/brain_observatory_qc/pipeline_dev/scripts/example/')
    python_file = base_dir / 'get_new_dff_from_oeid.py'

    # #
    # job_dir = Path('//allen/programs/mindscope/workgroups/learning/pipeline_validation/dff')
    # job_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\GH_data\dff'.replace('\\', '/'))
    
    job_dir = Path(r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\Jinho\data\VB_data\dff'.replace('\\', '/'))

    stdout_location = job_dir / 'job_records'
    stdout_location.mkdir(parents=True, exist_ok=True)

    # # Get ophys_experiment_ids from the text file
    # oeid_list_txt_fn = f'oeid_to_run_221021.txt'
    # oeid_list = read_oeid_list(job_dir, oeid_list_txt_fn)

    # # Get ophys_experiment_ids from a pkl file
    # oeid_list_pkl_fn = f'three_mice_ids_df.pkl'
    # oeid_df = pd.read_pickle(job_dir.parent / oeid_list_pkl_fn)
    # oeid_list = oeid_df.ophys_experiment_id.values
    # load_dir = Path(
    #     '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/Jinho')
    # load_fn = 'GH_experiment_table_directory_info.pkl'
    # GH_experiments_dir_info = pd.read_pickle(load_dir / f'{load_fn}')
    # oeid_list = GH_experiments_dir_info.ophys_experiment_id.values

    # Get ophys_experiment_ids from lims
    cache = bpc.from_lims()
    exp_table = cache.get_ophys_experiment_table(passed_only=True)
    # target_project_code_list = ['VisualBehaviorMultiscope4areasx2d']
    target_project_code_list = ['VisualBehaviorTask1B', 'VisualBehavior', 'VisualBehaviorMultiscope']
    target_exp_table = exp_table[exp_table.project_code.isin(target_project_code_list)]
    oeid_list = target_exp_table.index.values

    # Collect already processed oeids
    processed_oedid_list = []
    for fn in os.listdir(job_dir):
        if fn.endswith('new_dff.h5'):
            processed_oedid_list.append(int(fn.split('_')[0]))
    
    # Remove already processed oeids
    oeid_list = [oeid for oeid in oeid_list if oeid not in processed_oedid_list]

    job_count = 0
    job_string = "{} {}"

    rerun = True
    for oeid in oeid_list:
        if already_processed(oeid, job_dir, rerun=rerun):
            print(f'{oeid} processed already. Skipping.')
        else:
            job_count += 1
            print('starting cluster job for {}, job count = {}'.format(
                oeid, job_count))
            job_title = 'ophys_experiment_id_{}'.format(oeid)
            walltime = '2:00:00'
            cpus_per_task = 50
            mem = '300gb'
            job_id = Slurm.JOB_ARRAY_ID
            job_array_id = Slurm.JOB_ARRAY_MASTER_ID
            output = stdout_location / f'{job_array_id}_{job_id}_{oeid}.out'

            # instantiate a SLURM object
            slurm = Slurm(
                cpus_per_task=cpus_per_task,
                job_name=job_title,
                time=walltime,
                mem=mem,
                output=output,
                partition="braintv"
            )

            args_string = job_string.format(oeid, str(job_dir))
            slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
            )
            time.sleep(0.01)
