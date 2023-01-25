import h5py
import pandas as pd
import numpy as np
import mjd_dev.image_utils as iu

import ophys_etl.modules.suite2p_registration.suite2p_utils as su
from pathlib import Path
from tifffile import tifffile
from typing import Union
import glob

# move to learning when more space
GIFS_FOLDER = '/data/learning/pipeline_validation/nonrigid_registration/'
GIFS_FOLDER = "/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/non_rigid_gifs/"
GIFS_FOLDER = "/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/rigid_gifs/"


# move to learning when more space
OG_GIFS_FOLDER = "/allen/programs/braintv/workgroups/nc-ophys/Doug/matt/og_rigid_gifs/"

# JK generated these files. Can use this or calculated the OG motion correction
RIGID_SEGMENT_DIR = '/data/learning/pipeline_validation/fov_tilt/tif'


def get_acutance_df(nonrigid_fn_df: pd.DataFrame,
                    og_rigid: bool = False) -> pd.DataFrame:
    """
    For each fn load the h5 file and compute the acutance for each frame, and put each value in dataframe

    """

    EXTRA_CROP_PX = 15

    acutance_df = pd.DataFrame()
    acutance_mean_df = pd.DataFrame()

    df = get_rigid_df(nonrigid_fn_df)

    for i in range(len(nonrigid_fn_df)):
        fn = nonrigid_fn_df['fn'].iloc[i]
        with h5py.File(fn, 'r') as hr:
            data = hr['data'][:]

        oeid = nonrigid_fn_df['oeid'].iloc[i]

        # RIGID SEGMENT
        rigid_fn = df[df['eid'] == oeid]['tif_fn'].iloc[0]
        rigid_tif = tifffile.imread(rigid_fn)
        rigid_mean = np.mean(rigid_tif, axis=0)
        print(rigid_mean.shape)
        rigid_mean = iu.crop_image_edges(rigid_mean, EXTRA_CROP_PX)
        rigid_mean = rigid_mean / np.max(rigid_mean)
        rigid_acutance = su.compute_acutance(rigid_mean)

        # load orginal data and calc mean
        if og_rigid:
            og_rigid_mean = iu.get_corrected_movie_projection(oeid)
            og_rigid_mean = iu.crop_oeid(og_rigid_mean, oeid)
            og_rigid_mean = iu.crop_image_edges(og_rigid_mean, EXTRA_CROP_PX)
            og_rigid_mean = og_rigid_mean / np.max(og_rigid_mean)
            og_rigid_acutance = su.compute_acutance(og_rigid_mean)
        else:
            og_rigid_acutance = np.nan
            og_rigid_mean = np.nan

        if 'raw' in fn:
            data = iu.crop_oeid(data, oeid)

        # crop 15 px
        data = iu.crop_image_edges(data, 15)

        fn = fn.split('.')[0].split('/')[-1:][0]
        iu.save_gif(data, GIFS_FOLDER, fn)

        # iterate through each frame and compute acutance
        for j in range(data.shape[0]):
            acutance = su.compute_acutance(data[j, :, :])
            # rewrite with pd.concat

            # append acutance to dataframe
            acutance_df = acutance_df.append({'oeid': oeid, 'bs': nonrigid_fn_df['bs'].iloc[i], 'mc': nonrigid_fn_df['mc'].iloc[i],
                                              'raw': nonrigid_fn_df['raw'].iloc[i], 'acutance': acutance}, ignore_index=True)

        # take mean of data
        nr_img = np.mean(data, axis=0)
        nr_img = nr_img / np.max(nr_img)
        acutance = su.compute_acutance(nr_img)

        # append mean acutance to dataframe
        # (really messy way to build this dfs, could iterate the rows in loop above)
        acutance_mean_df = acutance_mean_df.append({'oeid': oeid, 'bs': nonrigid_fn_df['bs'].iloc[i], 'mc': nonrigid_fn_df['mc'].iloc[i],
                                                    'raw': nonrigid_fn_df['raw'].iloc[i], 'acutance': acutance, "img": nr_img,
                                                    'rigid_img': rigid_mean, 'rigid_acutance': rigid_acutance,
                                                    'og_rigid_mean': og_rigid_mean, 'og_rigid_acutance': og_rigid_acutance}, ignore_index=True)

        # calc diff
        acutance_mean_df['acutance_norm'] = acutance_mean_df['acutance'] / acutance_mean_df['rigid_acutance']

        # calc diff
        acutance_mean_df['acutance_og_rig_norm'] = acutance_mean_df['acutance'] / acutance_mean_df['og_rigid_acutance']

    return acutance_df, acutance_mean_df


def get_rigid_df(segment_fn_df):

    oeids = segment_fn_df.oeid.unique()

    RIGID_SEGMENT_DIR = '/data/learning/pipeline_validation/fov_tilt/tif'

    # get all tif files
    tif_list = glob.glob(str(Path(RIGID_SEGMENT_DIR) / '*.tif'))
    eid_list = [int(fn.split('/')[-1].split('_')[0]) for fn in tif_list]

    # put into df
    df = pd.DataFrame({'eid': eid_list, 'tif_fn': tif_list})

    # get ids that match oeids
    df = df[df['eid'].isin(oeids)]

    return df


def nonrigid_segmented_files_to_df(path):
    """Get the filepaths for all the nonrigid segmented files in a directory"""

    # get all files with 'segment' in all subdirectories
    segment_fn_list = glob.glob(f'{path}**/*segment.h5', recursive=True)

    segment_fn_list

    # put filenames into a dataframe
    nonrigid_fn_df = pd.DataFrame(segment_fn_list, columns=['fn'])
    nonrigid_fn_df['oeid'] = nonrigid_fn_df['fn'].apply(lambda x: int(x.split('/')[-2]))
    nonrigid_fn_df['bs'] = nonrigid_fn_df['fn'].apply(lambda x: int(x.split('_bs')[-1].split('_')[0]))
    nonrigid_fn_df['mc'] = nonrigid_fn_df['fn'].apply(lambda x: 'mc' in x)
    nonrigid_fn_df['raw'] = nonrigid_fn_df['fn'].apply(lambda x: 'raw' in x)

    return nonrigid_fn_df
