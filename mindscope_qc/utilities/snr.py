import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import functools
import multiprocessing as mp

import visual_behavior.data_access.from_lims as from_lims
import visual_behavior.data_access.utilities as utilities
from skimage.util import view_as_blocks
import tifffile

###############################################################################
# METRICS
###############################################################################
def compute_photon_flux(image: np.ndarray) -> float:
    """Compute the photon flux of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute photon flux of.

    Returns
    -------
    photon_flux : float
    """
    photon_flux = np.sqrt(np.mean(image.flatten()))

    return photon_flux

def compute_basic_snr(image: np.ndarray) -> float:
    """Compute the photon flux of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute basic snr of.
    
    Returns
    -------
    basic_snr : float
    """
    basic_snr = np.std(image)/np.mean(image)

    return basic_snr

def compute_image_contrast(image: np.ndarray, percentile_max: int =95, 
                           percentile_min: int =5) -> float:
    """Compute the contrast of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute contrast of.
    percentile_max : int
        Percentile to use for max.
    percentile_min : int
        Percentile to use for min.

    Returns
    -------
    contrast : float
    """
    Imax = np.percentile(image, percentile_max)
    Imin = np.percentile(image, percentile_min)
    contrast = (Imax-Imin)/(Imax+Imin)\

    return contrast 
#
def compute_acutance(image: np.ndarray,
                     cut_y: int = 0,
                     cut_x: int = 0) -> float:
    """Compute the acutance (sharpness) of an image.

    Note: function from ophys_etl_pipeline

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    cut_y : int
        Number of pixels to cut from the begining and end of the y axis.
    cut_x : int
        Number of pixels to cut from the begining and end of the x axis.
    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    if cut_y <= 0 and cut_x <= 0:
        cut_image = image
    elif cut_y > 0 and cut_x <= 0:
        cut_image = image[cut_y:-cut_y, :]
    elif cut_y <= 0 and cut_x > 0:
        cut_image = image[:, cut_x:-cut_x]
    else:
        cut_image = image[cut_y:-cut_y, cut_x:-cut_x]
    grady, gradx = np.gradient(cut_image)

    return (grady ** 2 + grady ** 2).mean()

def compute_block_snr(img: np.ndarray, block_shape: tuple = (128,128),
                      snr_metric: str = "basic", 
                      blocks_to_agg: tuple = (6,10),
                      return_block: bool = False):
    """Compute the SNR of nonoverlapping blocks of an image, return aggregate
    SNR from certrain blocks.
    
    Parameters
    ----------
    img : np.ndarray
        Image to compute SNR of.
    block_shape : tuple
        Shape of blocks to compute SNR of. For 512x512 input (128,128)
        is 16 blocks.
    snr_metric : str
        Type of SNR metric to compute.
    blocks_to_agg : tuple
        Start and end index of block to aggregate SNR (e.g (6,10) 
        will aggregate the middle 5 blocks of 16 total blocks).
    return_block : bool
        If True, return the blocks used to compute SNR.

    Returns
    -------
    mean_block_snr : float
        Mean SNR of blocks.
    """
    try:
        view = view_as_blocks(img, block_shape)
        flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

        # calculate basic SNR. TODO: add more metrics
        if snr_metric == "basic":
            block_snr = np.std(flatten_view, axis=2)/np.mean(flatten_view, axis=2)
        
        s,e = blocks_to_agg
        mid_snr_blocks = np.sort(block_snr.flatten())[s:e]
        mean_block_snr = np.mean(mid_snr_blocks)
    except:
        print(f"Failed blocks: {img.shape}")
        mean_block_snr = np.nan

    if return_block:
        return block_snr, mean_block_snr
    else:
        return mean_block_snr

###############################################################################
# IMAGE HANDLING
# Note: functions could be replaced by data-loader functions when available.
###############################################################################

def get_average_projection(expt_id):
    """NOTE: no docstring, function will be replaced by data-loader function"""
    try:
        avg_path = from_lims.get_average_intensity_projection_filepath(expt_id)
        img = plt.imread(avg_path)
    except:
        img = np.zeros((512,512))

    return img

def get_max_projection_hack(expt_id):
    """NOTE: no docstring, function will be replaced by data-loader function"""
    try:
        avg_path = from_lims.get_average_intensity_projection_filepath(expt_id)
        max_path = Path(str(avg_path).replace("average","maximum"))
        img = plt.imread(max_path)
    except:
        img = np.zeros((512,512))

    return img

def get_average_depth_image(expt_id):
    """NOTE: no docstring, function will be replaced by data-loader function"""
    try:
        expt_dir = utilities.get_ophys_experiment_dir(utilities.get_lims_data(expt_id))
        depth_path = glob.glob(f"{expt_dir}/*depth*.tif")# %%
        assert len(depth_path) == 1
        img = plt.imread(depth_path[0])
    except:
        print(f"No depth file found: {expt_id}")
        img = np.zeros((512,512))

    return img

def get_natalia_projection_path_df():
    """One off functions for grabbing pre-generated projections, may remove"""

    #sf = 'first_1000_frames' #subfolder
    sf = 'last_1000_frames'
    
    pojections_path = '/allen/programs/mindscope/workgroups/learning/SNR/{sf}"'

    # get files with "max" in the name
    max_files = glob.glob(f"{pojections_path}/*max_int.tif") # ignore offest
    # get files with "avg" in the name
    avg_files = glob.glob(f"{pojections_path}/*avg_int.tif") # ignore offest

    expt_ids = [int(f.split("/")[-1].split("_")[0]) for f in max_files]

    file_df = pd.DataFrame(zip(expt_ids,max_files,avg_files),
              columns=["ophys_experiment_id","max_file","avg_file"])

    return file_df

###############################################################################
# BUILD OUT DATAFRAMES
###############################################################################

def add_all_snr_metrics(expt_table: pd.DataFrame,
                        image_sources: list = ["limsAvg","limsMax","16FrameAvg"],
                        metrics: list = ["basic","acutance","medianBlocks",
                                         "photonFlux","contrast"]) -> pd.DataFrame:
    """Adds all the SNR metrics to an  experiment table

    Top level function:  add_all_snr_metrics() -> add_snr_col_parallel() 
    -> calc_snr()
    
    Parameters
    ----------
    expt_table : pandas.DataFrame
        Table of experiments to add SNR metrics to.
    image_sources : list
        List of image sources to add SNR metrics to.
        Options: ["limsAvg","limsMax","16FrameAvg","nataliaProjAvg",
                  "nataliaProjMax"]
    metrics : list
        List of metrics to add to the table.
        Options: ["basic","acutance","medianBlocks","photonFlux","contrast"]
        
    Returns
    -------
    expt_table : pandas.DataFrame
        Table of experiments with SNR metrics added."""

    for source in image_sources:
        for metric in metrics:
            print(source,": ", metric)
            expt_table = add_snr_col_parallel(expt_table,input_source=source,snr_metric=metric)

    return expt_table

def add_snr_col_parallel(expt_table: pd.dataframe, 
                        input_source: str = "16frame", 
                        snr_metric: str = "basic") -> pd.DataFrame:
    """Adds a SNR column to the ophys experiment table with multiprocessing
    
    Parameters
    ----------
    expt_table : pandas.DataFrame
        Table of experiments to add SNR metrics to.
    input_source : str
        Source of images to compute SNR of.
        See add_all_snr_metrics() for options
    snr_metric : str
        Type of SNR metric to compute.
        See add_all_snr_metrics() for options
        
    Returns
    -------
    expt_table : pandas.DataFrame
        Table of experiments with SNR metrics added.
    """

    df = expt_table.copy()
    expt_list = df.index 
   
    with mp.Pool(mp.cpu_count()) as p:
        func = functools.partial(calc_snr, input_source=input_source,
                                 snr_metric=snr_metric)
        # map keeps order (i think), so re-assigning to df should be OK.
        df['snr_'+input_source+'_'+snr_metric] = p.map(func,expt_list)

    return df

def calc_snr(expt_id, input_source ="16frame", snr_metric="basic"):
    """Calculates the SNR of given input_source and snr_metric for expt_id

    Parameters
    ----------
    expt_id : int
        Ophys experiment id to calculate SNR for.
    input_source : str
        Source of images to compute SNR of.
        See add_all_snr_metrics() for options
    snr_metric : str
        Type of SNR metric to compute.
        See add_all_snr_metrics() for options

    Returns
    -------
    snr : float
        SNR of the given input_source and snr_metric for expt_id
    """

    if input_source == "limsAvg":
        img = get_average_projection(expt_id)
    elif input_source == "limsMax":
        img = get_max_projection_hack(expt_id)
    elif input_source == "nataliaProjAvg":
        try:
            file_df = get_natalia_projection_path_df()
            file = file_df.query("ophys_experiment_id == \
                                @expt_id").avg_file.values[0]

            with tifffile.TiffFile(file, mode ='rb') as tiff:
                img = tiff.asarray()
        except:
            img = np.zeros((512,512))
            pass

    elif input_source == "nataliaProjMax":
        try:
            file_df = get_natalia_projection_path_df()
            file = file_df.query("ophys_experiment_id == \
                                @expt_id").max_file.values[0]

            with tifffile.TiffFile(file, mode ='rb') as tiff:
                img = tiff.asarray()
        except:
            img = np.zeros((512,512))
            pass
    elif input_source == "16FrameAvg":
        img = get_average_depth_image(expt_id)
    else:
        raise ValueError("input_source must be one of 'lims_avg','lims_max','16frame'")

    if snr_metric == "basic":
        return compute_basic_snr(img)
    elif snr_metric == "acutance":
        return compute_acutance(img)
    elif snr_metric == "medianBlocks":
        return compute_block_snr(img)
    elif snr_metric == "contrast":
        return compute_image_contrast(img)
    elif snr_metric == "photonFlux":
        return compute_photon_flux(img)
    else:
        raise ValueError("metric must be one of 'basic','acutance',"
                         "'medianBlocks', 'contrast', 'photonFlux'")


