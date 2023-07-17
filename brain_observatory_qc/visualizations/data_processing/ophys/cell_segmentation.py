from brain_observatory_qc.data_access import from_lims
import matplotlib.pyplot as plt
import numpy as np

def get_lims_max_proj_img(oeid, low_percentile=1, high_percentile=99.8):
    """ Get max projection image from LIMS and clip to low and high percentiles

    Parameters
    ----------
    oeid : int
        ophys experiment id
    low_percentile : float, optional
        lower percentile to clip image, default 1
    high_percentile : float, optional
        upper percentile to clip image, default 99.8

    Returns
    -------
    max_img : np.ndarray
        clipped max projection image
    """
    max_img_fn = from_lims.get_max_intensity_projection_filepath(oeid)
    max_img = plt.imread(max_img_fn)
    max_img = np.clip(max_img, np.percentile(max_img, low_percentile), np.percentile(max_img, high_percentile))
    return max_img

def get_lims_average_img(oeid, low_percentile=1, high_percentile=99.8):
    """ Get average image from LIMS and clip to low and high percentiles

    Parameters
    ----------
    oeid : int
        ophys experiment id
    low_percentile : float, optional
        lower percentile to clip image, default 1
    high_percentile : float, optional
        upper percentile to clip image, default 99.8

    Returns
    -------
    avg_img : np.ndarray
        clipped average image
    """
    avg_img_fn = from_lims.get_average_intensity_projection_filepath(oeid)
    avg_img = plt.imread(avg_img_fn)
    avg_img = np.clip(avg_img, np.percentile(avg_img, low_percentile), np.percentile(avg_img, high_percentile))
    return avg_img

def get_img_dims(oeid):
    """ Get image dimensions from LIMS

    Parameters
    ----------
    oeid : int
        ophys experiment id

    Returns
    -------
    img_dim : tuple
        image dimensions
    """

    avg_img_fn = from_lims.get_average_intensity_projection_filepath(oeid)
    avg_img = plt.imread(avg_img_fn)
    img_dim = avg_img.shape
    return img_dim

def add_full_frame_roi(roi_table, oeid, rerun=False):
    """ Add full frame mask to roi_table

    Parameters
    ----------
    roi_table : pd.DataFrame
        roi_table from LIMS
    oeid : int
        ophys experiment id
    rerun : bool, optional
        whether to rerun the function, default False

    Returns
    -------
    roi_table : pd.DataFrame
        roi_table with full frame ROI masks added
    """

    if 'full_frame_mask' in roi_table.columns:
        if not rerun:
            return roi_table
        else:
            roi_table.drop(columns=['full_frame_mask'], inplace=True)
    img_dim = get_img_dims(oeid)
    blank_fov = np.zeros(img_dim, dtype=bool)
    roi_table['full_frame_mask'] = [blank_fov]*len(roi_table)
    
    for i, roi in roi_table.iterrows():
        temp_fov = blank_fov.copy()
        temp_fov[roi.y:(roi.y+roi.height), roi.x:(roi.x+roi.width)] = roi.mask_matrix
        roi_table.at[i, 'full_frame_mask'] = temp_fov
    return roi_table

def draw_roi_outlines(oeid, ax=None, max_or_mean_img='mean', valid='valid', figsize=(10,10),
                      low_percentile=1, high_percentile=99.8, colors=['r','y'], linewidth=1):
    """ Draw ROI outlines on max or mean projection image

    Parameters
    ----------
    oeid : int
        ophys experiment id
    ax : matplotlib.axes.Axes, optional
        axes to plot on, default None
    max_or_mean_img : str, optional
        whether to plot (clipped) max or mean projection image, default 'mean'
    valid : str, optional
        whether to plot valid or invalid ROIs, default 'valid'
    figsize : tuple, optional
        figure size, default (10,10)
        Used only when ax is not provided
    low_percentile : float, optional
        lower percentile to clip image, default 1
    high_percentile : float, optional
        upper percentile to clip image, default 99.8
    colors : list, optional
        list of colors to use for valid and invalid ROIs, default ['r','y']
    linewidth : int, optional
        linewidth for contour, default 1

    Returns
    -------
    ax: matplotlib.axes.Axes
        axes with ROI outlines drawn
    When ax is not provided:
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
        figure and axes with ROI outlines drawn
    """

    roi_table = from_lims.get_cell_rois_table(oeid)
    roi_table = add_full_frame_roi(roi_table, oeid)
    if valid == 'valid':
        roi_table = roi_table[roi_table.valid_roi]
    elif valid == 'invalid':
        roi_table = roi_table[~roi_table.valid_roi]
    if max_or_mean_img == 'mean':
        img = get_lims_average_img(oeid, low_percentile=low_percentile, high_percentile=high_percentile)
    elif max_or_mean_img == 'max':
        img = get_lims_max_proj_img(oeid, low_percentile=low_percentile, high_percentile=high_percentile)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray')
    
    for _, roi in roi_table.iterrows():
        if roi.valid_roi:
            ax.contour(roi.full_frame_mask, colors=colors[0], linewidths=linewidth)
        else:
            ax.contour(roi.full_frame_mask, colors=colors[1], linewidths=linewidth)
    if fig is not None:
        return fig, ax
    else:
        return ax
