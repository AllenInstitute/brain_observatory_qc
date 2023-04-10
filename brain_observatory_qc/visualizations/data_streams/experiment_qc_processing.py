from bokeh.plotting import curdoc, ColumnDataSource, figure
from bokeh.models import ColumnDataSource, RangeTool, Slider, Select, Plot, Rect
from bokeh.layouts import column, layout
from bokeh.palettes import Greys256
import brain_observatory_qc.data_access.from_lims as fl
import imageio
import glob

import pandas as pd
import h5py as h5
from tifffile import tifffile as tiff
from skimage.segmentation import find_boundaries
import numpy as np
    

def get_events_filepath(oeid):
    return fl.get_observatory_events_filepath(oeid)

def get_mean_motion_corrected_filepath(oeid):
    file_dir = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/fov_tilt/tif"
    try:
        mean_mc_fp = glob.glob(f"{file_dir}/{oeid}_segment_fov.tif")[0]
    except KeyError:
        mean_mc_fp = None
    return mean_mc_fp

def get_raw_filepath(oeid):
    file_dir = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/fov_tilt/raw_tif"
    try:
        raw_mc_fp = glob.glob(f"{file_dir}/{oeid}_segment_fov.tif")[0]
    except KeyError:
        raw_mc_fp = None
    return raw_mc_fp

def get_max_projection_filepath(oeid):
    return fl.get_max_intensity_projection_filepath(oeid)

def get_suite2p_segmentation_df(oeid):
    return fl.get_cell_rois_table(oeid)

def generate_dff_trace_plot(range=None):
    if range:
        range = range
    return figure(
        title="",
        height = 300,
        width = 800,
        tools="xpan",
        x_axis_type="linear",
        x_axis_location="above",
        background_fill_color="#efefef",
        x_range = range
    )

def generate_range_tool(plot):
    range_tool = RangeTool(x_range=plot.x_range)
    range_tool.overlay.fill_color = "purple"
    range_tool.overlay.fill_alpha = 0.2
    return range_tool

def generate_image_select_tool(plot, range_tool, source):
    select = figure(
        title="Drag the middle and edges of the selection box to change the range of plots above",
        height=130, 
        width=800, 
        y_range=plot.y_range,
        y_axis_type=None,
        tools="", 
        toolbar_location=None, 
        background_fill_color="#efefef"
        )
    select.line(
        'timestamps', 
        'new_dff', 
        source=source
        )
    select.toolbar.active_multi = range_tool
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    return select

def generate_mask_plot():
    return Plot(
        width=512,
        height=512,
        toolbar_location=None,
        tooltips=[
        ("cell_specimen_id", "$cell_specimen_id"),
        ("x", "$x") 
        ("y", "$y"), 
        ("value", "@image")
        ]
    )
def generate_image_figure(title = ""):
    p = figure(
        title = title,
        tooltips=[
        ("x", "$x"), 
        ("y", "$y"), 
        ("value", "@image")
        ]
    )

    p.x_range.range_padding = p.y_range.range_padding = 0
    return p

def build_mask_matrix(dataframe):
    cells_matrix = np.zeros((512, 512), dtype=np.uint8)
    for index, row in dataframe.iterrows():
        r = row['y']
        c = row['x']
        cells_matrix[
            r:r+row['height'], 
            c:c+row['width']
            ] += np.asarray(row['mask_matrix'])
    cells_matrix = find_boundaries(cells_matrix, mode='thick')
    segmentation_source = ColumnDataSource({
        'image': [cells_matrix]
        }
    )
    return segmentation_source

def generate_mask_plot(figure, color_palette, source): 
    return figure.image(
        image='image',
        x=0,
        y=0,
        dw=512,
        dh=512,
        palette=color_palette,
        level="image",
        source=source
    )