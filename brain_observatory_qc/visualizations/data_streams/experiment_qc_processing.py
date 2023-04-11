import glob
from skimage.segmentation import find_boundaries
from bokeh.models import ColumnDataSource, RangeTool, Slider, Select
from bokeh.palettes import Greys256
from bokeh.plotting import figure
import numpy as np
import brain_observatory_qc.data_access.from_lims as fl


def get_events_filepath(oeid):
    """Gets the filepath to the h5 file with firing events

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pathlib.Path: path to events h5 file
    """
    return fl.get_observatory_events_filepath(oeid)

def get_mean_motion_corrected_fov_filepath(oeid):
    """Gets the filepath to the tif images containing the averaged, motion corrected acquisition images

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pathlib.Path: path to events h5 file
    """
    file_dir = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/fov_tilt/tif"
    try:
        mean_mc_fp = glob.glob(f"{file_dir}/{oeid}_segment_fov.tif")[0]
    except KeyError:
        mean_mc_fp = None
    return mean_mc_fp

def get_raw_mean_fov_filepath(oeid):
    """Gets the filepath to the tif images containing the averaged, raw acquistiion images

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pathlib.Path: path to events h5 file
    """
    file_dir = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/fov_tilt/raw_tif"
    try:
        raw_mc_fp = glob.glob(f"{file_dir}/{oeid}_segment_fov.tif")[0]
    except KeyError:
        raw_mc_fp = None
    return raw_mc_fp

def get_dff_trace_filepath(oeid):
    """Gets the filepath to the h5 file containing new and old dF/F and np corrected traces

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pathlib.Path: path to dF/F trace h5 file
    """
    file_dir = "/allen/programs/mindscope/workgroups/learning/pipeline_validation/dff"
    try:
        events_fp = glob.glob(f"{file_dir}/{oeid}_new_dff.h5")[0]
    except KeyError:
        events_fp = None
    return events_fp

def get_max_projection_filepath(oeid):
    """Gets the filepath to the maximum projected image for an ophys experiment ID

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pathlib.Path: path to max projected image 
    """
    return fl.get_max_intensity_projection_filepath(oeid)

def get_suite2p_segmentation_df(oeid):
    """Returns the segmentation dataframe as calculated by LIMS

    Args:
        oeid (int): ophys experiment ID

    Returns:
        pandas.dataframe: dataframe of cell ROI segmentation information 
    """
    return fl.get_cell_rois_table(oeid)

def generate_dff_trace_plot(xrange):
    """Returns a linear, bokeh figure to plot dF/F traces

    Args:
        xrange (tuype): _description_. Defaults to None.

    Returns:
        bokeh.plotting.figure: Bokeh linear figure object
    """
    return figure(
        title="",
        height = 300,
        width = 800,
        tools="xpan",
        x_axis_type="linear",
        x_axis_location="above",
        background_fill_color="#efefef",
        x_range = xrange
    )

def generate_range_tool(xrange):
    """Returns the Bokeh range tool

    Args:
        xrange (tuple): range to synchronize with teh x-dimension overlay

    Returns:
        bokeh.model.Range: Bokeh range object
    """
    range_tool = RangeTool(x_range=xrange)
    range_tool.overlay.fill_color = "purple"
    range_tool.overlay.fill_alpha = 0.2
    return range_tool

def generate_image_select_tool(yrange, range_tool, source):
    """Returns a Bokeh figure to dynamically interact with the 

    Args:
        yrange (tuple): Customize the y-range of the plot
        range_tool (bokeh.models.Range): Range tool to add to selection
        source (bokeh.models.ColumnDataSource): Add this sourse to tool selector

    Returns:
        bokeh.models.Select: Bokeh tool to add to layout
    """
    select = figure(
        title="Drag the middle and edges of the selection box to change the range of plots above",
        height=130,
        width=800,
        y_range=yrange,
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

def generate_image_figure(title = ""):
    """Returns generic imaging figure

    Args:
        title (str, optional): Title for plot. Defaults to "".

    Returns:
        bokeh.models.figure: Bokeh figure with hover capability
    """
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

def build_segmentation_mask_matrix(dataframe):
    """Fills in image with all segmentations applied

    Args:
        dataframe (pandas.dataframe): LIMS built pandas.df object

    Returns:
        np.array: (512, 512) numpy array
    """
    cells_matrix = np.zeros((512, 512), dtype=np.uint8)
    for index, row in dataframe.iterrows():
        r = row['y']
        c = row['x']
        cells_matrix[
            r:r+row['height'], 
            c:c+row['width']
            ] += np.asarray(row['mask_matrix'])
    cells_matrix = find_boundaries(cells_matrix, mode='thin')
    segmentation_source = ColumnDataSource({
        'image': [cells_matrix]
        }
    )
    return segmentation_source

def generate_greys_image_figure(fig, source):
    """Adds image array to figure in Greys256 palette

    Args:
        fig (bokeh.models.figure): Bokeh figure with hover capability
        source (bokeh.models.ColumnDataSource): ColumnDataSource with type 'image' key

    Returns:
        bokeh.models.figure (image): figure with image
    """
    return fig.image(
        image='image',
        x=0,
        y=0,
        dw=512,
        dh=512,
        palette=Greys256,
        level="image",
        source=source
    )

def generate_mask_plot(fig, color_palette, source):
    """Adds masked ColumnDataSource to figure and return figure

    Args:
        fig (bokeh.models.figure): Bokeh image figure
        color_palette (list): rgba specified
        source (bokeh.models.ColumnDataSource): ColumnDataSource with type 'image' key

    Returns:
        bokeh.models.figure (image): figure with image
    """
    return fig.image(
        image='image',
        x=0,
        y=0,
        dw=512,
        dh=512,
        palette=color_palette,
        level="image",
        source=source
    )

def generate_slider(end):
    """Returs slider object to add to layout

    Args:
        end (int): length of slider

    Returns:
        bokeh.models.Slider: Bokeh slider object
    """
    return Slider(
        start=0,
        end=end,
        step=1,
        value=0
    )

def generate_select_tool(title, data):
    """Returns dropdown tool to select from a list of strings

    Args:
        title (str): Slide title
        data (list): data to select from

    Returns:
        bokeh.models.Select: Bokeh Select object
    """
    return Select(
        title=title, 
        options=data, 
        value=data[0]
    )
