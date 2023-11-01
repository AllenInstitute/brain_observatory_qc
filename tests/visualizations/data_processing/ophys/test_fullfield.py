import pathlib
import brain_observatory_qc.data_access.from_lims as from_lims
import brain_observatory_qc.visualizations.data_processing.ophys.fullfield as fullfield
import pytest
import matplotlib

@pytest.mark.onprem
def test_display_fullfield_roi_image():
    #test path that works
    fullfield_h5_path = pathlib.Path('/allen/aibs/informatics/danielsf/full_field_stitching/1212880506_stitched_full_field_img.h5')
    metadata = fullfield.ScanImageMetadataFromH5(fullfield_h5_path)
    fig = metadata.get_fullfield_roi_figure()
    assert isinstance(fig, matplotlib.figure.Figure)
    
    assert tuple(fig.get_size_inches()+fig.dpi) > (0,0)


