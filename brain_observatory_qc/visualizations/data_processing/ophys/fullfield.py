from ophys_etl.modules.mesoscope_splitting.tiff_metadata import ScanImageMetadata
import h5py
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np

class ScanImageMetadataFromH5(ScanImageMetadata):
    """
    A class to handle reading and parsing of the metadata that comes with 
    the tiff files produced by scanIMage
    
    Parameters
    tiff_path: pathlib.Path
        Path to the tiff file to be read
    
    
    """
    def __init__(self, tiff_path: pathlib.Path):
        self._file_path = tiff_path
        if not tiff_path.is_file():
            raise FileNotFoundError(f'Could not find file {tiff_path}')
        with h5py.File(self._file_path, 'r') as f:
            self._metadata = json.loads(f['full_field_metadata'][()])
            self._surface_roi_metadata = json.loads(f['surface_roi_metadata'][()])

            #check that the fullfield images are not empty 
            assert f['stitched_full_field'].shape > (0,0)
            assert f['stitched_full_field'].shape == f['stitched_full_field_with_rois'].shape


    def get_fullfield_figure(self, dpi = 200):
        """
        Returns the fullfield image as a matplotlib figure

        """
        with h5py.File(self._file_path, 'r') as f:
            fullfield = f['stitched_full_field']

            fig = plt.figure(dpi)
            fig.add_subplot(111)
            #add fullfield image to figure
            plt.imshow(fullfield)
            fig.tight_layout()
            return fig
        
    def get_fullfield_roi_figure(self, dpi = 200):
        """
        Returns the fullfield roi image as a matplotlib figure
        """
        with h5py.File(self._file_path, 'r') as f:
            fullfield_roi = f['stitched_full_field_with_rois']

            fig = plt.figure(dpi)
            fig.add_subplot(111)
            #add fullfield image to figure
            plt.imshow(fullfield_roi)
            fig.tight_layout()
            return fig