"""
Helper class for generating PDF reports (using the fpdf2 library)
"""

import os
import platform
import pandas as pd
import numpy as np
from fpdf import FPDF
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class PdfReport(FPDF):
    """
    Sub-class of FPDF with additional methods for embedding images and tables
    """

    def __init__(self, title="Running title"):
        """
        Initialize the PDF report with a title
        """
        super().__init__()
        self.title = title
        self.set_matplotlib_defaults()

    def footer(self):
        """
        Add a footer with the page number
        """
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def set_matplotlib_defaults(self):
        """
        Set the default matplotlib parameters for the report
        """
        plt.rcParams["font.family"] = "sans-serif"

        if platform.system() == "Linux":
            plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
        else:
            plt.rcParams["font.sans-serif"] = ["Arial"]  # pragma: no cover


    def embed_figure(self, fig:matplotlib.figure.Figure, width=190):
        """
        Convert a matplotlib figure to an image and embed it in the PDF

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to embed
        width : int
            The width of the image in the PDF
        """

        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        self.image(img, w=width)

    def embed_table(self, dataframe:pd.DataFrame, table_width=190, column_widths=None):
        """
        Embed a pandas DataFrame as a table in the PDF.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame to embed.
        table_width : int
            The width of the table in the PDF.
        column_widths : list of int, optional
            List of column widths. If None, widths are evenly distributed.
        """
        # Set default column widths if not provided
        if column_widths is None:
            column_widths = [table_width / len(dataframe.columns)] * len(dataframe.columns)
        
        # Set the font for the table
        self.set_font("courier", size=8)

        # Column headers
        for column_index, column_name in enumerate(dataframe.columns):
            self.cell(column_widths[column_index], 10, column_name, border=1, align='C')
        self.ln()

        # Data rows
        for row in dataframe.itertuples(index=False):
            for column_index, cell_value in enumerate(row):
                self.cell(column_widths[column_index], 10, str(cell_value), border=1, align='C')
            self.ln()

    def add_plot_from_filepath(self, plot_path:str, title:str=None):
        """
        Add an image plot from a file path to the PDF

        Parameters
        ----------
        plot_path : str
            Path to the image file
        title : str, optional
            Title for the plot, by default None
        """
        self.add_page()
        if title:
            self.set_font("helvetica", "B", 12)
            self.cell(0, 10, title, align="C")
            self.ln(10)
        self.image(plot_path, x=10, y=30, w=190)