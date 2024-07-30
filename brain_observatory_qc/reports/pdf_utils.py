d"""
Helper class for generating PDF reports (using the fpdf2 library)
"""

import os
import platform
import numpy as np
from fpdf import FPDF
from PIL import Image
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

    def header(self):
        """
        Add a header with the Neural Dynamics logo and a running title
        """
        self.image(
            os.path.join(os.path.dirname(__file__), "images", "aind-logo.png"),
            x=150,
            y=10,
            w=50,
        )
        self.set_font("courier", "", 9)
        self.cell(30, 15, self.title, align="L")
        self.ln(20)

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

    def embed_figure(self, fig, width=190):
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

    def embed_table(self, df, width=190):
        """
        Create a table out of a pandas DataFrame and embed it in the PDF

        Parameters
        ----------
        df : pandas.DataFrame
            The table to embed
        width : int
            The width of the image in the PDF
        """

        DF = df.astype(str)  # convert all elements to string
        DATA = [
            list(DF)
        ] + DF.values.tolist()  # Combine columns and rows in one list

        with self.table(
            borders_layout="SINGLE_TOP_LINE",
            cell_fill_color=240,
            cell_fill_mode="ROWS",
            line_height=self.font_size * 2,
            text_align="CENTER",
            width=width,
        ) as table:
            for data_row in DATA:
                row = table.row()
                for datum in data_row:
                    row.cell(datum)

    def add_plot(self, plot_path:str, title:str=None):
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