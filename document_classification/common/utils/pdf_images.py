"""Memory-efficient PDF-to-Image conversion."""

import shutil
import subprocess as sp
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np


class PdfImages:
    """
    A container that holds images of PDF pages.

    It is a generic python sequence that supports `len()` and indexing.
    """

    def __init__(self, path: Path, dpi: int = 201, password: str = "") -> None:
        """
        Convert all pages to images and stores them in a temporary directory.

        Args:
            path: str
                Path to PDF file
            dpi: int
                Dots per inch (quality of image)
            password: str
                Password to decrypt encrypted PDFs

        """
        self.tempdir = tempfile.TemporaryDirectory()
        tempdir_name = self.tempdir.name

        # Find the full path of pdftoppm
        pdftoppm_path = shutil.which("pdftoppm")
        if pdftoppm_path is None:
            msg = "pdftoppm executable not found in PATH"
            raise FileNotFoundError(msg)

        # Validate inputs

        pdf_path = path
        if not pdf_path.is_file():
            msg = "Invalid PDF file path"
            raise ValueError(msg)

        if not isinstance(dpi, int) or dpi <= 0:
            msg = "DPI must be a positive integer"
            raise ValueError(msg)

        # Construct command list
        cmd: list[str] = [
            pdftoppm_path,
            str(pdf_path),
            f"{tempdir_name}/out",
            "-jpeg",
            "-r",
            str(dpi),
        ]

        # Add password arguments only if a password is provided
        if password:
            cmd.extend(["-upw", password, "-opw", password])

        sp.run(  # noqa: S603
            cmd,
            check=True,
            shell=False,
            capture_output=True,
            text=True,
        )

        self.images = sorted(Path(tempdir_name).glob("*.jpg"))

    def __del__(self) -> None:
        """Remove the temporary directory where the page-wise images are stored."""
        self.tempdir.cleanup()

    def __len__(self) -> int:
        """Return the number of pages."""
        return len(self.images)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the image of the particular page/index."""
        return cv.imread(str(self.images[index]))
