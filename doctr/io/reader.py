# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

from doctr.utils.common_types import AbstractFile

from .image import read_img_as_numpy

# from .pdf import PDF, read_pdf

__all__ = ["DocumentFile"]


class DocumentFile:
    """Read a document from multiple extensions"""

    # @classmethod
    # def from_pdf(cls, file: AbstractFile, **kwargs) -> PDF:
    #     """Read a PDF file

    #     Example::
    #         >>> from doctr.documents import DocumentFile
    #         >>> doc = DocumentFile.from_pdf("path/to/your/doc.pdf")

    #     Args:
    #         file: the path to the PDF file or a binary stream
    #     Returns:
    #         a PDF document
    #     """

    #     doc = read_pdf(file, **kwargs)

    #     return PDF(doc)

    # @classmethod
    # def from_url(cls, url: str, **kwargs) -> PDF:
    #     """Interpret a web page as a PDF document

    #     Example::
    #         >>> from doctr.documents import DocumentFile
    #         >>> doc = DocumentFile.from_url("https://www.yoursite.com")

    #     Args:
    #         url: the URL of the target web page
    #     Returns:
    #         a PDF document
    #     """
    #     pdf_stream = read_html(url)
    #     return cls.from_pdf(pdf_stream, **kwargs)

    @classmethod
    def from_images(cls, files: Union[Sequence[AbstractFile], AbstractFile], **kwargs) -> List[np.ndarray]:
        """Read an image file (or a collection of image files) and convert it into an image in numpy format

        >>> from doctr.documents import DocumentFile
        >>> pages = DocumentFile.from_images(["path/to/your/page1.png", "path/to/your/page2.png"])

        Args:
            files: the path to the image file or a binary stream, or a collection of those

        Returns:
            the list of pages decoded as numpy ndarray of shape H x W x 3
        """
        if isinstance(files, (str, Path, bytes)):
            files = [files]

        return [read_img_as_numpy(file, **kwargs) for file in files]
