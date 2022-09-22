import base64
from typing import Union, List

from numpy import ndarray
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
from werkzeug.datastructures import FileStorage

POPPLER_PATH = r'C:\Users\Bryzgalov.Pavel\Documents\Общее\Разработка\GetDataFromSTOADoc\utilities\poppler-22.04.0\Library\bin'


def convert_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        bytes_file = base64.b64encode(image_file.read())
        return bytes_file.decode('utf-8')


def convert_from_base64(encoded_file):
    return base64.b64decode(encoded_file.encode('utf-8'))


def convert_bytes_to_image(bytes_file):
    return convert_from_bytes(bytes_file, poppler_path=POPPLER_PATH)


def pdf_to_png(pdf_document: Union[FileStorage, str]) -> List[ndarray]:
    if isinstance(pdf_document, str):
        images = convert_from_path(pdf_document, 300, poppler_path=POPPLER_PATH)
    elif isinstance(pdf_document, FileStorage):
        images = convert_from_bytes(pdf_document.read())
    else:
        raise ValueError("")

    return [np.array(image) for image in images]
