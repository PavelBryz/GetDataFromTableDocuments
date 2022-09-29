from typing import Union

import cv2
from numpy import ndarray

from classes.contour import ContourOfCell
from classes.image import Image
from utilities.helpers import find_box


class LineImage(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
