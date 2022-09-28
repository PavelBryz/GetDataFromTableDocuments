from typing import Union
from numpy import ndarray

from classes.contour import ContourOfCell
from classes.image import Image


class Cell(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
