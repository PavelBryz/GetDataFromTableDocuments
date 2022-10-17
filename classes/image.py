from typing import Union, List, Tuple
from typing_extensions import Final

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import Contour
from utilities.helpers import find_box, angle_transform


class Image:
    def __init__(self, file: Union[ndarray, str]):
        if isinstance(file, str):
            self.image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        elif isinstance(file, ndarray):
            if len(file.shape) == 2:
                self.image = file
            else:
                self.image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Incorrect input type")

    def crop_image(self, box: Union[Contour]):
        crop = self.image[box.top_left[1]: box.top_left[1] + box.get_height(),
                          box.top_left[0]: box.top_left[0] + box.get_wight()]
        return crop

    def display(self, path_to_save: str = None):
        if path_to_save is None:
            cv2.imshow('Display', self.image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(path_to_save, self.image)

    def draw_contour(self, contour: Contour, color=(0, 255, 0), thickness=1):
        if len(self.image.shape) == 2 and len(color) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.image = cv2.rectangle(self.image, contour.top_left, contour.down_right, color, thickness=thickness)

    def get_width_height(self):
        return self.image.shape[:2]

    def resize(self, scale: float = 0.5):
        try:
            height, width = self.get_width_height()
            self.image = cv2.resize(self.image, (int(width * scale), int(height * scale)), cv2.INTER_NEAREST)
        except Exception:
            pass

    @staticmethod
    def static_display(img: ndarray):
        cv2.imshow('Display', img)
        cv2.waitKey(0)
