from typing import Union

import cv2
import numpy as np
from numpy import ndarray
from werkzeug.datastructures import FileStorage
from utilities.helpers import find_box

from classes.contour import Contour, ContourOfTable

OPERATION_TYPE_TABLE = 0
OPERATION_TYPE_TEXT = 1


class Image:
    def __init__(self, file: Union[ndarray, str]):
        if isinstance(file, str):
            self.image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        elif isinstance(file, ndarray):
            self.image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Incorrect input type")

        self.counters = []

        # self.resize()

    def get_width_height(self):
        return self.image.shape[:2]

    def resize(self, scale: float = 0.5):
        height, width = self.get_width_height()
        self.image = cv2.resize(self.image, (int(width * scale), int(height * scale)), cv2.INTER_NEAREST)

    def display(self, path_to_save: str = None):
        if path_to_save is None:
            cv2.imshow('Display', self.image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(path_to_save, self.image)

    def rotate(self, angle: float):
        height, width = self.get_width_height()
        center = (int(width / 2), int(height / 2))
        # производим поворот с уменьшением изображения для того, что оно не срезалось при повороте
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # вычисляем абсолютное значение cos и sin
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[1, 0])

        # находим новые границы ширины и высоты
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # используем старые центры изображения и добавляем новые координаты
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        self.image = cv2.warpAffine(self.image, rotation_matrix, (bound_w, bound_h))

    def crop_image(self, counter: Contour):
        crop = self.image[counter.top_left[1]: counter.top_left[1] + counter.get_height(),
               counter.top_left[0]: counter.top_left[0] + counter.get_wight()]
        return crop

    def contrast_image(self, clipLimit: float = 7.0):
        """
        :return: преобразованный массив цветов каждого пикселя с улучшенной контрастностью
        """
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image)


    def find_counters(self, type_of_operation):
        thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        if type_of_operation == OPERATION_TYPE_TEXT:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for counter in counters:
            box, w, h = find_box(counter)
            if box[0][0] == 0 or box[0][1] == 0 or w == 0 or h == 0: continue
            if type_of_operation == OPERATION_TYPE_TEXT:
                if abs(box[2][1] - box[0][1]) < 5 or abs(box[2][0] - box[0][0]) < 50: continue
                self.counters.append(Contour(box))
            else:
                # нахождение точек таким образом мы можем найти прямоугольники и определить их длину
                sm = cv2.arcLength(counter, True)
                apd = cv2.approxPolyDP(counter, 0.01 * sm, True)

                if len(apd) != 4 or w <= 50 or h <= 30: continue
                self.counters.append(ContourOfTable(box))
