from typing import Union

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import Contour
from extensions.displayable import Displayable
from extensions.resizeable import Resizeable
from extensions.drawable import Drawable
from utilities.helpers import find_box, angle_transform

OPERATION_TYPE_TABLE = 0
OPERATION_TYPE_TEXT = 1


class Image(Displayable, Resizeable, Drawable):
    def __init__(self, file: Union[ndarray, str]):
        if isinstance(file, str):
            self.image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        elif isinstance(file, ndarray):
            self.image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Incorrect input type")

        self.counters = []

        # self.resize()

    def rotate(self):
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        angles = np.array([])
        for counter in counters:
            rect = cv2.minAreaRect(counter)
            sm = cv2.arcLength(counter, True)
            if sm <= 1000: continue
            angles = np.append(angles, rect[2])

        vf = np.vectorize(angle_transform)
        angles = vf(angles[(angles > 80) | (angles < 10)])

        height, width = self.get_width_height()
        center = (int(width / 2), int(height / 2))
        # производим поворот с уменьшением изображения для того, что оно не срезалось при повороте
        rotation_matrix = cv2.getRotationMatrix2D(center, np.average(angles), 1)

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

    def erase_contour(self, counter: Contour):
        self.draw_contour(counter, ([255]), cv2.FILLED)

    def erase_tables(self):
        for contour in self.counters:
            self.erase_contour(contour)

    def contrast_image(self, clipLimit: float = 7.0):
        """
        :return: преобразованный массив цветов каждого пикселя с улучшенной контрастностью
        """
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image)

    def find_counters(self, type_of_operation):
        self.counters = []
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if type_of_operation == OPERATION_TYPE_TEXT:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for counter in counters:
            box, w, h = find_box(counter)
            if box[0][0] == 0 or box[0][1] == 0 or w == 0 or h == 0: continue
            if type_of_operation == OPERATION_TYPE_TEXT:
                if abs(box[2][1] - box[0][1]) < 5 or abs(box[2][0] - box[0][0]) < 50: continue
            else:
                # нахождение точек таким образом мы можем найти прямоугольники и определить их длину
                sm = cv2.arcLength(counter, True)
                apd = cv2.approxPolyDP(counter, 0.01 * sm, True)
                if 6 <= len(apd) <= 4 or w <= 50 or h <= 30: continue
            self.counters.append(Contour(box))
