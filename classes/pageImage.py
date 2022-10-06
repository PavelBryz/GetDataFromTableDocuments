from typing import Union, List
from typing_extensions import Final

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import Contour, ContourOfTable, ContourOfLine
from classes.image import Image

from utilities.helpers import find_box, angle_transform

OPERATION_TYPE_TABLE: Final[int] = 0
OPERATION_TYPE_TEXT: Final[int] = 1


class PageImage(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
        self.counters: List[Union[ContourOfTable, ContourOfLine]] = []

    def rotate(self):
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if len(self.image.shape) == 2:
        #     self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        #
        # ct = []
        angles = np.array([])
        for counter in counters:
            rect = cv2.minAreaRect(counter)
            sm = cv2.arcLength(counter, True)
            approx = cv2.approxPolyDP(counter, 0.01 * sm, True)
            if sm <= 1000 or len(approx) != 4: continue
            angles = np.append(angles, rect[2])
        #     ct.append(counter)
        #
        # self.image = cv2.drawContours(self.image, ct, -1, (0, 0, 255), 1, cv2.LINE_8)
        # self.display()

        vf = np.vectorize(angle_transform)
        if angles.size == 0: return
        angles = vf(angles[((angles > 80) & (angles != 90)) | ((angles < 2) & (angles > -2) & (angles != 0)) | ((angles < -80) & (angles != -90))])

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
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
            thresh = cv2.dilate(thresh, kernel, iterations=2)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = self.get_width_height()

        for counter in counters:
            _, _, w, h = cv2.boundingRect(counter)
            rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            box = np.sort(box, axis=0)

            if box[0][0] == 0 or box[0][1] == 0 or w == 0 or h == 0: continue
            if box[0][0] < 0 or box[0][1] < 0 or \
               box[1][0] < 0 or box[1][1] < 0 or \
               box[2][0] < 0 or box[2][1] < 0 or \
               box[3][0] < 0 or box[3][1] < 0: continue

            if type_of_operation == OPERATION_TYPE_TEXT:
                if box[0][1] < 10 or box[2][1] > (height - 10) or box[3][0] < 10 or box[1][0] > (width - 10): continue
                if abs(box[2][1] - box[0][1]) < 5 or abs(box[2][0] - box[0][0]) < 50: continue
                if len(counter) < 15 : continue

                self.counters.append(ContourOfLine(box))
            else:
                # нахождение точек таким образом мы можем найти прямоугольники и определить их длину
                sm = cv2.arcLength(counter, True)
                approx = cv2.approxPolyDP(counter, 0.02 * sm, True)

                if w < (width * 0.05) or len(approx) != 4: continue
                self.counters.append(ContourOfTable(box))
