from typing import Union

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import ContourOfCell
from classes.image import Image
from utilities.helpers import find_box


class TableImage(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
        self.cells = []
        
    def find_counters(self):
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        counters, hi = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for counter in counters:
            _, _, w, h = cv2.boundingRect(counter)
            rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            box = np.sort(box, axis=0)

            sm = cv2.arcLength(counter, True)
            approx = cv2.approxPolyDP(counter, 0.1 * sm, True)

            if box[0][0] == 0 or box[0][1] == 0 or w <= 5 or h <= 5: continue
            # if box[0][0] < 0 or box[0][1] < 0 or \
            #    box[1][0] < 0 or box[1][1] < 0 or \
            #    box[2][0] < 0 or box[2][1] < 0 or \
            #    box[3][0] < 0 or box[3][1] < 0: continue

            if sm <= 225 and len(approx): continue
            self.cells.append(ContourOfCell(box))
