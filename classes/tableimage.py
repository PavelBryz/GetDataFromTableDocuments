from typing import Union

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import ContourOfCell, Contour
from classes.image import Image
from utilities.helpers import find_box


class TableImage(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
        self.cells = []
        
    def find_counters(self):
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        counters, hi = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        height, width = self.get_width_height()

        for counter in counters:
            _, _, w, h = cv2.boundingRect(counter)
            rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            box = np.sort(box, axis=0)

            sm = cv2.arcLength(counter, True)

            # if box[0][0] == 0 or box[0][1] == 0 or w <= 5 or h <= 5: continue
            # # if abs(box[0][1] - box[1][1]) < 10 : continue
            # if box[0][0] < 0 or box[0][1] < 0 or \
            #    box[1][0] < 0 or box[1][1] < 0 or \
            #    box[2][0] < 0 or box[2][1] < 0 or \
            #    box[3][0] < 0 or box[3][1] < 0: continue

            box_width = Contour.get_wight_static(box)
            box_height = Contour.get_height_static(box)

            if (box[0][0] + box[0][1]) < 25 and (box_width + box_height > (height + width - 25)): continue

            # if sm > (height * width * 2 * 0.9): continue

            if (box_width * box_height) < (height * width / 300) or sm <= 225: continue
            # if area < (height * width / 100) or sm <= 225: continue

            # if sm <= 225 or len(approx) != 4: continue
            # if (1 < abs(box[2][1] - box[0][1]) < 24) or (1 < abs(box[2][0] - box[0][0]) < 80): continue
            self.cells.append(ContourOfCell(box))

        if len(self.cells) == 0:
            box = [[0, 0],
                   [width, 0],
                   [width, height],
                   [0, height]]
            self.cells.append(ContourOfCell(box))