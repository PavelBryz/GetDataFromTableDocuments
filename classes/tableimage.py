import math
from typing import Union

import cv2
import numpy as np
from numpy import ndarray

from classes.contour import ContourOfCell, Contour
from classes.image import Image
from utilities.helpers import find_box


KERNEL_5 = np.array([[0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0],
                     [1, 1, 1, 1, 1],
                     [0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0]],
                     dtype='uint8')

KERNEL_3 = np.array([[0, 1, 1],
                     [1, 1, 1],
                     [0, 1, 0]],
                     dtype='uint8')

class TableImage(Image):
    def __init__(self, file: Union[ndarray, str]):
        super().__init__(file)
        self.cells = []

    def draw_lines(self):
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 4  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 2  # maximum gap in pixels between connectable line segments
        line_image = np.copy(self.image) * 0  # creating a blank to draw lines on

        edges = cv2.Canny(self.image, 100, 200, apertureSize=3)
        edges = cv2.dilate(edges, KERNEL_3, iterations=1)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        h_lines = []
        w_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if y1 == y2 and length >= self.image.shape[1] / 4 and not len([y for y in h_lines if abs(y - y1) < 10]):
                    cv2.line(line_image, (0, y1), (self.image.shape[1], y2), (255, 255, 255), 2)
                    h_lines.append(y1)
                    # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if x1 == x2 and (length >= self.image.shape[0] / 4 or y1 < 10 or y2 < 10) and not len([x for x in w_lines if abs(x - x1) < 10]):
                    cv2.line(line_image, (x1, 0), (x2, self.image.shape[0]), (255, 255, 255), 2)
                    w_lines.append(x1)
                    # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the lines on the  image
        self.image = cv2.addWeighted(self.image, 1, line_image, -1, 0)
        
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

            # Filter values less than zero
            if box[0][0] < 0 or box[0][1] < 0 or \
               box[1][0] < 0 or box[1][1] < 0 or \
               box[2][0] < 0 or box[2][1] < 0 or \
               box[3][0] < 0 or box[3][1] < 0: continue

            box_width = Contour.get_wight_static(box)
            box_height = Contour.get_height_static(box)

            # Filter box if it's size close ot the size of table
            if (box[0][0] + box[0][1]) < 25 and (box_width + box_height > (height + width - 25)): continue

            # Filter box if it's size less than 0,3% of table or perimeter less than 255. 255 - magic number
            # ToDo think what to do with 255
            sm = cv2.arcLength(counter, True)
            if (box_width * box_height) < (height * width / 300) or sm <= 225: continue

            self.cells.append(ContourOfCell(box))

        # If none contours was found it means that image its self is a cell, so we add it
        if len(self.cells) == 0:
            box = [[0, 0],
                   [width, 0],
                   [width, height],
                   [0, height]]
            self.cells.append(ContourOfCell(box))