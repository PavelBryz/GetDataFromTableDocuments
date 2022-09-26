import cv2
from numpy import ndarray

from classes.contour import ContourOfCell
from utilities.helpers import find_box

from extensions.displayable import Displayable
from extensions.resizeable import Resizeable
from extensions.drawable import Drawable


class Table(Displayable, Resizeable, Drawable):
    def __init__(self, image: ndarray):
        self.image = image
        self.cells = []
        
    def find_counters(self):
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        counters, hi = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for counter in counters:
            box, w, h = find_box(counter)
            if box[0][0] == 0 or box[0][1] == 0 or w == 0 or h == 0: continue
            if abs(box[2][1] - box[0][1]) < 12 or abs(box[2][0] - box[0][0]) < 40: continue
            self.cells.append(ContourOfCell(box))
