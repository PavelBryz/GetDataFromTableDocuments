import itertools

import numpy as np
from typing import Type, List

from classes.textProcessor import TextProcessor


class Contour:
    top_left = None
    top_right = None
    down_right = None
    down_left = None

    def __init__(self, box):
        self.top_left = (box[0][0], box[0][1])
        self.top_right = (box[1][0], box[1][1])
        self.down_right = (box[2][0], box[2][1])
        self.down_left = (box[3][0], box[3][1])

        self.count_inner = 0

    def get_wight(self):
        return max(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0]) - \
               min(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0])

    def get_height(self):
        return max(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1]) - \
               min(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1])

    @staticmethod
    def get_wight_static(box):
        return max(box[0][0], box[1][0], box[2][0], box[3][0]) - \
               min(box[0][0], box[1][0], box[2][0], box[3][0])

    @staticmethod
    def get_height_static(box):
        return max(box[0][1], box[1][1], box[2][1], box[3][1]) - \
               min(box[0][1], box[1][1], box[2][1], box[3][1])

    def __repr__(self) -> str:
        return f"{self.top_left}"

    def __lt__(self, other) -> bool:
        return self.top_left[0] + self.top_left[1] < other.top_left[0] + other.top_left[1]

    def __le__(self, other) -> bool:
        return self.top_left[0] + self.top_left[1] <= other.top_left[0] + other.top_left[1]

    def __gt__(self, other) -> bool:
        return self.top_left[0] + self.top_left[1] > other.top_left[0] + other.top_left[1]

    def __ge__(self, other) -> bool:
        return self.top_left[0] + self.top_left[1] >= other.top_left[0] + other.top_left[1]

    def __eq__(self, other) -> bool:
        return self.top_left == other.top_left and self.top_right == other.top_right and \
               self.down_left == other.down_left and self.down_right == other.down_right

    def __ne__(self, other):
        return self.top_left[1] != other.top_left[1]

    def is_inside(self, other):
        if (self.top_left[0] > other.top_left[0]) and (self.top_left[1] + 2 > other.top_left[1]) and (
                self.down_right[0] < other.down_right[0]) and (self.down_right[1] - 2 < other.down_right[1]) or (
                self.top_right[0] < other.top_right[0]) and (self.top_right[1] + 2 > other.top_right[1]) and (
                self.down_left[0] > other.down_left[0]) and (self.down_left[1] - 2 < other.down_left[1]):
            self.count_inner += 1

    def coordinates(self):
        coordinate_array = np.array([[int(self.top_left[0]), int(self.top_left[1])],
                                     [int(self.top_right[0]), int(self.top_right[1])],
                                     [int(self.down_right[0]), int(self.down_right[1])],
                                     [int(self.down_left[0]), int(self.down_left[1])]])
        return coordinate_array


class ContourOfCell(Contour):
    column = None
    row = None
    text_processor = None
    text = None

    def __init__(self, box):
        super().__init__(box)
        self.text_processor = TextProcessor()

    def __repr__(self) -> str:
        return f"r={self.row}|c={self.column}|text={self.text}"

    @staticmethod
    def set_rows(array_counters: List[Type['ContourOfCell']]) -> None:
        row = 0

        while [y for y in array_counters if y.row is None].__len__() > 0:
            row += 1
            array = [y for y in array_counters if y.row is None]
            min_y = int(min([y.top_left[1] for y in array]))
            col_y = [y for y in array if (abs(y.top_left[1] - min_y) <= 20)]
            for el in col_y:
                el.row = row
                # el.top_left = (el.top_left[0], min_y)

    @staticmethod
    def set_columns(array_counters: List[Type['ContourOfCell']]) -> None:
        col = 0

        while [y for y in array_counters if y.column is None].__len__() > 0:
            col += 1
            array = [y for y in array_counters if y.column is None]
            min_x = int(min([y.top_left[0] for y in array]))
            col_x = [x for x in array if (abs(x.top_left[0] - min_x) <= 10)]
            for el in col_x:
                el.column = col
                # el.top_left = (min_y, el.top_left[1])


    @staticmethod
    def sort_contours(contours: List[Type['ContourOfCell']]) -> List[Type['ContourOfCell']]:
        return sorted(contours, key=lambda counter: [counter.row, counter.column])

    @staticmethod
    def filter_contours(contours: List[Type['ContourOfTable']]) -> List[Type['ContourOfTable']]:
        for base_contour, compare_with in itertools.combinations(contours, 2):
            base_contour.is_inside(compare_with)
        return list(filter(lambda x: x.count_inner == 0, contours))


class ContourOfTable(Contour):
    number = None

    def __init__(self, box):
        super().__init__(box)

    def __repr__(self) -> str:
        return f"{self.top_left}|n={self.number}"

    @staticmethod
    def set_numbers(array_counters: List[Type['ContourOfTable']]) -> None:
        number = 0

        while [y for y in array_counters if y.number is None].__len__() > 0:
            number += 1
            array = [y for y in array_counters if y.number is None]
            min_y = int(min([y.top_left[1] for y in array]))
            col_y = [y for y in array if (abs(y.top_left[1] - min_y) <= 5)]
            for el in col_y:
                el.number = number
                # el.top_left = (el.top_left[0], min_y)

    @staticmethod
    def sort_contours(contours: List[Type['ContourOfTable']]) -> List[Type['ContourOfTable']]:
        return sorted(contours, key=lambda counter: [counter.number])

    @staticmethod
    def filter_contours(contours: List[Type['ContourOfTable']]) -> List[Type['ContourOfTable']]:
        for base_contour, compare_with in itertools.combinations(contours, 2):
            base_contour.is_inside(compare_with)

        return list(filter(lambda x: x.count_inner == 0, contours))


class ContourOfLine(Contour):
    number = None
    text_processor = None
    text = None

    def __init__(self, box):
        super().__init__(box)
        self.text_processor = TextProcessor()

    def __repr__(self) -> str:
        return f"n={self.number}|t={self.text}"

    @staticmethod
    def set_numbers(array_counters: List[Type['ContourOfLine']]) -> None:
        number = 0

        while [y for y in array_counters if y.number is None].__len__() > 0:
            number += 1
            array = [y for y in array_counters if y.number is None]
            min_y = int(min([y.top_left[1] for y in array]))
            col_y = [y for y in array if (abs(y.top_left[1] - min_y) <= 5)]
            for el in col_y:
                el.number = number
                # el.top_left = (el.top_left[0], min_y)

    @staticmethod
    def sort_contours(contours: List[Type['ContourOfLine']]) -> List[Type['ContourOfLine']]:
        return sorted(contours, key=lambda counter: [counter.number])

