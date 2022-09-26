import numpy as np
from typing import Type, List


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

    def __repr__(self) -> str:
        return f"{self.top_left}"

    def __lt__(self, other) -> bool:
        return self.top_left[1] < other.top_left[1]

    def __le__(self, other) -> bool:
        return self.top_left[1] <= other.top_left[1]

    def __gt__(self, other) -> bool:
        return self.top_left[1] > other.top_left[1]

    def __ge__(self, other) -> bool:
        return self.top_left[1] >= other.top_left[1]

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

    def __init__(self, box):
        super().__init__(box)

    def __repr__(self) -> str:
        return f"{self.top_left}|{self.column}"

    @staticmethod
    def set_rows(array_counters: List[Type['ContourOfCell']]) -> None:
        col = 0


        while [y for y in array_counters if y.column is None].__len__() > 0:
            col += 1
            array = [y for y in array_counters if y.column is None]
            min_y = int(min(array).top_left[1])
            col_y = [y for y in array if (abs(y.top_left[1] - min_y) <= 15)]
            for el in col_y:
                el.column = col
                el.top_left = (el.top_left[0], min_y)

    @staticmethod
    def sort_contours(contours: List[Type['ContourOfCell']]) -> List[Type['ContourOfCell']]:
        array_all = []  # массив внутри которого другие массивы

        for counter in contours:
            our_counters_part = sorted(counter, key=lambda counter: [counter.top_left_y_sorted, counter.top_left_x])
            array_all.append(our_counters_part)

        return array_all
