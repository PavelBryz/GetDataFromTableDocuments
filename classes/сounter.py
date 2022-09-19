import numpy as np


class Counter:
    def __init__(self, box):
        self.top_left = (box[0][0], box[0][1])
        self.top_right = (box[1][0], box[1][1])
        self.down_right = (box[2][0], box[2][1])
        self.down_left = (box[3][0], box[3][1])

        self.top_left_y_sorted = box[0][1]

        # self.standart_image = standart_image
        # self.image_contrast = image_contrast

        self.column = None

        self.count_inner = 0

    def get_wight(self):
        return max(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0]) - \
               min(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0])

    def get_height(self):
        return max(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1]) - \
               min(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1])

    def __repr__(self) -> str:
        return f"{self.top_left[1]}|{self.column}"

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


    def coordinares(self):
        coordinate_array = np.array([[int(self.top_left[0]), int(self.top_left[1])],
                                     [int(self.top_right[0]), int(self.top_right[1])],
                                     [int(self.down_right[0]), int(self.down_right[1])],
                                     [int(self.down_left[0]), int(self.down_left[1])]])
        return coordinate_array
