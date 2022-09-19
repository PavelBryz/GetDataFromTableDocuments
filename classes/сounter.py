import numpy as np


class Counter():
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

    def __repr__(self):
        return f"{self.top_left_y}|{self.column}"

    def __lt__(self, other):
        return self.top_left_y < other.top_left_y

    def __le__(self, other):
        return self.top_left_y <= other.top_left_y

    def __gt__(self, other):
        return self.top_left_y > other.top_left_y

    def __ge__(self, other):
        return self.top_left_y >= other.top_left_y

    def __eq__(self, other):
        return self.top_left_y == other.top_left_y

    def __ne__(self, other):
        return self.top_left_y != other.top_left_y

    def is_inside(self, other):
        if (self.top_left_x > other.top_left_x) and (self.top_left_y + 2 > other.top_left_y) and (
                self.down_right_x < other.down_right_x) and (self.down_right_y - 2 < other.down_right_y) or (
                self.top_right_x < other.top_right_x) and (self.top_right_y + 2 > other.top_right_y) and (
                self.down_left_x > other.down_left_x) and (self.down_left_y - 2 < other.down_left_y):
            self.count_inner += 1

    def crop_image(self):
        crop = self.standart_image[self.top_left_y: self.top_left_y + self.h,
               self.top_left_x: self.top_left_x + self.w]
        return crop

    def coordinares(self):
        coordinate_array = np.array([[int(self.top_left_x), int(self.top_left_y)],
                                     [int(self.top_right_x), int(self.top_right_y)],
                                     [int(self.down_right_x), int(self.down_right_y)],
                                     [int(self.down_left_x), int(self.down_left_y)]])
        return coordinate_array

    def get_wight(self):
        return max(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0]) - \
               min(self.top_left[0], self.top_right[0], self.down_left[0], self.down_right[0])

    def get_height(self):
        return max(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1]) - \
               min(self.top_left[1], self.top_right[1], self.down_left[1], self.down_right[1])
