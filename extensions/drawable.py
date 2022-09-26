import cv2

from classes.contour import Contour


class Drawable:
    def draw_contour(self, contour: Contour, color=(0, 255, 0), thickness=1):
        if len(self.image.shape) == 2 and len(color) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.image = cv2.rectangle(self.image, contour.top_left, contour.down_right, color, thickness=thickness)