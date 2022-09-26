import cv2


class Displayable:
    def display(self, path_to_save: str = None):
        if path_to_save is None:
            cv2.imshow('Display', self.image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(path_to_save, self.image)