import cv2


class Resizeable:
    def get_width_height(self):
        return self.image.shape[:2]

    def resize(self, scale: float = 0.5):
        height, width = self.get_width_height()
        self.image = cv2.resize(self.image, (int(width * scale), int(height * scale)), cv2.INTER_NEAREST)