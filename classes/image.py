import cv2


class Image():
    def __init__(self, file_path: str):
        self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        # self.resize()

    def resize(self, scale: float = 0.5):
        (h, w) = self.image.shape[:2]
        self.image = cv2.resize(self.image, (int(w * scale), int(h * scale)), cv2.INTER_NEAREST)

    def display(self, path_to_save: str = None):
        if path_to_save is None:
            cv2.imshow('Display', self.image)
            cv2.waitKey(0)
        else:
            cv2.imwrite(path_to_save, self.image)

    def rotate(self):
        (h, w) = self.image.shape[:2]
        center = (int(w / 2), int(h / 2))
        # производим поворот с уменьшением изображения для того, что оно не срезалось при повороте
        rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1)

        # вычисляем абсолютное значение cos и sin
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[1, 0])

        # находим новые границы ширины и высоты
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        # используем старые центры изображения и добавляем новые координаты
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
