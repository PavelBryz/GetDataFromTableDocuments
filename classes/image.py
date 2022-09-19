import cv2
import numpy as np

from classes.сounter import Counter


class Image:
    def __init__(self, file_path: str):
        self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        self.contrasted_image = None
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

        rotated = cv2.warpAffine(self.image, rotation_matrix, (bound_w, bound_h))

    def crop_image(self, counter: Counter):
        crop = self.image[counter.top_left[1]: counter.top_left[1] + counter.get_height(),
                          counter.top_left[0]: counter.top_left[0] + counter.get_wight()]
        return crop

    def contrast_image(self):
        """
        :param image: получает массив цветов каждого пикселя
        :return: преобразованный массив цветов каждого пикселя с улучшенной контрастностью
        """
        # разделяем массив для получения контрастности пикселей
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # увеличиваем контрастность кадого пикселя
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # объединяем назад в один массив
        limg = cv2.merge((cl, a, b))
        self.contrasted_image = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    def find_counters(self, image, standart_image):
        all_counters = []
        (height, weight) = image.shape[:2]
        image_contrast = cv2.resize(image, (image.shape[1], image.shape[0]))
        # преобразовывем изображение в оттенки серого
        gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
        # производим бинаризацию
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        counters, hi = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        image_counter = cv2.drawContours(image_contrast, counters, -1, (0, 0, 255), 1, cv2.LINE_8, hierarchy=hi)
        for counter in counters:
            x, y, w, h = cv2.boundingRect(counter)
            rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            my_data = np.sort(box, axis=0)
            if abs(my_data[2][1] - my_data[0][1]) >= 20 and abs(
                    my_data[2][0] - my_data[0][0]) >= 40 and w < weight - 50 and h < height - 50:
                all_counters.append(Counter(i))

        our_counters = []
        for i in all_counters:
            for j in all_counters:
                i.is_inside(j)
            if i.count_inner == 0:
                our_counters.append(i)

        return our_counters
