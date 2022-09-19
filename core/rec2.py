import cv2
import numpy as np
# import pytesseract
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Niyazov.I\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


class Counter():
    def __init__(self, box, w, h, standart_image, image_contrast):
        self.top_left_x = box[0][0]
        self.top_left_y = box[0][1]

        self.top_left_y_sorted = box[0][1]

        self.down_left_x = box[1][0]
        self.down_left_y = box[2][1]

        self.top_right_x = box[2][0]
        self.top_right_y = box[1][1]

        self.down_right_x = box[3][0]
        self.down_right_y = box[3][1]

        self.standart_image = standart_image
        self.image_contrast = image_contrast

        self.w = w
        self.h = h

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


# Функция открытия изображения
def open_image(standart_image, resize_weight=0.5, resize_height=0.5):
    """
    Данная функция отвечает за открытие необходимого изображения и при необходимости изменить размер

    :param image: имя фотографии
    :param resize_weight: на сколько увеличить ширину
    :param resize_height: на сколько увеличить высоту
    :return: массив с цветом каждого пикселя
    """
    image = cv2.imread(standart_image, 1)
    # получаем высота и ширина изображения
    (h, w) = image.shape[:2]
    # изменение размеров изображения
    image = cv2.resize(image, (int(w * resize_weight), int(h * resize_height)), cv2.INTER_NEAREST)
    cv2.imshow('asd', image)
    cv2.waitKey(0)
    return image


# Функция поворота изображения
def rotation_images(image, rot):
    """
    :param image: получает изображение
    :return: изображение повернутое на определенный градус
    """

    (h, w) = image.shape[:2]
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
    cv2.imshow('rotated', rotated)
    cv2.waitKey()
    return rotated


# Функция увеличения контрастности
def contrast_image(image):
    """
    :param image: получает массив цветов каждого пикселя
    :return: преобразованный массив цветов каждого пикселя с улучшенной контрастностью
    """
    standart_image = image
    # разделяем массив для получения контрастности пикселей
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # увеличиваем контрастность кадого пикселя
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # объединяем назад в один массив
    limg = cv2.merge((cl, a, b))
    image_contrast = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    return image_contrast, standart_image


# Функция нахождения конутров
def find_counters(image, standart_image):
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
        for i in [my_data]:
            if abs(my_data[2][1] - my_data[0][1]) >= 20 and abs(
                    my_data[2][0] - my_data[0][0]) >= 40 and w < weight - 50 and h < height - 50:
                all_counters.append(Counter(i, w, h, standart_image, image))

    our_counters = []
    for i in all_counters:
        for j in all_counters:
            i.is_inside(j)
        if i.count_inner == 0:
            our_counters.append(i)

    return our_counters


# Функция распределения контуров по линии
def share_counters(array_counters):
    col = 0
    array_all = []

    while [y for y in array_counters if y.column is None].__len__() > 0:
        col += 1
        arr = [y for y in array_counters if y.column is None]
        min_y = min(arr).top_left_y
        col_y = [y for y in arr if (abs(y.top_left_y - min_y) <= 10)]
        for el in col_y:
            el.column = col
            el.top_left_y_sorted = min_y
        array_all.append(col_y)

    return array_all


# Функция сортировки контуров
def sorted_counters(our_counters, image):
    array_all = []  # массив внутри которого другие массивы

    for i in our_counters:
        our_counters_part = sorted(i, key=lambda counter: [counter.top_left_y_sorted, counter.top_left_x])
        array_all.append(our_counters_part)

    for i in array_all:
        for j in i:
            cv2.drawContours(image, [j.coordinares()], 0, (255, 0, 0), 2)
    cv2.imshow('Draw Contours', image)
    cv2.waitKey(0)
    return array_all


# Функция изменения размеров всех контуров
def resize_array_images(array_all, res):
    images = crop_images(array_all)
    array_all_resize = []
    for img_line in images:
        array_all_resize_part = []
        for img in img_line:
            temp_img = resize_image(img, res)
            array_all_resize_part.append(temp_img)
        array_all_resize.append(array_all_resize_part)
    return array_all_resize


# Функция изменения размеров ячейки
def resize_image(image, value):
    if value <= 0: raise ValueError
    h, w = (image.shape[1], image.shape[0])
    image = cv2.resize(image, (int(h * value), int(w * value)))
    return image


# Функция разделения контуров
def crop_images(counters_array):
    croped_images = []

    for counter_line in counters_array:
        croped_images_line = []
        for counter in counter_line:
            image = counter.crop_image()
            croped_images_line.append(image)
        croped_images.append(croped_images_line)
    return croped_images


# Функция настройки ячейки
def processing_image(array_images, method, threshhold):
    array_all_thresh, array_all_dst = [], []
    for images_line in array_images:
        array_part_thresh, array_part_dst = [], []
        for image in images_line:
            # image = cv2.bilateralFilter(image, 5, 10, 10)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            image_ke = cv2.filter2D(gray, -1, kernel)  # увеличиваем резкость
            thresh = cv2.threshold(image_ke, threshhold, 255, method)[1]  # производим обработку threshold
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dst = cv2.erode(thresh, kernel_erode, iterations=1)  # производим коррозию

            array_part_thresh.append(text_on_image_recognition(thresh))
            array_part_dst.append(text_on_image_recognition(dst))

        array_all_thresh.append(array_part_thresh)
        array_all_dst.append(array_part_dst)

    return array_all_thresh, array_all_dst


# Функция распознавания текста
def text_on_image_recognition(images):
    config = r'--psm 6'
    text_on_image = 0

    text_on_image = text_on_image_replace(text_on_image).replace("\n", " ").strip()

    if len(text_on_image) == 0:
        return 'не распознал'
    elif text_on_image.lower() == 'nn' or text_on_image.lower() == 'пп':
        return 'ПП'
    elif text_on_image.lower() == 'wt':
        return 'шт'
    else:
        return text_on_image


# Функция чистки распознаного текста
def text_on_image_replace(text_on_image):
    text_on_image = text_on_image.replace("\n", " ").strip()
    a = ['___', '>', '<', '^', '=', '—', '©', '«', '`', "'", ';', '»', ']', '[', '’', "|", '_', '}', '{', '°', 'o']
    for j in a:
        if len(text_on_image) == 0:
            pass
        elif j == '—':
            text_on_image = text_on_image.replace(j, '-')
            if text_on_image[0] == '-':
                text_on_image = text_on_image.replace('-', '')
            elif text_on_image[-1] == '-':
                text_on_image = text_on_image.replace('-', '')
        elif text_on_image[-3:-2] == '.':
            text_on_image = text_on_image.replace('.', ',')
        elif text_on_image[-1] == '/':
            text_on_image = text_on_image.replace('/', '')
        elif text_on_image[0] == '-' or text_on_image[-1] == '-':
            text_on_image = text_on_image.replace('-', '')
        elif text_on_image[-1] == '.':
            text_on_image = text_on_image.replace('.', '')
        elif text_on_image[0] == ',':
            text_on_image = text_on_image[1:]
        elif text_on_image[-1] == ',':
            text_on_image = text_on_image[:-1]
        else:
            text_on_image = text_on_image.replace(j, '')
        text_on_image = text_on_image.replace("\n", " ").strip()
    return text_on_image


# Функция создания единного текста
def my_func(array_image_single):
    qwe = []
    if np.all(array_image_single == 'не распознал'):
        return 'не распознал'
    array_image_single = np.delete(np.array(array_image_single),
                                   np.where(np.array(array_image_single) == 'не распознал'))
    for first in range(len(array_image_single)):
        for second in range(first + 1, len(array_image_single)):
            asd = []
            c = SequenceMatcher(None, a=array_image_single[first], b=array_image_single[second])
            for tag, i1, i2, j1, j2 in c.get_opcodes():
                if tag == 'equal':
                    if array_image_single[first][i1:i2] != ' ':
                        asd.append(array_image_single[first][i1:i2])
            if len(asd) != 0:
                qwe.append(''.join(asd))
    test = max(qwe, key=len, default='совпадений нет')
    print(test)
    return np.array(test, dtype="<U1000")


# Функция создания одного массива
def unification_arrays(TRUNC_tresh, TRUNC_dst, TOZERO_tresh, TOZERO_dst):
    array_1 = TRUNC_tresh
    array_2 = TRUNC_dst
    array_3 = TOZERO_tresh
    array_4 = TOZERO_dst
    array_image_single = np.stack((array_1, array_2, array_3, array_4), axis=2)
    test = np.apply_along_axis(my_func, 2, np.array(array_image_single))
    return test


if __name__ == '__main__':
    start = datetime.now()
    img = open_image('..\\img1.png')
    rotated_img = rotation_images(img, 0.1)
    contrast_img = contrast_image(rotated_img)
    find_counters_img = find_counters(contrast_img[0], contrast_img[1])
    share_counters_img = share_counters(find_counters_img)
    sorted_counters_img = sorted_counters(share_counters_img, rotated_img)
    resize_img = resize_array_images(sorted_counters_img, 2)
    TRUNC_tresh, TRUNC_dst = processing_image(resize_img, 2, 145)
    TOZERO_tresh, TOZERO_dst = processing_image(resize_img, 3, 180)
    unification_array_img = unification_arrays(TRUNC_tresh, TRUNC_dst, TOZERO_tresh, TOZERO_dst)
    test_pd = pd.DataFrame(unification_array_img[1:], columns=unification_array_img[0])
    test_pd.to_excel(f'result4.xlsx', encoding='utf-16')
    print(datetime.now() - start)
