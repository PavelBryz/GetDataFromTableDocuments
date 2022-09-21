import collections
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
import math
from pdf2image import convert_from_path
import tesserocr
from PIL import Image
from matplotlib import cm

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Niyazov.I\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


class Counter():
    def __init__(self, box, w, h, standard_image, img_contrast, angle):
        self.angle = angle

        self.top_left_x = box[0][0]
        self.top_left_y = box[0][1]

        self.top_left_y_sorted = box[0][1]

        self.down_left_x = box[1][0]
        self.down_left_y = box[2][1]

        self.top_right_x = box[2][0]
        self.top_right_y = box[1][1]

        self.down_right_x = box[3][0]
        self.down_right_y = box[3][1]

        self.standard_image = standard_image
        self.img_contrast = img_contrast

        self.w = w
        self.h = h

        self.column = None

        self.count_inner = 0

    def __repr__(self):
        # return f"{self.top_left_y}|{self.column}"
        return f"{self.top_left_x}-{self.top_left_y}-{self.column}"

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
        if ((self.top_left_x + 2 > other.top_left_x) and (self.top_left_y + 2 > other.top_left_y) and (
                self.down_right_x - 2 < other.down_right_x) and (self.down_right_y - 2 < other.down_right_y)) or (
                (self.top_right_x - 2 < other.top_right_x) and (self.top_right_y + 2 > other.top_right_y) and (
                self.down_left_x + 2 > other.down_left_x) and (self.down_left_y - 2 < other.down_left_y)):
            self.count_inner += 1

    def crop_standard_image(self):
        crop = self.standard_image[self.top_left_y - 4: self.top_left_y + self.h + 4,
               self.top_left_x - 4: self.top_left_x + self.w + 4]
        return crop

    def crop_image_contrast(self):
        crop = self.img_contrast[self.top_left_y - 4: self.top_left_y + self.h + 4,
               self.top_left_x - 4: self.top_left_x + self.w + 4]
        return crop

    def coordinates(self):
        coordinate_array = np.array([[int(self.top_left_x - 5), int(self.top_left_y - 3)],
                                     [int(self.top_right_x + 5), int(self.top_right_y - 3)],
                                     [int(self.down_right_x + 5), int(self.down_right_y + 3)],
                                     [int(self.down_left_x - 5), int(self.down_left_y + 3)]])
        return coordinate_array

    def coordinates_for_table(self):
        coordinate_array = np.array([[int(self.top_left_x), int(self.top_left_y)],
                                     [int(self.top_right_x), int(self.top_right_y)],
                                     [int(self.down_right_x), int(self.down_right_y)],
                                     [int(self.down_left_x), int(self.down_left_y)]])
        return coordinate_array


# Функция преобразования pdf в png
def pdf_to_png(pdf_document):
    png = []
    images = convert_from_path(pdf_document, 300,
                               poppler_path=r'C:\Users\Niyazov.I\PycharmProjects\pythonProject\GetDataFromSTOADoc\utilities\poppler-22.04.0\Library\bin')
    for image in images:
        png.append(np.array(image))
    return png


# Функция увеличения контрастности
def contrast_image(image, lim):
    """
    :param image: получает массив цветов каждого пикселя
    :return: преобразованный массив цветов каждого пикселя с улучшенной контрастностью
    """
    standard_image = image
    # разделяем массив для получения контрастности пикселей
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # увеличиваем контрастность кадого пикселя
    clahe = cv2.createCLAHE(clipLimit=lim, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # объединяем назад в один массив
    limg = cv2.merge((cl, a, b))
    img_contrast = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    return img_contrast, standard_image


def find_angle(box):
    edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
    edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

    # выясняем какой вектор больше
    usedEdge = edge1
    if cv2.norm(edge2) > cv2.norm(edge1):
        usedEdge = edge2
    reference = (1, 0)  # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = 180.0 / math.pi * math.acos(
        (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv2.norm(reference) * cv2.norm(usedEdge)))
    if angle > 10:
        angle = 1
    return angle


# Функция нахождения конутров
def find_counters(image, standard_image, iter=False, table=False):
    all_counters = []
    our_counters = []
    (height, weight) = image.shape[:2]
    # преобразовывем изображение в оттенки серого
    gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
    # производим бинаризацию
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if iter:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if table:
        counters, hi = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:
        counters, hi = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_counter = cv2.drawContours(image_contrast, counters, -1, (0, 0, 255), 1, cv2.LINE_8, hierarchy=hi)
    # save_or_show_image(None, image_counter)
    for counter in counters:
        x, y, w, h = cv2.boundingRect(counter)
        rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        my_data = np.sort(box, axis=0)
        if my_data[0][0] == 0 or my_data[0][1] == 0 or w == 0 or h == 0:
            continue

        # нахождение точек таким образом мы можем найти прямоугольники и определить их длину
        sm = cv2.arcLength(counter, True)
        apd = cv2.approxPolyDP(counter, 0.01 * sm, True)

        if iter:
            angle = find_angle(box)
            for coordinates in [my_data]:
                if abs(my_data[2][1] - my_data[0][1]) >= 5 and abs(my_data[2][0] - my_data[0][0]) >= 50:
                    if box[2][1] > box[0][1]:
                        all_counters.append(Counter(coordinates, w, h, standard_image, image, -angle - 0.1))
                    else:
                        all_counters.append(Counter(coordinates, w, h, standard_image, image, angle))
        elif table:
            for coordinates in [my_data]:
                if abs(my_data[2][1] - my_data[0][1]) >= 12 and abs(
                        my_data[2][0] - my_data[0][0]) >= 40 and w < weight and h < height:
                    all_counters.append(Counter(coordinates, w, h, standard_image, image, 0))

        else:
            if 4 <= len(apd) <= 6 and w > 50 and h > 30:
                angle = find_angle(box)
                for coordinates in [my_data]:
                    if box[2][1] > box[0][1]:
                        all_counters.append(Counter(coordinates, w, h, standard_image, image, -angle - 0.1))
                    else:
                        all_counters.append(Counter(coordinates, w, h, standard_image, image, angle))

    if table:
        for counters in all_counters:
            for j in all_counters:
                counters.is_inside(j)
            if counters.count_inner == 2:
                our_counters.append(counters)

        return our_counters
    else:
        return all_counters


# Функция поворота изображения
def rotation_image(image):
    """
    :param image: получает изображение
    :return: изображение повернутое на определенный градус
    """
    img = image.crop_standard_image()
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, image.angle, 1)

    # вычисляем абсолютное значение cos и sin
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[1, 0])

    # находим новые границы ширины и высоты
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # используем старые центры изображения и добавляем новые координаты
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    rotated = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
    return rotated


# Функция распределения контуров по линии
def share_counters(array_counters):
    col = 0
    array_all = []

    while [y for y in array_counters if y.column is None].__len__() > 0:
        col += 1
        arr = [y for y in array_counters if y.column is None]
        min_y = min(arr).top_left_y
        col_y = [y for y in arr if (abs(y.top_left_y - min_y) <= 15)]
        for el in col_y:
            el.column = col
            el.top_left_y_sorted = min_y
        array_all.append(col_y)

    return array_all


# Функция сортировки контуров
def sorted_counters(our_counters):
    array_all = []  # массив внутри которого другие массивы

    for counters in our_counters:
        our_counters_part = sorted(counters, key=lambda counter: [counter.top_left_y_sorted, counter.top_left_x])
        array_all.append(our_counters_part)

    return array_all


# Функция изменения размеров всех контуров
def resize_array_images(array_all, res, quality):
    images = crop_images(array_all, quality)
    array_all_resize = []
    for img_line in images:
        array_all_resize_part = []
        for img in img_line:
            h, w = (img.shape[1], img.shape[0])
            if len(img) == 0 or h == 0 or w == 0:
                continue
            temp_img = resize_image(img, res, h, w)
            array_all_resize_part.append(temp_img)
        array_all_resize.append(array_all_resize_part)
    return array_all_resize


# Функция изменения размеров ячейки
def resize_image(image, value, h, w):
    if value <= 0:
        raise ValueError
    image = cv2.resize(image, (int(h * value), int(w * value)))
    return image


# Функция разделения контуров
def crop_images(counters_array, quality):
    croped_images = []

    for counter_line in counters_array:
        croped_images_line = []
        for counter in counter_line:
            # cv2.imshow('crop', counter.crop_standard_image())
            # cv2.waitKey(0)
            if quality == 'standard':
                image = counter.crop_standard_image()
                croped_images_line.append(image)
            if quality == 'contrast':
                image = counter.crop_image_contrast()
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
            h, w = image.shape[:2]
            crop_black = image[0 + 20:h - 20, 0 + 20:w - 20]
            gray = cv2.cvtColor(crop_black, cv2.COLOR_BGR2GRAY)
            image_black = cv2.threshold(gray, 25, 255, 0)[1]
            black_pic = np.sum(image_black == 0)
            print(black_pic)

            if black_pic != 0:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                image_ke = cv2.filter2D(gray, -1, kernel)  # увеличиваем резкость
                thresh = cv2.threshold(image_ke, threshhold, 255, method)[1]  # производим обработку threshold
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dst = cv2.erode(thresh, kernel_erode, iterations=1)  # производим коррозию

                # проверку на наличие текста и в зависимости от этого уже решаем что добавлять
                array_part_thresh.append(text_on_image_recognition(thresh))
                array_part_dst.append(text_on_image_recognition(dst))
            else:
                array_part_thresh.append('Пусто')
                array_part_dst.append('Пусто')

        array_all_thresh.append(array_part_thresh)
        array_all_dst.append(array_part_dst)

    return array_all_thresh, array_all_dst


def processing_image_single(image, method, threshhold):
    array_single_thresh, array_single_dst = [], []
    # image = cv2.bilateralFilter(image, 5, 10, 10)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    image_ke = cv2.filter2D(gray, -1, kernel)  # увеличиваем резкость
    thresh = cv2.threshold(image_ke, threshhold, 255, method)[1]  # производим обработку threshold
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dst = cv2.erode(thresh, kernel_erode, iterations=1)  # производим коррозию

    array_single_thresh.append(text_on_image_recognition(thresh))
    array_single_dst.append(text_on_image_recognition(dst))

    return array_single_thresh, array_single_dst


# Функция распознавания текста
def text_on_image_recognition(images):
    config = r'--psm 6'
    # text_on_image_2 = pytesseract.image_to_data(images, lang="cor+cog", config=config)
    image = Image.fromarray(np.uint8(cm.gist_earth(images) * 255))
    text_on_image = tesserocr.image_to_text(image, lang="cor+cog",
                                            path=r'C:\Users\Niyazov.I\AppData\Local\Programs\Tesseract-OCR\tessdata')
    text_on_image = text_on_image_replace(text_on_image).replace("\n", " ").strip()
    # print(text_on_image)

    if len(text_on_image) == 0:
        return 'не распознал'
    if text_on_image.lower() == 'nn' or text_on_image.lower() == 'пп':
        return 'ПП'
    elif text_on_image.lower() == 'wt':
        return 'шт'
    else:
        return text_on_image


# Функция очистки распознаного текста
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
        return np.array('не распознал', dtype="<U1000")
    if np.all(array_image_single == 'Пусто'):
        return np.array('Пусто', dtype="<U1000")
    array_image_single = np.delete(np.array(array_image_single),
                                   np.where(np.array(array_image_single) == 'не распознал'))
    array_image_single = np.delete(np.array(array_image_single),
                                   np.where(np.array(array_image_single) == 'Пусто'))
    if len(array_image_single) == 1:
        return np.array(array_image_single[0], dtype="<U1000")
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
    test_2 = collections.Counter(qwe).most_common(1)
    if len(test_2) == 0:
        return np.array(' ', dtype="<U1000")
    else:
        return np.array(test_2[0][0], dtype="<U1000")


# Функция создания одного массива для таблиц
def unification_arrays(TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                       TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                       TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                       TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4):
    array_all = [TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                 TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                 TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                 TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4]
    for i in array_all:
        append_table(i)
    array_image_single = np.stack((TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                                   TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                                   TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                                   TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4), axis=2)

    test = np.apply_along_axis(my_func, 2, array_image_single)
    return test


def unification_single(TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                       TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                       TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                       TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4):
    array_image_single = np.stack((TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                                   TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                                   TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                                   TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4), axis=1)

    test = np.apply_along_axis(my_func, 1, array_image_single)
    return test


def append_table(array):
    max_len = max(map(len, array))
    for i in array:
        while len(i) < max_len:
            i.insert(0, 'Пусто')
    return array


def save_or_show_image(name, image):
    if type(name) != str:
        cv2.imshow('show', image)
        cv2.waitKey(0)
    else:
        cv2.imwrite(name, image)


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    con = []
    min_top_left_x, min_top_left_y = min([test.top_left_x for test in img_list]), min(
        [test.top_left_y for test in img_list])
    max_top_right_x, min_top_right_y = max([test.top_right_x for test in img_list]), min(
        [test.top_right_y for test in img_list])
    min_down_left_x, max_down_left_y = min([test.down_left_x for test in img_list]), max(
        [test.down_left_y for test in img_list])
    max_down_right_x, max_down_right_y = max([test.down_right_x for test in img_list]), max(
        [test.down_right_y for test in img_list])
    coor = [[min_top_left_x, min_top_left_y], [max_top_right_x, min_top_right_y], [min_down_left_x, max_down_left_y],
            [max_down_right_x, max_down_right_y]]
    for image in img_list:
        con.append(image.crop_standard_image())
    # take maximum width
    h_min = max(img.shape[0]
                for img in con)

    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation
                                 =interpolation)
                      for img in con]
    # return final image
    return cv2.hconcat(im_list_resize), coor


def dataframe_table(dataframe, unification_array_img, id, page_count, table_count, table):
    if dataframe is not None:
        dataframe_2 = pd.DataFrame(unification_array_img).stack().reset_index()
        dataframe_2.rename(columns={'level_0': 'Строка', 'level_1': 'Колонка', 0: 'Значение'}, inplace=True)
        dataframe_2.insert(0, 'Счет', id, allow_duplicates=True)
        dataframe_2.insert(1, 'Номер страницы', page_count, allow_duplicates=True)
        dataframe_2.insert(2, 'Вид объекта', 'таблица', allow_duplicates=True)
        dataframe_2.insert(3, 'Номер таблицы', table_count, allow_duplicates=True)
        dataframe_2.insert(6, 'Номер строки', None, allow_duplicates=True)
        dataframe_2['Координаты'] = f'{table.coordinates_for_table()}'
        dataframe_2['Y'] = f'{table.top_left_y}'
        dataframe = pd.concat([dataframe, dataframe_2], axis=0, sort=False)
    else:
        dataframe = pd.DataFrame(unification_array_img).stack().reset_index()
        dataframe.rename(columns={'level_0': 'Строка', 'level_1': 'Колонка', 0: 'Значение'}, inplace=True)
        dataframe.insert(0, 'Счет', id, allow_duplicates=True)
        dataframe.insert(1, 'Номер страницы', page_count, allow_duplicates=True)
        dataframe.insert(2, 'Вид объекта', 'таблица', allow_duplicates=True)
        dataframe.insert(3, 'Номер таблицы', table_count, allow_duplicates=True)
        dataframe.insert(6, 'Номер строки', None, allow_duplicates=True)
        dataframe['Координаты'] = f'{table.coordinates_for_table()}'
        dataframe['Y'] = f'{table.top_left_y}'
    return dataframe


def dataframe_text(dataframe, unification_array_img, id, page_count, string_count, coordinates_text):
    if dataframe is not None:
        dataframe_2 = pd.DataFrame(unification_array_img).stack().reset_index()
        dataframe_2.rename(columns={'level_0': 'Строка', 'level_1': 'Колонка', 0: 'Значение'}, inplace=True)
        dataframe_2['Строка'] = None
        dataframe_2['Колонка'] = None
        dataframe_2.insert(0, 'Счет', id, allow_duplicates=True)
        dataframe_2.insert(1, 'Номер страницы', page_count, allow_duplicates=True)
        dataframe_2.insert(2, 'Вид объекта', 'строка', allow_duplicates=True)
        dataframe_2.insert(3, 'Номер таблицы', None, allow_duplicates=True)
        dataframe_2.insert(6, 'Номер строки', string_count, allow_duplicates=True)
        dataframe_2['Координаты'] = f'{coordinates_text}'
        dataframe_2['Y'] = f'{coordinates_text[0][1]}'
        dataframe = pd.concat([dataframe, dataframe_2], axis=0, sort=False)
    else:
        dataframe = pd.DataFrame(unification_array_img).stack().reset_index()
        dataframe.rename(columns={'level_0': 'Строка', 'level_1': 'Колонка', 0: 'Значение'}, inplace=True)
        dataframe['Строка'] = None
        dataframe['Колонка'] = None
        dataframe.insert(0, 'Счет', id, allow_duplicates=True)
        dataframe.insert(1, 'Номер страницы', page_count, allow_duplicates=True)
        dataframe.insert(2, 'Вид объекта', 'строка', allow_duplicates=True)
        dataframe.insert(3, 'Номер таблицы', None, allow_duplicates=True)
        dataframe.insert(6, 'Номер строки', string_count, allow_duplicates=True)
        dataframe['Координаты'] = f'{coordinates_text}'
        dataframe['Y'] = f'{coordinates_text[0][1]}'
    return dataframe


if __name__ == '__main__':
    dataframe = None
    resize = 2
    contrast = 2.5
    start = datetime.now()
    pdf_t = ['Счет3.pdf']
    for pdf_x in pdf_t:
        pdf = pdf_to_png(pdf_x)
        page_count = 1
        for single_image in pdf:
            print('Страница: ', page_count)
            image_contrast = contrast_image(single_image, 7.0)
            # Находим таблицы на изображении
            tables_on_image = find_counters(image_contrast[0], image_contrast[1])
            share_counters_tables = share_counters(tables_on_image)
            if len(share_counters_tables) != 0:
                sorted_counters_tables = np.hstack(sorted_counters(share_counters_tables))
                #  Производим распознавание текста и удаление таблицы с изображения
                table_count = 1
                for table in sorted_counters_tables:
                    rotation = rotation_image(table)
                    contrast_img = contrast_image(rotation, contrast)
                    find_counters_in_table = find_counters(contrast_img[0], contrast_img[1], table=True)
                    if len(find_counters_in_table) != 0:
                        share_counters_in_table = share_counters(find_counters_in_table)
                        sorted_counters_in_table = sorted_counters(share_counters_in_table)
                        resize_img_standard = resize_array_images(sorted_counters_in_table, resize, 'standard')
                        resize_img_contrast = resize_array_images(sorted_counters_in_table, resize, 'contrast')
                        TRUNC_tresh_1, TRUNC_dst_1 = processing_image(resize_img_contrast, 2, 145)
                        TOZERO_tresh_1, TOZERO_dst_1 = processing_image(resize_img_standard, 3, 180)
                        TRUNC_tresh_2, TRUNC_dst_2 = processing_image(resize_img_contrast, 2, 220)
                        TOZERO_tresh_2, TOZERO_dst_2 = processing_image(resize_img_standard, 3, 150)
                        TRUNC_tresh_3, TRUNC_dst_3 = processing_image(resize_img_standard, 2, 145)
                        TOZERO_tresh_3, TOZERO_dst_3 = processing_image(resize_img_contrast, 3, 180)
                        TRUNC_tresh_4, TRUNC_dst_4 = processing_image(resize_img_standard, 2, 220)
                        TOZERO_tresh_4, TOZERO_dst_4 = processing_image(resize_img_contrast, 3, 150)
                        unification_array_img = unification_arrays(TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1,
                                                                   TOZERO_dst_1,
                                                                   TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2,
                                                                   TOZERO_dst_2,
                                                                   TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3,
                                                                   TOZERO_dst_3,
                                                                   TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4,
                                                                   TOZERO_dst_4)
                        dataframe = dataframe_table(dataframe, unification_array_img, id, page_count, table_count,
                                                    table)
                        table_count += 1
                        single_image = cv2.drawContours(single_image, [table.coordinates()], -1, (255, 255, 255),
                                                        thickness=cv2.FILLED)
                        print(f'Table_{table}', datetime.now() - start)

            # Находим текст на изображении
            image_contrast = contrast_image(single_image, 7.0)
            text_on_image = find_counters(image_contrast[0], image_contrast[1], iter=True)
            share_counters_text = share_counters(text_on_image)
            if len(share_counters_text) != 0:
                sorted_counters_text = sorted_counters(share_counters_text)
                string_count = 1
                for text_image in sorted_counters_text:
                    if len(text_image) == 1:
                        coordinates_text = text_image[0].coordinates_for_table()
                        image = text_image[0].crop_standard_image()
                        (h, w) = image.shape[:2]
                        if h == 0 or w == 0:
                            continue
                        image_contrast_single = contrast_image(image, contrast)
                        image_standard = cv2.resize(image_contrast_single[1], (int(w * resize), int(h * resize)),
                                                    cv2.INTER_NEAREST)
                        image_contrast = cv2.resize(image_contrast_single[0], (int(w * resize), int(h * resize)),
                                                    cv2.INTER_NEAREST)
                    if len(text_image) >= 2:
                        image = hconcat_resize(text_image)
                        coordinates_text = image[1]
                        (h, w) = image[0].shape[:2]
                        if h == 0 or w == 0:
                            continue
                        image_contrast_single = contrast_image(image[0], contrast)
                        image_standard = cv2.resize(image_contrast_single[1], (int(w * resize), int(h * resize)),
                                                    cv2.INTER_NEAREST)
                        image_contrast = cv2.resize(image_contrast_single[0], (int(w * resize), int(h * resize)),
                                                    cv2.INTER_NEAREST)
                    TRUNC_tresh_1, TRUNC_dst_1 = processing_image_single(image_standard, 2, 145)
                    TOZERO_tresh_1, TOZERO_dst_1 = processing_image_single(image_contrast, 3, 180)
                    TRUNC_tresh_2, TRUNC_dst_2 = processing_image_single(image_standard, 2, 220)
                    TOZERO_tresh_2, TOZERO_dst_2 = processing_image_single(image_contrast, 3, 150)
                    TRUNC_tresh_3, TRUNC_dst_3 = processing_image_single(image_contrast, 2, 145)
                    TOZERO_tresh_3, TOZERO_dst_3 = processing_image_single(image_standard, 3, 180)
                    TRUNC_tresh_4, TRUNC_dst_4 = processing_image_single(image_contrast, 2, 220)
                    TOZERO_tresh_4, TOZERO_dst_4 = processing_image_single(image_standard, 3, 150)
                    unification_array_img = unification_single(TRUNC_tresh_1, TRUNC_dst_1, TOZERO_tresh_1, TOZERO_dst_1,
                                                               TRUNC_tresh_2, TRUNC_dst_2, TOZERO_tresh_2, TOZERO_dst_2,
                                                               TRUNC_tresh_3, TRUNC_dst_3, TOZERO_tresh_3, TOZERO_dst_3,
                                                               TRUNC_tresh_4, TRUNC_dst_4, TOZERO_tresh_4, TOZERO_dst_4)
                    dataframe = dataframe_text(dataframe, unification_array_img, id, page_count, string_count,
                                               coordinates_text)
                    string_count += 1

            page_count += 1
            print(f'Text', datetime.now() - start)
    dataframe[['Номер страницы', 'Y']] = dataframe[['Номер страницы', 'Y']].astype(int)
    dataframe.sort_values(by=['Номер страницы', 'Y'], inplace=True)
    dataframe.drop(columns='Y', axis=1, inplace=True)
    dataframe.to_excel(f'result.xlsx')
