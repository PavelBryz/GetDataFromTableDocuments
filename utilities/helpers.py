from typing import List

import cv2
import numpy as np


def angle_transform(el):
    if el > 80: el -= 90
    elif el < -80: el += 90
    else: pass
    return el


def find_box(counter):
    _, _, w, h = cv2.boundingRect(counter)
    rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
    box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координат
    box = np.sort(box, axis=0)
    return box, w, h


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
