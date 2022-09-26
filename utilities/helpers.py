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


