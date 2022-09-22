import numpy as np
import cv2
import math
from classes.contour import ContourOfTable


def find_angle(contour: ContourOfTable) -> float:
    edge1 = np.int0((contour.top_right[0] - contour.top_left[0], contour.top_right[1] - contour.top_left[1]))
    edge2 = np.int0((contour.down_right[0] - contour.top_right[0], contour.down_right[1] - contour.top_right[1]))

    # выясняем какой вектор больше
    used_edge = max(edge1, edge2)

    reference = (1, 0)  # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = 180.0 / math.pi * math.acos(
        (reference[0] * used_edge[0] + reference[1] * used_edge[1]) / (cv2.norm(reference) * cv2.norm(used_edge)))
    if angle > 10.0:  # Костыль. Есть ситуации при которых по какой-то причине не корректно определяется угол.
        angle = 1.0
    if contour.down_right[1] > contour.top_left[1]:
        angle = -angle - 0.1
    return angle


def find_box(counter):
    _, _, w, h = cv2.boundingRect(counter)
    rect = cv2.minAreaRect(counter)  # пытаемся вписать прямоугольник
    box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координат
    box = np.sort(box, axis=0)
    return box, w, h


