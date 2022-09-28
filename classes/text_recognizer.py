import PIL
import cv2
# from PIL.Image import Image
from numpy.core.records import ndarray
import numpy as np
from matplotlib import cm
import tesserocr
from typing_extensions import Final

from classes.contour import Contour
from utilities.helpers import text_on_image_replace

KARNEL: Final[ndarray] = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])

TYPE_ERODE: Final[int] = 0
TYPE_THRESHOLD: Final[int] = 1


class Recognizer:
    @staticmethod
    def __is_empty(image: ndarray):
        _, image = cv2.threshold(image, 200, 255, 0)
        percent = np.sum(image == 0) / image.size * 100
        return percent < 1

    @staticmethod
    def __text_on_image_recognition(images):
        image = PIL.Image.fromarray(np.uint8(cm.gist_earth(images) * 255))
        text_on_image = tesserocr.image_to_text(image, lang="cor+cog", path=r'C:\Users\Bryzgalov.Pavel\AppData\Local\Tesseract-OCR\tessdata')
        text_on_image = text_on_image_replace(text_on_image)

        if len(text_on_image) == 0:
            return 'не распознал'
        if text_on_image.lower() == 'nn' or text_on_image.lower() == 'пп':
            return 'ПП'
        elif text_on_image.lower() == 'wt':
            return 'шт'
        else:
            return text_on_image

    @staticmethod
    def processing_image(image: ndarray, method, threshold, type_of_operation):
        if Recognizer.__is_empty(image):
            return 'Пусто'

        image_ke = cv2.filter2D(image, -1, KARNEL)  # увеличиваем резкость
        thresh = cv2.threshold(image_ke, threshold, 255, method)[1]  # производим обработку threshold
        if type_of_operation == TYPE_ERODE:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dst = cv2.erode(thresh, kernel_erode, iterations=1)  # производим коррозию

        if type_of_operation == TYPE_ERODE:
            text = Recognizer.__text_on_image_recognition(dst)
        else:
            text = Recognizer.__text_on_image_recognition(thresh)

        return text


