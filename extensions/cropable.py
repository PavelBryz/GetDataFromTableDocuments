from classes.contour import Contour


class Cropable:
    def crop_image(self, counter: Contour):
        crop = self.image[counter.top_left[1]: counter.top_left[1] + counter.get_height(),
                          counter.top_left[0]: counter.top_left[0] + counter.get_wight()]
        return crop