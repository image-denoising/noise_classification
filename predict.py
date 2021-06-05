import model.gray_train as gray_classification
import model.color_train as color_classification
from PIL import Image


class Predict:

    # predict noise type by image classification
    @staticmethod
    def noise_type(path):
        # get image color_mode = (RGB, Grayscale)
        image = Image.open(path)
        color_mode = image.mode

        # grayscale image
        if color_mode == 'L':
            return gray_classification.GrayTrain.predict(path)
        # rgb color image
        else:
            return color_classification.ColorTrain.predict(path)
