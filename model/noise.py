import matplotlib.pyplot as plt
import numpy as np
import random


class Noise:

    # Add Salt-and-Pepper noise to an image.
    @staticmethod
    def sp_noise(image, probability):
        output = np.copy(image)
        x = 1 - probability
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rand = random.random()
                if rand < probability:
                    if random.randint(0, 1) == 0:
                        output[i][j] = 0
                    else:
                        output[i][j] = 255
                elif rand > x:
                    output[i][j] = image[i][j]
        return output

    # Add gauss noise to an image.
    @staticmethod
    def gauss_noise(image, probability):
        # generate noise
        noise = np.random.normal(loc=0, scale=1, size=image.shape)

        # noise overlaid over image
        result = np.clip((image + noise * probability), 0, 1)

        return result
