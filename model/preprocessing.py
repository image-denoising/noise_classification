import glob
import os.path as system
import cv2 as cv
from PIL import Image
import random
from .noise import Noise
import matplotlib.pyplot as plt

# Constants
gauss_color = "./dataset/color/train/gauss/"
sp_gray = "./dataset/gray/train/sp/"
sp_color = "./dataset/color/train/sp/"
none_color = "./dataset/color/train/none/"
none_gray = "./dataset/gray/train/none/"


class Preprocessing:

    # Transform & save images list from RGB to Grayscale
    @staticmethod
    def images_to_grayscale():
        path = glob.glob(f"{none_color}*[.jpg|.jpeg|.png]")
        count = 0
        for image in path:
            count = count + 1
            print(f"image {count} : {image}")
            image_loaded = Image.open(image).convert('LA')
            image_loaded.save(none_gray + system.basename(image).split('.')[0] + '.png')

    # Transform & save test images list from RGB to Grayscale
    @staticmethod
    def test_images_to_grayscale(from_folder,to_folder):
        path = glob.glob(f"{from_folder}*[.jpg|.jpeg|.png]")
        for image in path:
            image_loaded = Image.open(image).convert('LA')
            image_loaded.save(to_folder + system.basename(image).split('.')[0] + '.png')

    # Apply noise & save grayscale images
    @staticmethod
    def apply_noise_gray():
        # path = glob.glob(f"{none_gray}*.png")
        path = glob.glob("./dataset/gray/test/none/*.png")
        count = 0
        percent_count = 0
        probability = 0.05
        for image in path:
            count = count + 1
            print(f"image [{count}] -> Basename : {image}")
            # load image from path
            image_loaded = cv.imread(image, 0)

            # change probability of noise applied
            if percent_count == 250:
                probability = probability + 0.05
                percent_count = 0

            # increments number of images with same noise
            percent_count = percent_count + 1

            # apply salt and pepper noise to image
            result = Noise.sp_noise(image=image_loaded, probability=probability)

            # save noised image
            # plt.imsave(sp_gray + system.basename(image), result, vmin=0, vmax=255, cmap="gray")
            plt.imsave("./dataset/gray/test/sp/" + system.basename(image), result, vmin=0, vmax=255, cmap="gray")

    # Apply noise & save RGB images
    @staticmethod
    def apply_noise_color(noise_type="gauss"):
        # path = glob.glob(f"{none_color}*[.jpg|.jpeg|.png]")
        path = glob.glob("./dataset/color/test/none/*[.jpg|.jpeg|.png]")
        percent_count = 0
        probability = 0.05
        count = 0
        for image in path:
            count = count + 1
            print(f"Image [{count}] -> Basename : {image}")
            # load image from path
            image_loaded = cv.imread(image)
            image_loaded = cv.cvtColor(image_loaded, cv.COLOR_BGR2RGB)

            # change probability of noise applied
            if percent_count == 500:
                probability = probability + 0.05
                percent_count = 0

            # increments number of images with same noise
            percent_count = percent_count + 1

            # apply gaussian or salt and pepper noise to image
            if noise_type == "gauss":
                image_loaded = image_loaded / 255
                result = Noise.gauss_noise(image=image_loaded, probability=probability)
            else:
                result = Noise.sp_noise(image=image_loaded, probability=probability)

            # save noised image
            if noise_type == "gauss":
                # plt.imsave(gauss_color + system.basename(image), result)
                plt.imsave("./dataset/color/test/gauss/" + system.basename(image), result)
            else:
                # plt.imsave(sp_color + system.basename(image), result)
                plt.imsave("./dataset/color/test/sp/" + system.basename(image), result)

    # Apply noise & save RGB test images
    @staticmethod
    def apply_noise_test_color(from_folder, to_folder, noise_type="gauss"):
        path = glob.glob(f"{from_folder}*[.jpg|.jpeg|.png]")
        for image in path:
            print("Basename : "+image)
            # load image from path
            image_loaded = cv.imread(image)
            image_loaded = cv.cvtColor(image_loaded, cv.COLOR_BGR2RGB)

            # apply gaussian or salt and pepper noise to image
            probability = random.uniform(0.1, 1)
            print("Probability : "+str(probability))
            if noise_type == "gauss":
                image_loaded = image_loaded / 255
                result = Noise.gauss_noise(image=image_loaded, probability=probability)
            else:
                result = Noise.sp_noise(image=image_loaded, probability=probability)

            # save noised image
            plt.imsave(to_folder + system.basename(image), result)
