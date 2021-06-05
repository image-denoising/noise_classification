import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
import glob

# constants
train_path = './dataset/color/train/'
test_path = './dataset/color/test/'
validation_path = './dataset/color/validation/'
target_size = (32, 32)
batch_size = 128
class_names = ['gaussian noise', 'no filter applied', 'salt & pepper noise']


class ColorTrain:
    ds_train = None
    ds_validation = None
    ds_test = None

    # load & preprocessing train & validation & test dataset
    @staticmethod
    def load():
        data_generator = ImageDataGenerator(
            rescale=1./255,
            data_format="channels_last",
            dtype=tf.float32
        )

        ColorTrain.ds_train = data_generator.flow_from_directory(
            train_path,
            target_size=target_size,
            batch_size=batch_size,
            color_mode="rgb",
            class_mode="sparse",
            shuffle=True,
            seed=42
        )

        ColorTrain.ds_validation = data_generator.flow_from_directory(
            validation_path,
            target_size=target_size,
            batch_size=batch_size,
            color_mode="rgb",
            class_mode="sparse",
            shuffle=True,
            seed=42
        )

        ColorTrain.ds_test = data_generator.flow_from_directory(
            test_path,
            target_size=target_size,
            batch_size=batch_size,
            color_mode="rgb",
            class_mode="sparse",
            shuffle=True,
            seed=42
        )

    # convolution neural network model train
    @staticmethod
    def training():
        # load train & validation & test dataset
        ColorTrain.load()

        # convolution neural network model 'LetNet-5'
        model = models.Sequential()
        model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))

        # convolution neural network compile
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # convolution neural network training
        model.fit(ColorTrain.ds_train, epochs=50, validation_data=ColorTrain.ds_validation)

        # convolution neural network result
        loss, accuracy = model.evaluate(ColorTrain.ds_test)
        print(f"loss: {loss}")
        print(f"accuracy: {accuracy}")

        # convolution neural network save model
        model.save('./noise_classification_color.model')

    # convolution neural network model test
    @staticmethod
    def predict(path):
        # load model
        model = models.load_model('./classification/noise_classification_color.model')

        # loading & normalization of test image
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, target_size)
        image = np.array([image]) / 255

        # predict type of noise
        prediction = model.predict(image)
        print("Image type : "+class_names[np.argmax(prediction)])
        return np.argmax(prediction)

    # convolution neural network model test
    @staticmethod
    def global_testing():
        # load model
        model = models.load_model('./noise_classification_color.model')

        # load no filter test dataset
        none_count = 0
        path = glob.glob("./dataset/color/test/none/*.png")
        for image_path in path:
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=target_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = tf.expand_dims(image, axis=0)
            prediction = model.predict(image)
            if np.argmax(prediction) != 1:
                none_count = none_count + 1

        # load gauss test dataset
        gauss_count = 0
        path = glob.glob("./dataset/color/test/gauss/*.png")
        for image_path in path:
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=target_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = tf.expand_dims(image, axis=0)
            prediction = model.predict(image)
            if np.argmax(prediction) != 0:
                gauss_count = gauss_count + 1

        # load salt & pepper test dataset
        sp_count = 0
        path = glob.glob("./dataset/color/test/sp/*.png")
        for image_path in path:
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=target_size)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = tf.expand_dims(image, axis=0)
            prediction = model.predict(image)
            if np.argmax(prediction) != 2:
                sp_count = sp_count + 1

        print(f"Gaussian => success degree = {(4750 - gauss_count) * 100 / 4750} ")
        print(f"No filter => success degree = {(4750 - none_count) * 100/4750} ")
        print(f"Salt & pepper => success degree = {(4750 - sp_count) * 100/4750} ")
