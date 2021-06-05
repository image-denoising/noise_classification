from model.gray_train import GrayTrain
from model.color_train import ColorTrain
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# GrayTrain.training()
# ColorTrain.training()
