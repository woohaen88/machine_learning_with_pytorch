import cv2
import numpy as np

try:
    from config import configuration as C
except:
    from IMAGE_CLASSIFICATION_PRETRAINED_MODEL.config import configuration as C


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (C.IMAGE_SIZE, C.IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    image -= C.MEAN
    image /= C.STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return image
