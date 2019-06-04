import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot

CATEGORIES = ["Dog", "Cat"]


def predict(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = load_model("catndog.h5")

predict_image = model.predict(predict("cat.jpg"))

if(predict_image[0][0] == 1):
    print('Result of predict: ',CATEGORIES[0])
elif(predict_image[0][1] == 1):
    print('Result of predict: ',CATEGORIES[1])


print(predict)
