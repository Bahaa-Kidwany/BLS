import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import cv2
import imghdr
cnn_model = load_model("B:/B.Kidwany/Graduation Project/Models/Highest.h5")
im_path = ">>>PUT THE DIRECTORY HERE<<<"
img = cv2.imread(im_path)
resize = tf.image.resize(img, (128,128))
pred = loaded.predict(np.expand_dims(resize/255,0))
if np.argmax(pred) == 0:
  print("The Plant Is Infected")
else: print("The Plant Is Healthy")