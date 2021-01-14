import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

data = pd.read_csv('E:\BeCodeProjects\Mole_Detection_Project\Dataset\data\CLIN_DIA.csv', encoding='latin')
data = data.dropna()
data['kat.Diagnose'] = data['kat.Diagnose'].replace([0, 1], 0).replace([2, 3], 1)
# print(data)

images = os.listdir('E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/Datadef')
# creating list
benign = []
malignant = []

img_ids = []
for image in images:
    img_id = image  # .lower()#.split#('.')#.lower()
    img_ids.append(img_id)

# separating images based on danger(1.0) or no-danger(0.0) states
for i in range(len(images)):
    if data['kat.Diagnose'][i] == 0.0:
        benign.append(img_ids[i])
    elif data['kat.Diagnose'][i] == 1.0:
        malignant.append(img_ids[i])

for benign_data in benign:
    f = Image.open('E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/Datadef/' + benign_data)
    #f.save('E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/training_set/benign/' + benign_data)

# saves malignant images from original folder to newly created folder for malignant category
for malignant_data in malignant:
    f = Image.open('E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/Datadef/' + malignant_data)
    #f.save('E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/training_set/malignant/' + malignant_data)

# Data augmentation on training data and validation data
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255.0,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=0.2)

valid_datagen = ImageDataGenerator(rescale=1. / 255.0,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255.0)
training_generator = train_datagen.flow_from_directory(r'E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/training_set',
                                                       target_size=(128, 128),  # resize image,
                                                       batch_size=32,
                                                       save_to_dir=r'E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/resized',
                                                       save_format='bmp',
                                                       shuffle=True,
                                                       seed=42,
                                                       subset='training',
                                                       class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(directory=r'E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/training_set',
                                                    target_size=(128, 128),
                                                    subset='validation',
                                                    batch_size=32,
                                                    class_mode="binary",
                                                    shuffle=True,
                                                    seed=42)
"""
test_generator=test_datagen.flow_from_directory(directory=r'E:/BeCodeProjects/Mole_Detection_Project/Dataset/data/skin_cancer/test',
                                                    target_size=(128, 128),

                                                    batch_size=32,
                                                    class_mode="binary",
                                                    shuffle=True,
                                                    seed=42)
"""
