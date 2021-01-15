import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# from preprocess.preprocessing import test_generator
PATH_MODEL = os.getcwd()

MODEL_FOLDER = PATH_MODEL +'\\model\\mole_detector.h5'
model = load_model(MODEL_FOLDER)


def predict(image_file):
    test_image = image.load_img(
        image_file,
        target_size=(128, 128))
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image)

    # print(training_generator.class_indices)
    if result[0][0] == 0:
        return 'you are not in danger'
    else:
        return 'you are in danger'
