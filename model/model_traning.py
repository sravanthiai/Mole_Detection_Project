import os
from tqdm import tqdm
import matplotlib.pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers

from preprocess.preprocessing import valid_generator, training_generator,test_generator

STEP_SIZE_TRAIN = training_generator.n // training_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

classifier = Sequential()

# Step 1 - Convolution
classifier.add(
    Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a forth convolutional layer
classifier.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 4 - Flattening
classifier.add(Flatten())

# Step 5 - Full connection
classifier.add(Dense(units=128, kernel_initializer='he_uniform'))
# classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
# classifier.add(Dropout(0.5))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = classifier.fit_generator(generator=training_generator,
                                   steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=valid_generator,
                                   validation_steps=STEP_SIZE_VALID,
                                   epochs=10
                                   )
score=classifier.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)
#print(classifier.predict(valid_generator))
#model detector
classifier.save('mole_detector.h5')


fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()
for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
plt.savefig('CNN_Loss_Accuracy_plots')
