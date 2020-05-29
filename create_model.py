#loading keras models
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#save the model type to a variable
classifier = Sequential()
#adding layers to the model
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation='relu'))
#final layer of the cnn model
classifier.add(Dense(output_dim=1,activation='sigmoid'))
#optimizing
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#scale and preparing train dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
#load training dataset
training_set = train_datagen.flow_from_directory(
        'dataset_negitta/training/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#scale and preparing test dataset
test_set = test_datagen.flow_from_directory(
        'dataset_negitta/test/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#load test dataset
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=2000)
#save the ceated model
classifier.save('created_model.h5')
