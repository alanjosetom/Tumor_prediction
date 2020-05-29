#loading keras models
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from segmentation import segm
import cv2
#load saved model to the file for prediction
new_model = load_model('created_model.h5')
#load the testing image
test_image = image.load_img('/home/tech/Desktop/2 no.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
#make prediction 
result = new_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'yes'
else:
    prediction = 'no'
print(prediction)


#call the segmention function for make image segmentation
if(prediction == 'yes'):
    segm()