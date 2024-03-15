from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys

from queue import Queue

cap = cv2.VideoCapture(0)

class_name = ['0','10.000 VND','20.000 VND','50.000 VND']
number_of_classes = len(class_name)

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Freezing layers
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Create model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Add FC, Dropout layer
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load trained model weights
my_model = get_model()
my_model.load_weights("weights-41-0.98.hdf5")

prediction_threshold = 0.8

queue_of_frames = Queue()
length_of_queue = 6    #   Length of total frames are being considered
classes_predicted_count = np.empty(number_of_classes)
same_class_in_queue = 5 # (<= length_of_queue)

while(True):
    ret, image_org = cap.read()
    if not ret:
        continue

    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print(np.max(predict[0],axis=0), "\t", predict[0])
    
    # Money detected case
    if (np.max(predict[0]) > prediction_threshold):
        predicted_class_index = np.argmax(predict[0])
        classes_predicted_count[predicted_class_index] += 1
        queue_of_frames.put(predicted_class_index)
    # Undetected case
    else:
        classes_predicted_count[0] += 1
        queue_of_frames.put(0)

    # # When the queue is not full
    # if (queue_of_frames.qsize() <= length_of_queue):

    # When the queue was full
    if (queue_of_frames.qsize() > length_of_queue):
        popped_class_index = queue_of_frames.get()
        classes_predicted_count[popped_class_index] -= 1

    # Result
    for i in range(1, number_of_classes):
        if (classes_predicted_count[i] > same_class_in_queue):
            print("This picture is: ", class_name[np.argmax(predict[0])])
            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 0.5
            color = (0, 0, 255)
            thickness = 1

            cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

