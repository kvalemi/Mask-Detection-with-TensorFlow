## Load packages and dependancies ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import sys



## Utility Functions ##

# Function for applying transformations
def apply_image_trans(img):
    final_image = cv2.resize(img, (img_size, img_size))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    
    return final_image

# Function for interpreting the sigmoid output
def convert_sigmoid_output(sigmoid):
    if(sigmoid < 0.50):
        print('Mask Detected!')
    elif(sigmoid > 0.50):
        print('Mask not Detected!')
    else:
        print('Not Sure!')



## Starting Point ##

# Upload the model (if you don't want to train it)
new_model = keras.models.load_model('./Mask_detection_Model.h5')

## some variables we will need for the Web Cam detection

# font size of detection box
font_scale = 1.5

# font type of detection text
font = cv2.FONT_HERSHEY_PLAIN

# path to haar cascades model
path = './haarcascade_face.xml'

# Coverting to image size that is required by ImageNet (224 x 224)
img_size = 224

# input image path
img_path = sys.argv[1]


# Load image
img = cv2.imread(img_path)

# apply transformations to chosen image
frame = apply_image_trans(img)

# this is the output of the model
Prediction = new_model.predict(frame)
convert_sigmoid_output(Prediction)


# focus on face inside of the frame
faceCascade = cv2.CascadeClassifier(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 3)


for x,y,w,h in faces:
    
    # Apply transformations
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)

    # check if any faces were found
    if len(facess) != 0:
        
        # Cropping the face
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]
        
        # obtain transformed face
        final_image = apply_image_trans(face_roi)

        if(Prediction > 0):

            # Set No Mask status
            status = "No Mask"

            # Add the text
            cv2.putText(img, status, (x, y), font, font_scale, (0,0,255), 3)

            # Add the rectangle
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 3)

            # Save the frame and or picture
            # cv2.imwrite('./No_mask_evidence_face.png', (cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)))
            # cv2.imwrite('./No_mask_evidence_frame.png', (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))        

        elif(Prediction < 0):

            # Set No Mask status
            status = "Face Mask"

            # Add the text
            cv2.putText(img, status, (x, y), font, font_scale, (255,0,0), 1)

            # Add the rectangle
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)


# show image
if len(faces) != 0:
    cv2.imwrite('./Mask_detection_Evidence.png', (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))        
