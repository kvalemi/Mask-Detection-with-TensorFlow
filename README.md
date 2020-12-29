## Project Description

In this project I wanted to train a robust deep learning model to detect if the individuals in a captured frame, either being images, video, or real-time camera footage, are wearing masks. I primarily used OpenCV to process the dataset of images and TensorFlow for actually training and testing the models. I considered a lightweight Convolutional Neural Network structure called MobileNet as my Deep Learning model. I then tailored the model for binary classification (Mask or No Mask) by using a Sigmoid function to configure my output. After 5 epochs, I managed to acheive a 98% validation accuracy on test images. In addition, to support the feature of analyzing multiple faces in a frame I used Haar Cascades to first extract every individual frontal faces located in the frame, and then fed into my model for prediction. Furthermore, the initial go at the project was done in a Jupyter Notebok, but after finishing the testing I split the entire project into three Python scripts to make three standalone pipelines. The three scripts are:


(1) `Train_Model.py`: Used to train the model and save the parameters of the model to disk.

(2) `Real_Time_Mask_Detection.py`: Utilizes saved model from training pipeline to perform analysis of mask detection on a real-time camera footage.

(3) `Image_Mask_Detection.py`: Utilizes saved model from training pipeline to determining if the individuals in a static image are wearing masks. This script actually highlights the frontal face features of the frame and places a text nearby identifying if the frontal face contains a face mask.


Here are some examples of running the `Image_Mask_Detection` script on some test images (images the CNN model has not seen before):


- Test Image 1 passed into the script:

![](/Mask%20Detection%20Examples/Test_Image_1.jpg)

- Output of script from Test Image 1:

![](/Mask%20Detection%20Examples/Mask_detection_Evidence_1.png)


- Test Image 2 passed into the script:

![](/Mask%20Detection%20Examples/Test_Image_2.jpg)

- Output of script from Test Image 2:

![](/Mask%20Detection%20Examples/Mask_detection_Evidence_2.png)

- Test Image 3 passed into the script:

![](/Mask%20Detection%20Examples/Test_Image_3.jpg)

- Output of script from Test Image 3:

![](/Mask%20Detection%20Examples/Mask_detection_Evidence_3.png)


**If you would like to learn more about the project and the work I completed, please first take a look at the Jupyter Notebook and then consider checking out the individual Python scripts.**


## How to Build the Project

1) Ensure the following dependancies are downloaded on your target machine. Make sure TensorFlow is actually built from source on your target machine for an optimal experience.

- keras
- tensorflow
- cv2
- os
- matplotlib
- numpy as np
- random
- pickle


2) Obtain a dataset that has example images of people wearing and not wearing a face mask. I used a pretty small dataset (source found in credits).

- Ensure the folder structure follows the following schema in your repository:

-> ./Dataset

--> ./Dataset/Face_Mask

--> ./Dataset/No_Mask


3) Train the model by running the following command: `Python3 Train_Model.py`


4) After the model is trained we can run the remaining pipeline scripts as such:

- **Image_Mask_Detection**: `Python3 Image_Mask_Detection.py [Path to Image]`

- **Real_Time_Mask_Detection**: `Python3 Real_Time_Mask_Detection.py`


(I have configured the **Real_Time_Mask_Detection** script to use the front camera of the target system, but if you would like to use it for other cameras on your system, configure the `VideoCapture(0)` line according to the ID of your camera (i.e. 0 corresponds to the front camera of your system)


## Credits

TensorFlow MobileNet: https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet

Data Obtained From: https://github.com/prajnasb/observations/tree/master/experiements/data
