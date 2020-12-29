## Project Description

In this project I wanted to train a robust deep learning model to detect if people in a frame, either being images, video, or real-time capture, are wearing a mask or not. I primarily used OpenCV to process the image datasets and TensorFlow for actually training and testing the model. I used a lightweight Convolutional Neural Network called MobileNet as my Deep Learning Model. I then tailored the model for binary classification (Mask or No Mask) by using a Sigmoid function to configure my output. After 5 epochs, I managed to acheive a 98% validation accuracy. In addition, to support analyzing multiple faces in a frame I used Haar Cascades to extract the individual frontal faces of the frame, which is then fed into my CNN model. Furthermore, the general workflow of training and testing can be found in the Jupyter Notebook, but I also split the entire project into three Python scripts to make three standalone pipelines. The three scripts are:

(1) `Train_Model.py`: Used to train the model and save the parameters of the model.

(2) `Real_Time_Mask_Detection.py`: Utilizes saved model from previous pipeline to perform analysis of mask detection on a real-time webcam or security camera.

(3) `Image_Mask_Detection.py`: Utilizes saved model from previous pipeline to perform analysis of mask detection on an inputted static image. The result of the analysis is outputted as text in the terminal and a saved picture.

Here are some examples of running the `Image_Mask_Detection` script on some test images (images the CNN model has not seen before):

Test Image 1 passed into the script:

![](/Test_Image_1.jpg)

Output of pipeline when ran on Test Image 1:

![](/Mask_detection_Evidence_1.png)


Test Image 2 passed into the script:

![](/Test_Image_2.jpg)

Output of pipeline when ran on Test Image 1:

![](/Mask_detection_Evidence_2.png)



If you would like to learn more about the project and the work I completed, please first take a look at the Jupyter Notebook and then the Python scripts.


## How to Build the Project

1) Ensure the following dependancies are downloaded on your target machine. Make sure TensorFlow is actually built from source on your target machine.

- keras
- tensorflow
- cv2
- os
- matplotlib
- numpy as np
- random
- pickle

2) Obtain a dataset that has example images of people wearing and not wearing face masks. I used a small images dataset


## Credits

TensorFlow MobileNet: https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNet

Data Obtained From: https://github.com/prajnasb/observations/tree/master/experiements/data
