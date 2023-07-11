# Deep Learning Image Classification

This repository contains code for a deep learning image classification model using TensorFlow and OpenCV. The model is trained on a dataset of happy and sad images and is able to classify new images as happy or sad. The steps involved in the code are explained below.
Setup and Installation

## To run the code, you need to have the following libraries installed:

    TensorFlow
    OpenCV
    Matplotlib

You can install these libraries using the following command:

### pip install tensorflow opencv-python matplotlib

# Dataset

The dataset used in this code consists of images of happy and sad faces. The images are organized in separate directories for each class. The code uses the image_dataset_from_directory function from TensorFlow to load the dataset. It also performs some preprocessing steps, such as resizing the images and scaling the pixel values between 0 and 1.
Preprocessing

Before training the model, some preprocessing steps are applied to the images. The dodgy or irrelevant images are removed using OpenCV and the imghdr library. The remaining images are then scaled by dividing the pixel values by 255 to bring them in the range of 0 to 1.
Model Architecture

## The model used in this code is a deep neural network with multiple convolutional and dense layers. The architecture of the model is as follows:

###    Convolutional layer with 16 filters and a kernel size of 3x3.
###    MaxPooling layer to downsample the feature maps.
###    Convolutional layer with 32 filters and a kernel size of 3x3.
###    MaxPooling layer.
###    Convolutional layer with 16 filters and a kernel size of 3x3.
###    MaxPooling layer.
###    Flatten layer to convert the multi-dimensional feature maps into a one-dimensional vector.
###    Dense layer with 256 units and ReLU activation.
###    Dense layer with 1 unit and sigmoid activation (output layer).
