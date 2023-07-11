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

The dataset used in this code consists of images of happy and sad faces. The images are organized in separate directories for each class. 
### The code uses the image_dataset_from_directory function from TensorFlow to load the dataset. It also performs some preprocessing steps, such as resizing the images and scaling the pixel values between 0 and 1.

# Preprocessing

Before training the model, some preprocessing steps are applied to the images. The dodgy or irrelevant images are removed using OpenCV and the imghdr library. The remaining images are then scaled by dividing the pixel values by 255 to bring them in the range of 0 to 1.
# Model Architecture

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
# EXPLANATION OF LIBRARIES AND FUNCTIONS USED:
functions and libraries used:

## Installing Required Packages:
pip install tensorflow opencv-python matplotlib: This command installs the required packages, including TensorFlow, OpenCV, and Matplotlib.

## Importing Libraries
import tensorflow as tf: Imports the TensorFlow library for deep learning operations.

import os: Allows navigating through directories.

import cv2: Provides computer vision functions, including image reading and manipulation.

from matplotlib import pyplot as plt: Enables visualization of images and plots.

## Removing Dodgy Images
This section aims to remove images with unsupported extensions or issues.
The code iterates over each image in the specified directory and checks the file extension using the imghdr moduIf the image extension is not in the allowed list (jpeg, jpg, bmp, png), the image is removed using ### os.remove().

## Loading Data
### The code uses tf.keras.utils.image_dataset_from_directory to load the images from a directory structure.
The function reads images from subdirectories, where each subdirectory represents a different class or label.
### The loaded data is converted into a tf.data.Dataset object, which provides an efficient way to work with large datasets.
The data is then batched and preprocessed by scaling the pixel values between 0 and 1.
# BUILDING THE PIPELINE

### The tf.keras.utils.image_dataset_from_directory() function is a convenient way to load images from a directory and create a tf.data.Dataset object. It simplifies the process of loading and preprocessing image data for training a deep learning model.

Here's an explanation of the parameters and functionalities of this function:

    'data': This is the directory path where your images are stored. The function will recursively search for image files within this directory and its subdirectories.

    labels='inferred': By default, the function infers the class labels from the directory structure. Each subdirectory within the main directory represents a different class or label. The function assigns the corresponding label to each image based on the subdirectory it is located in.

    label_mode='int': Specifies the type of labels returned. The default value is 'int', which means the labels will be returned as integer values representing the class indices. Other options include 'categorical', which returns one-hot encoded labels, or 'binary', which returns binary labels.

    color_mode='rgb': Specifies the color mode of the loaded images. The default value is 'rgb', which loads the images in RGB format. Other options include 'grayscale' or 'rgba'.

    batch_size=32: Sets the batch size for the dataset. The loaded images will be divided into batches of this size. A larger batch size can speed up training but may require more memory.

    image_size=None: Resizes the images to the specified dimensions. If None, the images will not be resized and will retain their original dimensions. Providing a tuple like (height, width) will resize the images accordingly.

    validation_split=None: If not None, it splits the dataset into training and validation sets. The fraction specified here represents the proportion of images used for validation.

    subset=None: If validation_split is set, subset can be used to specify whether to load the training or validation subset of the data.

    seed=None: Provides a random seed for shuffling the data during the loading process. This ensures reproducibility if the same seed is used in different runs.

The function returns a tf.data.Dataset object that represents the loaded image data. This dataset can be further processed and used for training a deep learning model. It will contain batches of images and their corresponding labels, ready to be fed into the model.

### By using tf.keras.utils.image_dataset_from_directory(), you can easily load and preprocess image data, including resizing, batching, and labeling, to build an efficient data pipeline for deep learning tasks.

## Splitting Data:
### The loaded data is split into training, validation, and testing sets using the tf.data.Dataset methods ###take() and skip().

## Building the Deep Learning Model:
The code uses the Sequential model from TensorFlow to build a deep neural network.

## The model consists of multiple layers:

Convolutional layers (Conv2D) with different filter sizes and strides, using the ReLU activation function.

Max pooling layers (MaxPooling2D) to downsample the spatial dimensions.

A flatten layer to convert the multi-dimensional output to a one-dimensional vector.

Dense layers (Dense) with different units and activation functions.

The model is compiled using the Adam optimizer and the binary cross-entropy loss function.

A summary of the model architecture is displayed using model.summary().

## Training the Model:
The model is trained using the ### fit() method.
The training data is provided along with the specified number of epochs, validation data, and a tensorboard callback for monitoring.

The training progress, including the loss and accuracy, is displayed during the training process.

## Evaluating Model Performance:
The model's performance is evaluated using the test data.
The metrics used include precision, recall, and binary accuracy.

### The tf.keras.metrics classes, such as Precision, Recall, and BinaryAccuracy, are used to calculate these metrics.
The values of precision, recall, and accuracy are printed out.

## Making Predictions:

An image (sad.jpg) is loaded using OpenCV and displayed using Matplotlib.
The image is resized to match the input size expected by the model.
The model predicts the class probabilities using ### model.predict() .
The predicted class is determined based on the threshold of 0.5, and the result is printed out.

## Saving the Model
The trained model is saved using ### model.save() to a specified path.
This code demonstrates a basic pipeline for building and training a deep learning model using TensorFlow and performing image classification tasks. It includes data preprocessing, model construction, training, evaluation, and prediction.

