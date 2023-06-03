# Facial Emotion Classifier

![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)

# Project Overview

The goal of this project is to create a deep learning model capable of accurately categorizing facial expressions into seven distinct classes:</br>
***Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral***

We will leverage the power of **Convolutional Neural Networks (CNNs)**, which have proven to be highly effective in image classification tasks.</br>
Our approach involves training a CNN model from scratch using the FER-2013 dataset.

</br>

<img src="Al-Pacino.jpeg">

</br>

# Data Description

The FER-2013 dataset consists of grayscale images of faces, each measuring 48x48 pixels.</br>
These images have undergone preprocessing to ensure consistent alignment and similar spatial occupation across all samples.</br>
The dataset comprises a total of 24,400 images, with 28,709 images allocated to the training set and 7,178 images designated for the public test set.

# Main Process

## Steps

1. [First Organization](#step1)
2. [Second Step: Data Preprocessing](#step2)
3. [Third and Final Step: Modeling](#step3)

</br>

<a name="step1"></a>
## &emsp; 1. First Organization

In the first steps we'll get to know about our images and their specifications.</br>
To get familiar with the project, we'll show them.
###  &emsp; **Packages**

&emsp;  &emsp; ***Tensorflow***</br>
&emsp;  &emsp; ***OpenCV***</br>
&emsp;  &emsp; ***os, random, time, pickle***</br>
&emsp;  &emsp; ***Numpy***</br>
&emsp;  &emsp; ***Pandas***</br>
&emsp;  &emsp; ***Matplotlib***</br>

<a name="step2"></a>

## &emsp; 2. Second Step: Data Preprocessing

Here we'll equip our dataset to be ready foe modeling.</br>
As the dataset is well gathered and processed before, there is no need to take a lot of effort to make it trainable.


<a name="step3"></a>

## &emsp; 3. Third and Final Step: Modeling

In this step we train the convolutional neural network model, which we have built.

# Resources

The data we have used in this project was downloaded from <a href = "https://www.kaggle.com/datasets/msambare/fer2013" style="text-decoration:none;" target="_blank"> **Kaggle** </a>.</br>
There are more useful information about this dataset in the notebook.</br>
You can also access the notebook uploaded on kaggle website through <a href = "https://www.kaggle.com/code/mamishere/customer-churn-complete-analysis-and-prediction/notebook" style="text-decoration:none;" target="_blank"> **This Link** </a>. 