# **Facial Emotion Classifier**

![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)

## Project Overview

This project is based on the tutorial by **Nicholas Renotte** on YouTube, where we create a deep learning model capable of accurately categorizing facial expressions into seven distinct classes:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

We leverage the power of **Convolutional Neural Networks (CNNs)**, which have proven to be highly effective in image classification tasks. Our approach involves training a CNN model from scratch using the FER-2013 dataset.

<img src="Al-Pacino.jpeg" alt="Sample Image" width="400">

## Data Description

The FER-2013 dataset consists of grayscale images of faces, each measuring 48x48 pixels. These images have undergone preprocessing to ensure consistent alignment and similar spatial occupation across all samples. The dataset comprises a total of **28,709** images in the training set and **7,178** images in the public test set.

## Main Process

### Steps

1. [**First Step: Data Exploration and Organization**](#step1)
2. [**Second Step: Data Preprocessing**](#step2)
3. [**Third Step: Model Building and Training**](#step3)

---

<a name="step1"></a>
## 1. First Step: Data Exploration and Organization

In the first step, we explore the images and their specifications to get familiar with the project. We begin by loading the necessary libraries and displaying some sample images from the dataset.

### Packages Used

- **TensorFlow**
- **OpenCV**
- **os, random, time, pickle**
- **NumPy**
- **Pandas**
- **Matplotlib**

---

<a name="step2"></a>
## 2. Second Step: Data Preprocessing

In this step, we prepare the dataset for modeling. Given that the dataset is well-curated, only minimal preprocessing is needed to make it trainable. This includes scaling and splitting the data into training and testing sets.

---

<a name="step3"></a>
## 3. Third Step: Model Building and Training

Here, we define the architecture of the Convolutional Neural Network (CNN) model and train it using the preprocessed dataset. We also evaluate the model’s performance using various metrics and visualize its accuracy and loss over the training process.

---

## Resources

The data used in this project was sourced from [**Kaggle**](https://www.kaggle.com/datasets/msambare/fer2013). The project is inspired by a YouTube tutorial from [**Nicholas Renotte**](https://www.youtube.com/c/NicholasRenotte). For more detailed information, refer to the notebook available on [**Kaggle**](https://www.kaggle.com/code/mamishere/customer-churn-complete-analysis-and-prediction/notebook).

