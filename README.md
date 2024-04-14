# Introduction

This is just my personal notes where I put my research resources. This is basically my journey from a noob.

## Deep Learning

https://github.com/sxudan/artificial-intelligence-journey/blob/main/Introductions/Deep_Learning.ipynb]

## Convolution Neural Network

https://github.com/sxudan/artificial-intelligence-journey/blob/main/Introductions/Convolution_Neural_Network.ipynb

## Image Classification

https://github.com/sxudan/artificial-intelligence-journey/blob/main/Classification/dogvscat.ipynb

## Pattern Prediction

### Model that Generates grayscale image

![image](https://github.com/sxudan/artificial-intelligence-journey/assets/31989781/d03dee6a-1a6f-4372-b077-150684ec1336)

https://github.com/sxudan/artificial-intelligence-journey/blob/main/Transformation/blackandwhite.ipynb

## GAN Architecture

A Generative Adversarial Network (GAN) is a type of artificial intelligence model used in unsupervised machine learning, particularly for generating new data samples from a given distribution.

The basic structure of a GAN involves two neural networks: a generator and a discriminator. Here's how it works:

Generator: This network takes random noise as input and tries to generate data samples that resemble the real data. For example, if you're training a GAN to generate images of cats, the generator network will take random noise vectors as input and output images that ideally look like real cat images.

Discriminator: This network is like a binary classifier. It takes both real data samples and generated data samples as input and tries to distinguish between them. It's trained to output a high probability if the input is real (i.e., from the true data distribution) and a low probability if the input is generated by the generator.

https://github.com/sxudan/artificial-intelligence-journey/tree/main/GAN

## Image Segmentation using NN

Fully Convolutional Networks (FCNs): These networks replace fully connected layers with convolutional layers to enable end-to-end pixel-wise prediction.

U-Net: U-Net is a popular architecture that consists of an encoder-decoder structure with skip connections. Skip connections help preserve spatial information during upsampling.

DeepLab: DeepLab is based on the atrous convolution (also known as dilated convolution) and employs techniques like atrous spatial pyramid pooling for capturing multi-scale context.

![image](https://github.com/sxudan/artificial-intelligence-journey/assets/31989781/18c5be6e-1054-4515-824d-b097627958cc)


https://github.com/sxudan/artificial-intelligence-journey/tree/main/Segmentation
