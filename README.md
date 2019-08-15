# DAM-CNN
the demo code of model DAM-CNN

## Brief Introduction
This code is corresponding to the paper: [Xie, Siyue, Haifeng Hu, and Yongbo Wu. "Deep Multi-path Convolutional Neural Network Joint with Salient Region Attention for Facial Expression Recognition." Pattern Recognition (2019)](https://www.sciencedirect.com/science/article/abs/pii/S0031320319301268).

In this code, I just exemplify the training and validation/testing process of DAM-CNN on FER2013 dataset. (Because only FER2013 is open-access and all other datasets mentioned in the paper should be licenced before using them.)

## Enviornment Requirements
* Python 3.4
* TensorFlow 1.2-gpu
* Numpy
* Scipy

## Setup
* I have appended the pre-processed FER2013 in the file 'DB.rar'. Just extract DB.rar in the current directory. You can access to the original FER2013 in https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 
* The model needs to load the pretrained VGG-Face model. The parameters of VGG-Face (in the form of .npy) can be downloaded [here](https://drive.google.com/file/d/1qX8ED0zogJ-s8FwHxIhSKAZbmcwstvKe/view?usp=sharing). Just place it in the current directory. The original pretrained VGG-Face model can be downloaded [here]( http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) 

## Other Information
* In the paper, DAM-CNN is trained following a two-stage strategy. In this code, I have improved it so the model can be trained in an end-to-end manner. The training strategy of this code is a little different with the description in our paper. But the classification results of this code are consistent with the accuracy we reported in the paper.
* The model gets benefit from the code of [MTAE](https://github.com/ghif/mtae).
