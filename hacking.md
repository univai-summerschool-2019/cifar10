
# Cifar-10 Hacking

This fun exercise is taken almost verbatim from a homework for Sebastian Rashcka's stats 479 course at Madison-Wisconsin

Implement a convolutional neural network for classifying images in the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html). 

### Dataset Overview

- The CIFAR-10 dataset contains 60,000 color images with pixel dimensions 32x32. 
- There are 50,000 training images and 10,000 test images
- Shown below is a snapshot showing a random selection for the 10 different object classes (from https://www.cs.toronto.edu/~kriz/cifar.html):

![](cifar-snapshot.png)



### Your Tasks

Your main task is to implement a simple convolutional neural network that is loosely inspired by the AlexNet architecture that one the ImageNet competition in 2012: 

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) (pp. 1097-1105).

Then, you will make several simple modifications to this network architecture to improve its performance.

Note that in this homework, as explained above, you will NOT be working with ImageNet but CIFAR-10, which is a much smaller dataset, in order for you to be able to train the network in a timely manner.

In particular, you will be asked to first implement a basic convolutional neural network based on AlexNet and then make several improvements to optimize the performance and reduce overfitting. These "improvements" include Dropout, BatchNorm, and image augmentation, which will serve as a good exercise for familiarizing yourself with "Deep Learning Tricks" as well as convolutional neural networks.


## Model Settings

You can use these settings in your code as a start..

```
# Hyperparameters
BATCH_SIZE = 256
NUM_EPOCHS = 20

# Architecture
NUM_FEATURES = 32*32
NUM_CLASSES = 10
```


## 1) Implement a Convolutional Neural Network

In this part, you will be implementing the AlexNet-variant that you will be using and modifying throughout this homework. On purpose, this will be a bit more "hands-off" than usual, so that you get a chance to practice implementing neural networks from scratch based on sketches and short descriptions (which is a useful real-world skill as it is quite common to reimplement architectures from literature in order to verify results and compare those architectures to your own methods).

The architecture is as follows:

![](architecture-1.png)

Note that Sebastian made this network based on AlexNet, but there are some differences. Overall though, there are 7 hidden layers in total: 5 convolutional layers and 2 fully-connected layers. There is one output layers mapping the last layer's activations to the classes. For this network, 

- all hidden layers are connected via ReLU activation functions
- the output layer uses a softmax activation function


## 2) Adding Dropout

In this second part, your task is now to add dropout layers to reduce overfitting. You can copy&paste your architecture from above and make the appropriate modifications. In particular,

- place a Dropout2d (this is also referred to as "spatial dropout") before each maxpooling layer with dropout probability p=0.2,
- place a regular dropout after each fully connected layer with probability p=0.5, except for the last (output) layer.

The architecture is as follows (changes, compared to the previous section, are highlighted in red):

![](architecture-2.png)

## 3) Add BatchNorm

In this 3rd part, you are now going to add BatchNorm layers to further improve the performance of the network. This use BatchNorm2D for the convolutional layers and BatchNorm1D for the fully connected layers.


The architecture is as follows (changes, compared to the previous section, are highlighted in red):

![](architecture-3.png)

## 4) Going All-Convolutional

In this 4th part, your task is to remove all maxpooling layers and replace the fully-connected layers by convolutional layers. Note that the number of elements of the activation tensors in the hidden layers should not change. I.e., when you remove the max-pooling layers, you need to increase the stride of the convolutional layers from 1 to 2 to achieve the same scaling. Furthermore, you can replace a fully-connected conmvolutional layer by a convolutional layer using stride=1 and a kernel with height and width equal to 1.

The new architecture is as follows (changes, compared to the previous section, are highlighted in red):

![](architecture-4.png)

## 5) Add Image Augmentation

In this last section, you should use the architecture from the previous section (section 4) but use additional image augmentation during training to improve the generalization performance. 


In particular, you should modify the training generator so that it

- performs a random horizontal flip with propbability 50%
- resizes the image from 32x32 to 40x40
- performs a 32x32 random crop from the 40x40 images
- normalizes the pixel intensities such that they are within the range [-1, 1]

The `test_transform = transforms.Compose([...` function should be modified accordingly, such that it 

- resizes the image from 32x32 to 40x40
- performs a 32x32 **center** crop from the 40x40 images
- normalizes the pixel intensities such that they are within the range [-1, 1]

## 6) Training the network for 200 epochs

In this optional section, train the network from the previous part for 200 epochs to see how it performs as the training loss converges. This will take about 50 minutes.
