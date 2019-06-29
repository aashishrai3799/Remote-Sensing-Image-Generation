# Remote-Sensing-Image-Generation

This is a tensorflow implementation of GAN to generate RS Images.

## Introduction
Remote sensing is the acquisition of information about an object or phenomenon without making physical contact with the object and thus in contrast to on-site observation, especially the Earth. Scene classification of remote sensing images plays a 
role in many remote sensing image applications. Training a good classifier needs a large number of training samples. The labeled samples are often scarce and difficult to obtain. Thus, I implemented Generative Adversarial Networks (GANs) to generate remote sensing images. The network is capable of generating 128x128 RGB images .

## Working:

A random noise vector of size 100 is given to the generator. The generater has 4 transpose convolution layers, each layer is followed by batch normalisation and leaky_relu non-liearity. The output of the 4th layer is again convolved to obtain an image vector of shape 128x128x3. The generated image is then passed to the discriminator network, composed of five convolution layers, each of which is also followed by batch normalisation and leaky_relu non-linearity. The output of the 5th layer is flattened and compressed into a single probability value. Based on this value the model is optimised.

### Platform: Python 3.7

## Libraries:
* tensorflow__
* matplotlib__
* PIL_________
* numpy_______
* glob________
* os
* time
* tqdm
* urllib


#### I trained the model on Tesla T4 GPU and it took around 50 minutes for 100 epochs.
#### On CPU, it took around 25-30 minutes for 1 epoch.
#### NOTE: Contact if you get HTTP error while downloading RSI-CB dataset

## References:
* Model architecture: 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' by Alec   Radford, Luke Metz, Soumith Chintala
* Dataset: RSI-CB (A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data)
* www.tensorflow.org/api_docs/python/tf/layers
* www.towardsdatascience.com
