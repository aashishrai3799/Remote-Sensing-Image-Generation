
# Remote Sensing Image Generation using GAN

##Dataset used: RSI-CB

## Install Libraries
"""

#!pip install pyunpack
#!pip install patool
url = 'https://public.sn.files.1drv.com/y4mSyFO9W2kdbgyFoKgCeqvs2BbUWTQfG3nxXivj5TyekxL88LjUVjx4ME2j86hs2vknRHkRZ4unGzozP_G4CK7WPtguJRt623V-mwo65qtT6j88Mbq0jbWGwv8m0njtA102USQpix1eKxEJRAJH8j3WraMNPJ46rl130HGxO6N1vtOQivHzkU8lgnRjN4rEAB445mX1AkpzjO9HSgrJpfMPg/RSI-CB128.rar?access_token=EwD4Aq1DBAAUcSSzoTJJsy%2bXrnQXgAKO5cj4yc8AAaIjxoxYgl50heU6XF7PWcX9KOFnfMGQJfgM6N6Cee5PvoYW7LHgSY%2fVPNb52eG%2fMPh6ezd0v%2f%2bDsrodLqQbQF0xEmkJkMPn9qOI0CWm2Mrd3sPAGXvrNyR4BdnL6MFr8kuPwOQZtyGRNWe%2fqL5criauSY5H8DExmhR2rVp7G854bxEsguFo5ZDxfcdorN5VarQlRLEWTb%2flXIhL7yE6Q9iYVrhrchNW05sxDvQhdppDMGCBdroWmPt5t9My4DcdSVdHuVeQSjsMLokfyHCYBxR0Yz3geuCL5itc77FU%2b3gzAjiGR91QBObPMPTaMia5MrRtqCEDqBLRVvrRmAhu13UDZgAACBjb%2byxWhx5byAGkoOSWiQ1WOx%2fLPfWSD3JvrNlC4%2ffN%2fuGyhyJPvyPDJfQuvBxN25DBHoZn1cncaQqSAF49U2k5vvpu0dciVxfpAXx1bSxznC3g74EhAJYtNpQ3CK0xXC0DPttAQIkYcLYotat9OgRmn5ksjf7Wr4WI8iJ671bzUvzaSODcjYGZs9pfxaTPzeDx%2bjM%2fbxS1uEdoptz%2fyxtO7%2bKjqX8Wk5sCz%2fzBshQ2fAy%2bjotclR2X2CS1Z1ezN39FOZ6rqZUx%2f%2fb9cBDLpLmPE9prF1l1i1E6%2fLqFciCVtsYTsm%2fGBY%2feelZ5DVknZ88mge1NO3f8iHnJBYexe2w39BuTbYm2jmU31P3wf6lfIgH5MGqLHUwOEAgjFaGi06bBLhYZ5GiE7p2H9GOAt3LFlzGc1mhGFYKA2k6lQALvi0aWF5Zl5IHUQi8ycHxmM9rk4AoJ5tKJz3nD%2bebEomR7lCdMbCi9yUmMwfurfWZTFl6bEIRBBUXAHPbpIXh%2bxyf15G5BZojxKvgEd9sJ2P7Nymjf0pEyG7ltQPWb6VPilXXpr7ScIHuI8MoKUVB%2bq8i%2fdqd7Iry5EtAEbPaFehn9vcV6l%2f2TYt%2bsnRqoBGVAPPAHAg%3d%3d'

"""##Import Libraries"""

import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from urllib.request import urlretrieve
from pyunpack import Archive


# %matplotlib inline

"""##Create Directories"""

path = '/content/'

if not os.path.exists(path + 'rsensing'):
    os.makedirs(path + 'rsensing')
    
if not os.path.exists(path + 'rsensing' + '/data'):
    os.makedirs(path + 'rsensing' + '/data')
    
if not os.path.exists(path + 'rsensing' + '/generated'):
    os.makedirs(path + 'rsensing' + '/generated')
    
INPUT_DATA_DIR = path + "rsensing/data/RSI-CB128/water area/hirst/" # Path to the folder with input images. For more info check simspons_dataset.txt
OUTPUT_DIR = path + 'rsensing/generated/'

"""## Download and Extract Dataset"""

# DOWNLOAD DATASET

data_dir = str(path) + 'rsensing/data/'

if not os.path.exists(data_dir + 'RSI-CB128.rar'):
  
  class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

  with DLProgress(unit='B', unit_scale=True, miniters=1, desc='RSI-CB Data Set') as pbar:
        urlretrieve(str(url), data_dir + 'RSI-CB128.rar', pbar.hook)

    
# EXTRACT DATASET    
if not os.path.exists(str(data_dir) + 'RSI-CB128'):
  Archive(str(data_dir) + 'RSI-CB128.rar').extractall(str(data_dir))

"""### Define Training Parameters"""

IMAGE_SIZE = 128
NOISE_SIZE = 100
BATCH_SIZE = 64
EPOCHS = 300
EPSILON = 0.00005
samples_num = 5

"""## Generator

*   Input: random vector noise of size  100
*   Output: Generated RGB Image of shape 128 X 128 X 3
"""

def generator(z, output_channel_dim, training):
    with tf.variable_scope("generator", reuse= not training):
      
        WEIGHT_INIT_STDDEV = 0.02
        k_init = tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)
        kernel = [5,5]
        strides = [2,2]
        
        
        # 8x8x1024        
        fully_connected = tf.layers.dense(z, 8*8*8*IMAGE_SIZE)
        fully_connected = tf.reshape(fully_connected, (-1, 8, 8, 8*IMAGE_SIZE))
        fully_connected = tf.nn.leaky_relu(fully_connected)

        
        # 8x8x1024 -> 16x16x512
        trans_conv1 = tf.layers.conv2d_transpose(fully_connected, 3*IMAGE_SIZE, kernel, strides, "SAME",
                                                                              kernel_initializer=k_init) 
        batch_trans_conv1 = tf.layers.batch_normalization(trans_conv1, training=training, epsilon=EPSILON)
        
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1)
        
        
        # 16x16x512 -> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(trans_conv1_out,2*IMAGE_SIZE,kernel, strides,"SAME",
                                                                           kernel_initializer=k_init)             
        batch_trans_conv2 = tf.layers.batch_normalization(trans_conv2, training=training, epsilon=EPSILON)        
        
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2)
        
        
        
        # 32x32x256 -> 64x64x128
        trans_conv3 = tf.layers.conv2d_transpose(trans_conv2_out,IMAGE_SIZE,kernel, strides,"SAME",
                                                                         kernel_initializer=k_init)        
        batch_trans_conv3 = tf.layers.batch_normalization(trans_conv3, training=training, epsilon=EPSILON)
        
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3)
        
        
        # 64x64x128 -> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(trans_conv3_out,int(IMAGE_SIZE/2),kernel, strides,"SAME",
                                                                                kernel_initializer=k_init)       
        batch_trans_conv4 = tf.layers.batch_normalization(trans_conv4, training=training, epsilon=EPSILON)
        
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4)
        
        
        # 128x128x64 -> 128x128x3
        logits = tf.layers.conv2d_transpose(trans_conv4_out,3,kernel,[1,1],"SAME",
                                                        kernel_initializer=k_init)
        
        out = tf.tanh(logits, name="out")

        return out

"""##Discriminator

*   Input: 128 X 128 X 3 RGB image
*   Output: It's probability of being real
"""

def discriminator(x, reuse):
    with tf.variable_scope("discriminator", reuse=reuse): 
        
        WEIGHT_INIT_STDDEV = 0.02
        k_init = tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV)
        kernel = [5,5]
        stride = [2,2]
        # 128*128*3 -> 64x64x64 
        
        
        conv1 = tf.layers.conv2d(x,int(IMAGE_SIZE/2),kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm1 = tf.layers.batch_normalization(conv1, training=True, epsilon=EPSILON)        
        conv1_out = tf.nn.leaky_relu(batch_norm1)
        
        
        # 64x64x64-> 32x32x128 
        conv2 = tf.layers.conv2d(conv1_out,IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)       
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True, epsilon=EPSILON)        
        conv2_out = tf.nn.leaky_relu(batch_norm2)
        
        
        # 32x32x128 -> 16x16x256  
        conv3 = tf.layers.conv2d(conv2_out,2*IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True, epsilon=EPSILON)        
        conv3_out = tf.nn.leaky_relu(batch_norm3)
        
        
        # 16x16x256 -> 16x16x512
        conv4 = tf.layers.conv2d(conv3_out,3*IMAGE_SIZE,kernel,[1, 1],"SAME", kernel_initializer=k_init)        
        batch_norm4 = tf.layers.batch_normalization(conv4, training=True, epsilon=EPSILON)        
        conv4_out = tf.nn.leaky_relu(batch_norm4)
        
        
        # 16x16x512 -> 8x8x1024
        conv5 = tf.layers.conv2d(conv4_out,8*IMAGE_SIZE,kernel, stride,"SAME", kernel_initializer=k_init)        
        batch_norm5 = tf.layers.batch_normalization(conv5, training=True, epsilon=EPSILON)        
        conv5_out = tf.nn.leaky_relu(batch_norm5)

        
        flatten = tf.reshape(conv5_out, (-1, 8*8*8*IMAGE_SIZE))
        
        logits = tf.layers.dense(inputs=flatten, units=1, activation=None)
        
        out = tf.sigmoid(logits)
        
        return out, logits

"""## Calculate loss and optimize model"""

def model_loss(input_real, input_z, output_channel_dim):
    
    BETA1 = 0.5

    g_model = generator(input_z, output_channel_dim, True)

    noisy_input_real = input_real + tf.random_normal(shape=tf.shape(input_real), mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1), dtype=tf.float32)
    
    d_model_real, d_logits_real = discriminator(noisy_input_real, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_model_real)*random.uniform(0.9, 1.0)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_model_fake)))
    d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]
    
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=LR_D, beta1=BETA1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=LR_G, beta1=BETA1).minimize(g_loss, var_list=g_vars)  
    return d_loss, g_loss, d_train_opt, g_train_opt

"""## Create Placeholders"""

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate_G = tf.placeholder(tf.float32, name="lr_g")
    learning_rate_D = tf.placeholder(tf.float32, name="lr_d")
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D

"""## Display loss and sample images"""

def display_loss(epoch, time, sess, d_losses, g_losses, input_z, data_shape):
  
    minibatch_size = int(data_shape[0]//BATCH_SIZE)
    
    print("Epoch {}/{}".format(epoch, EPOCHS))
    print("Duration: ", round(time, 5))
    print("D_Loss: ", round(np.mean(d_losses[-minibatch_size:]), 5))
    print("G_Loss: ", round(np.mean(g_losses[-minibatch_size:]), 5))
          
    out_channel_dim = data_shape[3]
    fig, ax = plt.subplots()
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    plt.savefig(OUTPUT_DIR + "losses_" + str(epoch) + ".png")
    plt.show()
    plt.close()
    example_z = np.random.uniform(-1, 1, size=[samples_num, input_z.get_shape().as_list()[-1]])
    samples = sess.run(generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]

    show_samples(sample_images, OUTPUT_DIR + "samples", epoch)

    
def show_samples(sample_images, name, epoch):
  for i in range(5):
    plt.imshow(sample_images[i])
    plt.show()
    img = Image.fromarray(np.uint8((sample_images[i]) * 255))
    img.save(str(OUTPUT_DIR) + str(epoch) + '_' + str(i) + '.png')

  plt.close()

"""## Create Batches"""

def get_batches(data):
    batches = []
    for i in range(int(data.shape[0]//BATCH_SIZE)):
        batch = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        augmented_images = []
        for img in batch:
            image = Image.fromarray(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        normalized_batch = (batch / 127.5) - 1.0
        batches.append(normalized_batch)
    return batches

"""## Train Model"""

def train(get_batches, data_shape, checkpoint_to_load=None):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], NOISE_SIZE)
    d_loss, g_loss, d_opt, g_opt = model_loss(input_images, input_z, data_shape[3])
    
    LR_D = 0.00004
    LR_G = 0.0004
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        iteration = 0
        d_losses = []
        g_losses = []
        
        for epoch in range(EPOCHS):        
            epoch += 1
            t1 = time.time()

            for batch_images in get_batches:
                iteration += 1
                batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, NOISE_SIZE))
                _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: LR_D})
                _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: LR_G})

                d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

            display_loss(epoch, time.time()-t1, sess, d_losses, g_losses, input_z, data_shape)
            
            
            

input_images = np.asarray([np.asarray(Image.open(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in glob(INPUT_DATA_DIR + '*')])
print ("Input: " + str(input_images.shape))

np.random.shuffle(input_images)

sample_images = random.sample(list(input_images), samples_num)
show_samples(sample_images, OUTPUT_DIR + "inputs", 0)

with tf.Graph().as_default():
    train(get_batches(input_images), input_images.shape)
