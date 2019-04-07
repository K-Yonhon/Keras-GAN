
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard


# In[2]:


import sys
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


# In[3]:


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def tf_summary_image(image):
    size = image.get_size_inches()*image.dpi # imageはMatplotlibのFigure形式
    height = int(size[0])
    width = int(size[1])
    channel= 1
    with io.BytesIO() as output:
        image.savefig(output, format="PNG")
        image_string = output.getvalue()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)
    
def write_img(callback, name, image, batch_no):
    tf_image = tf_summary_image(image)
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=tf_image)])
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

