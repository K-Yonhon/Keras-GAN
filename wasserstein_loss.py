#!/usr/bin/env python
# coding: utf-8

from keras import backend as K
import keras
from keras.activations import softplus
import numpy as np


# Wasserstein GAN
def WassersteinLoss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def GradientPenaltyLoss(y_true, y_pred,
                        averaged_samples,
                        gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, 
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(gradient_l2_norm - 1)
    return K.mean(gradient_penalty)


# Cramer GAN
def CramerGradientPenaltyLoss(y_true, y_pred, y_pred_2,
                              averaged_samples,
                              gradient_penalty_weight):
    sqr = K.square(y_pred)
    sqr_sum = K.sum(sqr, axis=np.arange(1, len(sqr.shape)))
    l2_norm_1 = K.sqrt(sqr_sum)
    sqr_2 = K.square(y_pred_2)
    sqr_sum_2 = K.sum(sqr_2, axis=np.arange(1, len(sqr.shape)))    
    l2_norm_2 = K.sqrt(sqr_sum_2)
    
    y = keras.layers.subtract([l2_norm_1, l2_norm_2])
    gradients = K.gradients(y, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, 
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(gradient_l2_norm - 1)
    return K.mean(gradient_penalty)


def L2Norm(y_true, y_pred):
    sqr = K.square(y_pred)
    sqr_sum = K.sum(sqr, axis=np.arange(1, len(sqr.shape)))
    return K.mean(y_true * K.sqrt(sqr_sum))


def Log_D(y_true, y_pred):
    return K.mean(y_true * K.log(y_pred + K.epsilon()))


def Log_1_minus_D(y_true, y_pred):
    return K.mean(y_true * K.log(1 - y_pred + K.epsilon()))


# DRAGAN
def loss_func_dcgan_dis_real(y_true, y_pred):
    return K.mean(y_true * K.log(1 + K.exp(- y_pred)))


def loss_func_dcgan_dis_fake(y_true, y_pred):
    return K.mean(y_true * K.log(1 + K.exp(y_pred)))


def BCE(y_true, y_pred):
    return K.mean(x - x * y_true + K.log(1 + K.exp(- y_pred)))


def get_fake_tag(dims, threshold):
                  
    prob2 = np.random.rand(34)
    tags = np.zeros((dims)).astype("f")
    tags[:] = -1.0
    tags[np.argmax(prob2[0:13])]=1.0
    tags[27 + np.argmax(prob2[27:])] = 1.0
    prob2[prob2<threshold] = -1.0
    prob2[prob2>=threshold] = 1.0
    
    for i in range(13, 27):
        tags[i] = prob2[i]
            
    return tags


def get_fake_tag_batch(batchsize, dims, threshold):
    tags = np.zeros((batchsize, dims)).astype("f")
    for i in range(batchsize):
        tags[i] = np.asarray(get_fake_tag(dims, threshold))
        
    return tags


'''  
# cf.Which Training Methods for GANs do actually Converge?
# Lock it because they make flicker compared to loss_func_dcgan_dis_real
def CommonVanillaLoss(y_true, y_pred):
    return K.mean(K.log(1 + K.exp(- y_true * y_pred)))

# cf.https://github.com/jjonak09/DRAGAN-keras/blob/master/DRAGAN/dragan_keras.py
# -> Losk them because they don't make converge compared to loss_func_dcgan_dis_real on dragan9 32x32 & make nan also
def loss_real(y_true, y_pred):
    return K.mean(softplus(- y_pred))


def loss_fake(y_true, y_pred):
    return K.mean(softplus(y_pred))


def loss_gen(y_true, y_pred):
    return K.mean(-softplus(y_pred))
''' 