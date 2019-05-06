#!/usr/bin/env python
# coding: utf-8

from keras import backend as K
import numpy as np

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
