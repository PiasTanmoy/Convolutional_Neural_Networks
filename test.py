# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:31:40 2019

@author: Pias Tanmoy
"""

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("hello")