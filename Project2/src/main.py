import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import random

from os import listdir
from os.path import isfile, join
from scipy import misc

path = os.getcwd()
path = path[0 : path.rfind('/')] + '/data/cell_images/Parasitized'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
path = path + '/'
img = misc.imread(path + onlyfiles[0])
print(img.shape)
img_tf = tf.Variable(img)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
im = sess.run(img_tf)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(im)
fig.add_subplot(1,2,2)
plt.imshow(img)
plt.show()