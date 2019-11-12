import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import random
import imageio
from os import listdir
from os.path import isfile, join
from scipy import misc

path = os.getcwd()
parasitized_path = path[0 : path.rfind('/')] + '/data/Parasitized'
parasitized_images = [f for f in listdir(parasitized_path) if isfile(join(parasitized_path, f))]
uninfected_path = path[0 : path.rfind('/')] + '/data/Uninfected'
uninfected_images = [f for f in listdir(uninfected_path) if isfile(join(uninfected_path, f))]


parasitized_path += '/'
uninfected_path += '/'
resized_p_images = []
resized_u_images = []
len_p = len(parasitized_images)
len_u = len(uninfected_images)

len_p = 100
len_u = 100
for i in range(len_p):
    img = imageio.imread(parasitized_path + parasitized_images[i])
    img = tf.image.resize_image_with_crop_or_pad(img, 128, 128)
    resized_p_images.append(img)
    if i%300 == 0:
        print((i/len_p)* 100)
for i in range(len_u):
    img = imageio.imread(uninfected_path + uninfected_images[i])
    img = tf.image.resize_image_with_crop_or_pad(img, 128, 128)
    resized_u_images.append(img)
    if i%300 == 0:
        print((i/len_u)* 100)


