import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from scipy import misc
from matplotlib.image import imread
from matplotlib import pyplot as plt
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
image_size = 128
len_p = 10000
len_u = 10000
for i in range(len_p):
    img = imread(parasitized_path + parasitized_images[i])
    img = np.resize(img, (128, 128))
    resized_p_images.append([img, 1])
    if i%300 == 0:
        print((i/len_p)* 100)
for i in range(len_u):
    img = imread(uninfected_path + uninfected_images[i])
    img = np.resize(img, (128, 128))
    resized_u_images.append([img, 0])
    if i%300 == 0:
        print((i/len_u)* 100)


learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

class Model:
	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self.build_net()
	def build_net(self):
		self.training = tf.placeholder(tf.bool)
		self.X = tf.placeholder(tf.float32, [None ,image_size, image_size])

		X_img = tf.reshape(self.X, [-1, image_size, image_size, 1])
		self.Y = tf.placeholder(tf.float32, [None, 2])

		conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
		dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

		conv2 = tf.layers.conv2d(inputs=dropout1, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding='SAME', strides=2)
		dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

		conv3 = tf.layers.conv2d(inputs=dropout2, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='SAME', strides=2)
		dropout3= tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

		flat = tf.reshape(dropout3, [-1, 128 * 4 * 4 * 4])
		dense4 = tf.layers.dense(inputs=flat, units=500, activation=tf.nn.relu)
		dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)
		self.logits = tf.layers.dense(inputs=dropout4, units=2)

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
		self.prediction = tf.argmax(self.logits, 1)
		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

	def predict(self, x_test, training=False):
		return self.sess.run(self.prediction, feed_dict={self.X: x_test, self.Y: [[1, 1]], self.training: training})
	def get_accuracy(self, x_test, y_test, training=False):
		return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
	def train(self, x_data,  y_data, training=True):
		return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})
	

sess = tf.Session()
m = Model(sess, "model1")
data = resized_u_images + resized_p_images
np.random.shuffle(data)
train_data = data[0 : int(.7 * len(data))]
test_data = data[int(.7 * len(data)) + 1 : ]
sess.run(tf.global_variables_initializer())
print(len(train_data))
print(len(test_data))
for epoch in range(training_epochs):
	total_batch = int(len(train_data)/batch_size)
	print(total_batch)
	avg_cost = 0
	x_data = []
	y_data = []
	for i in range(total_batch):
		for x in range(total_batch * epoch, total_batch * epoch + total_batch):
			elem = train_data[x]
			x_data.append(elem[0])
			if elem[1] == 0:
				y_data.append([1, 0])
			else:
				y_data.append([0, 1])
	c, _ = m.train(x_data, y_data)
	avg_cost += c / total_batch
	print("Average cost = ", avg_cost, " for Epoch " , + epoch)

x_data = []
y_data = []
for x in test_data:
	x_data.append(x[0])
	if x[1] == 0:
		y_data.append([1, 0])
	else:
		y_data.append([0, 1])
print("Accuracy: ", m.get_accuracy(x_data, y_data))

# x_data = np.array(x_data)
# y_data = np.array(y_data)
# print(x_data.shape)
# print(y_data)

# m.train(x_data, y_data)

# print('Accuracy', m.get_accuracy(x_data, y_data))

# cont = "y"
# i = 0
# while cont == "y":
	
# 	data = resized_u_images[i]
# 	plt.imshow(data[0])
# 	plt.show()
# 	p = m.predict([data[0]])
# 	print(p, 0)

# 	data = resized_p_images[i]
# 	p = m.predict([data[0]])
# 	print(p, 1)
# 	i += 1
	
# 	cont = input("Continue? y or n")
