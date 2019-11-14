import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# function used for normalizing our data using min max
def min_max_scaler(data):
    numerator = data-np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator/denominator+1e-7

# creating the data path
data_path = os.getcwd()
data_path = data_path[0: data_path.rfind('/')] + '/data/heart.csv'

# loading the heart data and removing the titles
heart_data = np.loadtxt(data_path, delimiter=',', dtype=np.float32, skiprows=1)
normalize_list = [0, 3, 4, 7, 9] # columns to normalize
for num in normalize_list:
    heart_data[:, num] = min_max_scaler(heart_data[:, num]) # normalization using min max scaler
np.random.shuffle(heart_data)
print(heart_data)
heart_data_train = heart_data[0 : int(len(heart_data) * 0.7)]
heart_data_test = heart_data[int(len(heart_data) * 0.7) : ]
print('__________________', len(heart_data_train))
learning_rate = 0.01
training_epochs = 1000
batch_size = 200
total_batch = len(heart_data)/batch_size
print(total_batch)

X = tf.placeholder(tf.float32, [None, 13])
Y = tf.placeholder(tf.float32, [None, 2])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[13, 7], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([7]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[7, 7], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([7]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[7, 7], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([7]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[7, 7], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([7]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# W5 = tf.get_variable("W5", shape=[7, 7], initializer=tf.contrib.layers.xavier_initializer())
# b5 = tf.Variable(tf.random_normal([7]))
# L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
# L5 = tf.nn.dropout(L5, keep_prob=keep_prob) 

W6 = tf.get_variable("W6", shape=[7, 2], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L4, W6) + b6

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0
for epoch in range(training_epochs):
	avg_cost = 0

	for j in range(int(total_batch)):
		data = heart_data_train[100 * j : 100 * (j+1)]
		#print(data)
		dataX = data[: , 0 : -1]
		dataY = (data[:, [-1]])
		one_hot_y = []
		for y in dataY:
			if y == 1:
				one_hot_y.append([0, 1])
			else:
				one_hot_y.append([1, 0])
		feed_dict = {X: dataX, Y: one_hot_y, keep_prob: 0.7}
		c, _ , h= sess.run([cost, optimizer, hypothesis], feed_dict=feed_dict)
		avg_cost += c/total_batch
	print('Epoch:', '%04d' % (epoch + 1), 'cost=','{:.9f}'.format(avg_cost))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dataX = heart_data_test[:, 0 : -1]
dataY = heart_data_test[:, [-1]]
one_hot_y = []
for y in dataY:
	if y == 1:
		one_hot_y.append([0, 1])
	else:
		one_hot_y.append([1, 0])
print('Accuracy:', sess.run(accuracy, feed_dict = {X:dataX, Y: one_hot_y, keep_prob: 1 }))

np.random.shuffle(heart_data)
for i in range(10):
	test = heart_data[i : i+1, :]
	xtest = test[:, 0 : -1]

	print("label", test[:, [-1]], end = '')
	print(" prediction", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: xtest, keep_prob: 1}))