import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

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

# checking the shape of the heart data
print("Shape of", data_path[data_path.rfind('/')+1:], "-", heart_data.shape, '\n') # (303, 14)

# slicing the data into our input and output
heart_data_x = heart_data[:, 0:-1]
normalize_list = [0, 3, 4, 7, 9] # columns to normalize
for num in normalize_list:
    heart_data_x[:, num] = min_max_scaler(heart_data_x[:, num]) # normalization using min max scaler
heart_data_y = heart_data[:, [-1]]

# storing the shape of the x and y data
x_shape = heart_data_x.shape
y_shape = heart_data_y.shape

# verifying the shape is correct
print ("Shape of x data - ", x_shape) # (303, 13)
print ("Shape of y data - ", y_shape, '\n') # (303, 1)

# number of columns for x and y shape
num_col_x = x_shape[1]
num_col_y = y_shape[1]

'''
Using logistic regression, we must use the formula
H(x) = Wx + b

Since we have multiple data for our x, we must use 
matrix multiplication to get our output. i.e.
    H(x1, x2, ..., xn) = (x1, x2, ..., xn)*(w1, w2, ..., wn) + b

Our hypothesis' output needs to be categorical, 
thus we use the sigmoid function on our hypothesis

We now need to train our model. Using the cost function
    cost = -avg(sum(y * log(hypothesis) + (1-y) * log (1 - hypothesis)))
we train our model using the gradient descent algorithm
'''

x = tf.compat.v1.placeholder(tf.float32, shape=[None, num_col_x]) # input data
y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_col_y]) # categorical output
W = tf.Variable(tf.random.normal([num_col_x, 1], name='weight')) # weights
b = tf.Variable(tf.random.normal([1]), name='bias') # bias

# H(x) = xW + b
hypothesis = tf.sigmoid(tf.matmul(x, W)+b)

# cost function
cost = -1 * tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis))

# minimizing the cost with the gradient descent algorithm
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 1 if H(x) > .5, 0 if H(x) <= .5
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)

# testing the accuracy between the actual output and the predicted output
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=np.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    # training the model
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:heart_data_x, y: heart_data_y})
        sess.run(train, feed_dict={x:heart_data_x, y: heart_data_y})
        if step%200 == 0:
            print("Step:", step, "Cost:", cost_val)
    # accuracy of the model
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:heart_data_x, y: heart_data_y})
    print("Accuracy:", a)