
import numpy as np
import math

def normalize(array):
    return (array - array.mean()) / array.std()

# Generate some house sizes between 1000 and 3500 (typical sq ft of houses)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=90, high=325, size=num_house)

# Generate house prices from the sizes with a random noise added
np.random.seed(42)
house_price = house_size * 1000.0 + np.random.randint(low=20000, high=70000, size=num_house)
num_train_samples = math.floor(num_house * 0.7)

# Training data (70%)
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])
train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#Test data( 30%)
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

import tensorflow as tf

tf_house_size = tf.placeholder(tf.float32, name='house_size')
tf_price = tf.placeholder(tf.float32, name='price')

tf_size_factor = tf.Variable(np.random.randn(), name='size_factor')
tf_price_offset = tf.Variable(np.random.randn(), name='price_offset')

tf_price_pred = tf_size_factor * tf_house_size + tf_price_offset

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    num_training_iter = 100
    for iteration in range(num_training_iter):
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print('Trained cost=', training_cost, 'size_factor=', sess.run(tf_size_factor), 'price_offset=', sess.run(tf_price_offset), '\n')

    def normalize_single_value(value, array):
        return (value - array.mean()) / array.std()

    def denormalize(value, array):
        return value * array.std() + array.mean()

    for (size, price) in zip(test_house_size, test_house_price):
        value = normalize_single_value(size, house_size)
        price_prediction = sess.run(tf_price_pred, feed_dict={tf_house_size:value})
        price_prediction = denormalize(price_prediction, house_price )
        print("House size:",size, " Original price:", price, " Price Prediction:", price_prediction, "Diff:", (price_prediction - price) )
