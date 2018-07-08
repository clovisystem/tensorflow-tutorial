#
#   house_price_prediction.py
#
#   This is a very simple prediction of house prices based on house size, implemented
#   in Tensorflow. This code is part of Pluralsight's course "Tensorflow: Getting Started"
#
from absl import flags
from absl import app
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/tmp/tf/', '')

# You need to normalize values to prevent under/overflows.
def normalize(array):
    return (array - array.mean()) / array.std()

def generate_initial_plot(house_size, house_price):
    # Plot generated hours and size
    plt.plot(house_size, house_price, "bx") # bx = blue x
    plt.ylabel("Price")
    plt.xlabel("Size")
    plt.show()

def prepare_train_data(house_price, house_size, num_train_samples):
    train_house_size = np.asarray(house_size[:num_train_samples])
    train_price = np.asarray(house_price[:num_train_samples])
    train_house_size_norm = normalize(train_house_size)
    train_price_norm = normalize(train_price)
    return train_house_size_norm, train_price_norm, train_house_size, train_price

def prepare_test_date(house_price, house_size, num_train_samples):
    test_house_size = np.asarray(house_size[num_train_samples:])
    test_house_price = np.asarray(house_price[num_train_samples:])
    return test_house_size, test_house_price

def generate_data():
    # Generate some house sizes between 1000 and 3500 (typical sq ft of houses)
    num_house = 160
    np.random.seed(42)
    house_size = np.random.randint(low=90, high=325, size=num_house)
    # house_size = np.random.randint(low=1000, high=3500, size=num_house)

    # Generate house prices from the sizes with a random noise added
    np.random.seed(42)
    house_price = house_size * 1000.0 + np.random.randint(low=20000, high=70000, size=num_house)
    num_train_samples = math.floor(num_house * 0.7)
    return house_size, house_price, num_train_samples

def generate_final_plot(train_house_size, train_price, test_house_size, test_house_price, train_house_size_norm, train_house_size_std, train_house_size_mean, sess, tf_size_factor, tf_price_offset, train_price_std, train_price_mean):
    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel('Price')
    plt.xlabel('Size (sq.mt)')
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
            (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
            label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()

def generate_animated_plot(train_price_std, train_house_size_mean, train_price_mean, fit_size_factor, train_house_size_norm, train_house_size_std, fit_price_offsets, house_price, house_size, train_house_size, train_price, test_house_size, test_house_price, fit_plot_idx):
    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)
        return line,

    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,

    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.title('Gradient Descent Fitting Regression Line')
    plt.ylabel('Price')
    plt.xlabel('Size (sq.mt)')
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim, interval=1000, blit=True)

    plt.show()

def main(args):

    house_size, house_price, num_train_samples = generate_data()

    # generate_initial_plot(house_size, house_price)

    train_house_size_norm, train_price_norm, train_house_size, train_price = prepare_train_data(house_price, house_size, num_train_samples)
    test_house_size, test_house_price = prepare_test_date(house_price, house_size, num_train_samples)

    # Set up Tensorflow palceholder that get updated as we descend down the gradient
    tf_house_size = tf.placeholder(tf.float32, name='house_size')
    tf_price = tf.placeholder(tf.float32, name='price')

    # Define the variables holding the size_factor and price we set during training.
    # We initialize them to some random values based on the normal distribution.
    tf_size_factor = tf.Variable(np.random.randn(), name='size_factor')
    tf_price_offset = tf.Variable(np.random.randn(), name='price_offset')

    # Define the operations for the predicting values.
    # Notice, the use of the tensorflow add and multiply operations.
    tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)
    tf.summary.histogram("price_pred", tf_price_pred)

    # Define the loss function (how much error) - Mean Squared Error
    tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)
    tf.summary.scalar("loss", tf_cost)

    # Optimizer learning rate. The size of the steps down towards the gradient.
    learning_rate = 0.1

    # Define a Gradient Descent optimizer that will minimize the loss defined in the operation "cost".
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph in the session
    with tf.Session() as sess:
        sess.run(init)

        # Set how often to display the training progress and number of training iterations
        display_every = 2
        num_training_iter = 150

        # Calculate the number of lines to animation
        fit_num_plots = math.floor(num_training_iter / display_every)
        # Add storage of factor and offset values from each epoch
        fit_size_factor = np.zeros(fit_num_plots)
        fit_price_offsets = np.zeros(fit_num_plots)
        fit_plot_idx = 0

        ## Define summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

        # Keep iterating the training data
        for iteration in range(num_training_iter):

            # Fit all training data
            for (x, y) in zip(train_house_size_norm, train_price_norm):
                sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

            # Display current status
            if (iteration + 1) % display_every == 0:
                c, summary = sess.run([tf_cost, merged], feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
                train_writer.add_summary(summary, iteration)

                print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                    "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
                # Save the fit size_factor and price_offfset to allow animation of learning process
                fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
                fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
                fit_plot_idx += 1

        print('Optimization Finished!')
        training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
        print('Trained cost=', training_cost, 'size_factor=', sess.run(tf_size_factor), 'price_offset=', sess.run(tf_price_offset), '\n')

        # Get values sued to normalized data so we can denormalize data back to its original scale
        train_house_size_mean = train_house_size.mean()
        train_house_size_std = train_house_size.std()

        train_price_mean = train_price.mean()
        train_price_std = train_price.std()


        generate_final_plot(train_house_size, train_price, test_house_size,
                            test_house_price, train_house_size_norm, train_house_size_std,
                            train_house_size_mean, sess, tf_size_factor, tf_price_offset,
                            train_price_std, train_price_mean)

        generate_animated_plot(train_price_std, train_house_size_mean, train_price_mean,
                               fit_size_factor, train_house_size_norm, train_house_size_std,
                               fit_price_offsets, house_price, house_size, train_house_size,
                               train_price, test_house_size, test_house_price, fit_plot_idx)

if __name__ == '__main__':
    app.run(main)
