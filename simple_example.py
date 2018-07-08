import tensorflow as tf
import numpy as np
from absl import flags
from absl import app
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir','/tmp/tf/', '')


def main(args):

    # training data
    x_train = np.array([100,135,180,200]) #area
    y_train = np.array([200000,300000,320000,400000])  #house price

    # Model parameters
    W = tf.Variable([0.00001], dtype=tf.float32)
    b = tf.Variable([0.00001], dtype=tf.float32)

    linear_model = W * x_train + b

    # loss
    loss = tf.reduce_mean(tf.square(linear_model - y_train)) # sum of the squares
    tf.summary.scalar("loss", loss)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()


    sess = tf.Session()
    sess.run(init) # reset values to wrong

    ## Define summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

    # training loop
    for i in range(10):
        curr_W, curr_b, curr_loss = sess.run([W, b, loss])
        print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
        summary, t = sess.run([merged, train])

        train_writer.add_summary(summary, i)

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss])

    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


app.run(main)
