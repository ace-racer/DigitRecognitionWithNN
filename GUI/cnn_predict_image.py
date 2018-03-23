# Imports
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

FINAL_CHECK_POINT = "../mnist_convnet_model/model.ckpt-20000.meta"
CHECK_POINTS_DIR = "../mnist_convnet_model"

def predict_digit(test_sample_number):
    print("Will try to predict: " + str(test_sample_number))
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    eval_data = mnist.test.images  # Returns np.array
    print("Number of test records: " + str(len(eval_data)))
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(FINAL_CHECK_POINT)
        saver.restore(sess, tf.train.latest_checkpoint(CHECK_POINTS_DIR))
