# Imports
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

FINAL_CHECK_POINT = "../mnist_convnet_model/model.ckpt-20000.meta"
CHECK_POINTS_DIR = "../mnist_convnet_model"
CHECK_POINT_INSPECTION = "H:\\KE4102-Sam\\code\\mnist_convnet_model\\model.ckpt-20000"

mnist = None

def predict_digit(test_sample_number):
    print("Will try to predict: " + str(test_sample_number))
    global mnist
    if mnist is None:
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    eval_data = mnist.test.images  # Returns np.array

    image_to_predict = eval_data[test_sample_number, :]
    #print(image_to_predict)

    #with tf.Session() as sess:
        #saver = tf.train.import_meta_graph(FINAL_CHECK_POINT)
        #saver.restore(sess, tf.train.latest_checkpoint(CHECK_POINTS_DIR))
        #chkp.print_tensors_in_checkpoint_file(CHECK_POINT_INSPECTION, tensor_name='', all_tensors=True, all_tensor_names=True)
        #print(sess.run(tf.all_variables()))
        #feed_dict = {"x": image_to_predict}
        #predictions = sess.run(tf.all_variables(), feed_dict)
        #print(predictions)
    return 11

if __name__ == "__main__":
    predict_digit(0)