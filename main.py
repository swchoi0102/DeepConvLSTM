import tensorflow as tf
import numpy as np
import cPickle as cp
import sklearn.metrics as metrics
import argparse
import sys

from sliding_window import sliding_window
from time import time, localtime

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 17

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128


def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = cp.load(f)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def convolution(x, num_fileters, kernel_shape, name, training):
    conv = tf.layers.conv2d(x, num_fileters, kernel_size=kernel_shape, activation=None, use_bias=False,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=name)
    bn = tf.layers.batch_normalization(conv, training=training)
    relu = tf.nn.relu(bn)
    return relu


# depthwise separable convolution
def ds_convolution(x, num_fileters, kernel_shape, training):
    depthwise_conv = tf.contrib.layers.separable_conv2d(x,
                                                        num_outputs=None,
                                                        depth_multiplier=1,
                                                        kernel_size=kernel_shape,
                                                        stride=[1, 1],
                                                        padding='VALID',
                                                        activation_fn=None,
                                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                        biases_initializer=None)
    bn = tf.layers.batch_normalization(inputs=depthwise_conv, training=training)
    relu = tf.nn.relu(bn)
    pointwise_conv = tf.layers.conv2d(relu, num_fileters, [1, 1], use_bias=False,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn = tf.layers.batch_normalization(pointwise_conv, training=training)
    relu = tf.nn.relu(bn)
    return relu


def build_graph(x):
    is_training = tf.placeholder(tf.bool, name='is_training')
    conv1 = convolution(x, NUM_FILTERS, [1, FILTER_SIZE], 'conv1/5x1', is_training)
    print(conv1.shape)
    conv2 = convolution(conv1, NUM_FILTERS, [1, FILTER_SIZE], 'conv2/5x1', is_training)
    print(conv2.shape)
    conv3 = convolution(conv2, NUM_FILTERS, [1, FILTER_SIZE], 'conv3/5x1', is_training)
    print(conv3.shape)
    conv4 = convolution(conv3, NUM_FILTERS, [1, FILTER_SIZE], 'conv4/5x1', is_training)
    print(conv4.shape)
    conv4_reshape = tf.reshape(conv4, [-1, 24, 97 * 64])

    lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(NUM_UNITS_LSTM) for _ in range(2)])
    output, state = tf.nn.dynamic_rnn(lstm, conv4_reshape, dtype=tf.float32)
    print("RNN output shape: {}".format(output.shape))

    output = tf.transpose(output, [1, 0, 2])
    last_output = tf.gather(output, int(output.get_shape()[0]) - 1)
    print('RNN last output shape: {}'.format(last_output.shape))

    logits = tf.layers.dense(inputs=last_output, units=NUM_CLASSES, activation=None)

    return logits, is_training


def build_graph2(x):
    is_training = tf.placeholder(tf.bool, name='is_training')
    conv1 = ds_convolution(x, NUM_FILTERS, [1, FILTER_SIZE], is_training)
    conv2 = ds_convolution(conv1, NUM_FILTERS, [1, FILTER_SIZE], is_training)
    conv3 = ds_convolution(conv2, NUM_FILTERS, [1, FILTER_SIZE], is_training)
    conv4 = ds_convolution(conv3, NUM_FILTERS, [1, FILTER_SIZE], is_training)
    conv4_reshape = tf.reshape(conv4, [-1, 24, 97 * 64])

    lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(NUM_UNITS_LSTM) for _ in range(2)])
    output, state = tf.nn.dynamic_rnn(lstm, conv4_reshape, dtype=tf.float32)
    print("RNN output shape: {}".format(output.shape))

    output = tf.transpose(output, [1, 0, 2])
    last_output = tf.gather(output, int(output.get_shape()[0]) - 1)
    print('RNN last output shape: {}'.format(last_output.shape))

    logits = tf.layers.dense(inputs=last_output, units=NUM_CLASSES, activation=None)

    return logits, is_training


def build_graph3(x):
    is_training = tf.placeholder(tf.bool, name='is_training')
    conv1 = tf.layers.separable_conv2d(x, NUM_FILTERS, [1, FILTER_SIZE],
                                       depthwise_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       pointwise_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv2 = tf.layers.separable_conv2d(conv1, NUM_FILTERS, [1, FILTER_SIZE],
                                       depthwise_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       pointwise_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv3 = tf.layers.separable_conv2d(conv2, NUM_FILTERS, [1, FILTER_SIZE],
                                       depthwise_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       pointwise_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv4 = tf.layers.separable_conv2d(conv3, NUM_FILTERS, [1, FILTER_SIZE],
                                       depthwise_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       pointwise_initializer=tf.contrib.layers.variance_scaling_initializer())
    conv4_reshape = tf.reshape(conv4, [-1, 24, 97 * 64])

    lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(NUM_UNITS_LSTM) for _ in range(2)])
    output, state = tf.nn.dynamic_rnn(lstm, conv4_reshape, dtype=tf.float32)
    print("RNN output shape: {}".format(output.shape))

    output = tf.transpose(output, [1, 0, 2])
    last_output = tf.gather(output, int(output.get_shape()[0]) - 1)
    print('RNN last output shape: {}'.format(last_output.shape))

    logits = tf.layers.dense(inputs=last_output, units=NUM_CLASSES, activation=None)

    return logits, is_training


def main(_):
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_dataset('data/oppChallenge_gestures.data')

    assert NB_SENSOR_CHANNELS == x_train.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    x_train, y_train = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    x_test, y_test = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(x_train.shape, y_train.shape))
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(x_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    x_train = x_train.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, 1))
    x_test = x_test.reshape((-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, 1))

    x = tf.placeholder(tf.float32, [None, 24, 113, 1])
    y = tf.placeholder(tf.int32, [None])

    with tf.name_scope('logits'):
        # FIXME
        if FLAGS.mode == 1:
            logits, is_training = build_graph(x)
        elif FLAGS.mode == 2:
            logits, is_training = build_graph2(x)
        else:
            logits, is_training = build_graph3(x)
        prediction = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar('cross_entropy', loss_op)

    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr).minimize(loss_op)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            y_pred = tf.argmax(prediction, 1, output_type=tf.int32)
            correct_pred = tf.equal(y_pred, y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        # get current time
        t = tuple(localtime(time()))[1:5]
        # summary directory path to save summaries
        summary_dir = '/tmp/deepconvlstm/run-%02d%02d-%02d%02d/summary' % t
        ckpt_dir = '/tmp/deepconvlstm/run-%02d%02d-%02d%02d/ckpt' % t

        # training and validation dataset summary writer
        train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
        # validation_writer = tf.summary.FileWriter(summary_dir + '/validation', sess.graph)

        sess.run(init)

        best_f1 = 0.8
        best_epoch = 0

        # step = 0
        # for epoch in range(FLAGS.num_epoch):
        #     for batch in iterate_minibatches(x_train, y_train, FLAGS.batch_size, shuffle=True):
        #         x_batch, y_batch = batch
        #         _, loss, acc, summary = sess.run([train_op, loss_op, accuracy, summary_op],
        #                                          feed_dict={x: x_batch, y: y_batch, is_training: True})
        #         train_writer.add_summary(summary, step)
        #
        #         step += 1
        #         if step % 100 == 0:
        #             print('step: {}, training loss: {}, training accuracy: {}'.format(step, loss, acc))
        #
        #     start_time = time()
        #     test_pred, test_true = list(), list()
        #     for batch in iterate_minibatches(x_test, y_test, FLAGS.batch_size):
        #         x_batch, y_batch = batch
        #         y_pred_ = sess.run([y_pred], feed_dict={x: x_batch, y: y_batch, is_training: False})
        #         test_pred.extend(np.squeeze(y_pred_))
        #         test_true.extend(y_batch)
        #     end_time = time()
        #
        #     test_accuracy = metrics.accuracy_score(test_true, test_pred)
        #     test_f1 = metrics.f1_score(test_true, test_pred, average='weighted')
        #     print('epoch: {}, accuracy: {}, f1-score: {}, elpased time: {}'
        #         .format(epoch + 1, test_accuracy, test_f1, end_time - start_time))
        #
        #     if test_f1 > best_f1:
        #         best_f1 = test_f1
        #         best_epoch = epoch + 1
        #         # saver.save(sess, ckpt_dir, global_step=epoch)
        #
        # print("*" * 50)
        # print("best epoch: {}, best f1 score: {}".format(best_epoch, best_f1))

        # measure inference time
        cnt = 0
        start_time = time()
        for batch in iterate_minibatches(x_test, y_test, 1):
            x_batch, y_batch = batch
            sess.run([y_pred], feed_dict={x: x_batch, y: y_batch, is_training: False})
            cnt += 1

            if cnt == 100:
                break
        end_time = time()

        print('Average inference time : {}'.format((end_time - start_time) / cnt))
        print('Average inference per second : {}'.format(cnt / (end_time - start_time)))

        # total_parameters = 0
        # for variable in tf.trainable_variables():
        #     # shape is an array of tf.Dimension
        #     shape = variable.get_shape()
        #     variable_parameters = 1
        #     for dim in shape:
        #         variable_parameters *= dim.value
        #     total_parameters += variable_parameters
        # print('total number of parameters: {}'.format(total_parameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--mode', type=int, default=3, help='mode 1, 2, or 3')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
