import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
# import random
# import os
import time

# Reseting the graph
tf.reset_default_graph()

fname = 'dinos.txt'
with open(fname) as f:
    training_data = f.read()
    training_data = training_data.replace('\n', ' ')
    training_data = training_data.lower()

chars = list(set(training_data))

data_size, vocab_size = len(training_data), len(chars)
print("The size of the data is %d and vocab size %d" % (data_size, vocab_size))

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

# no of hidden cells or units in rnn
n_hidden = 256

# hyperparameters
learning_rate = .005
training_iters = 20000
display_step = 2000
time_steps = 7
num_input = 1
batch_size = 512

# placeholders
x = tf.placeholder("float", [batch_size, time_steps, num_input], name='Input_X')  # the size of each input is 1*19
y = tf.placeholder("float", [batch_size, vocab_size], name='y')

# initializing weights and biases
weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), name='weights')}
biases = {'out': tf.Variable(tf.random_normal([vocab_size]), name='biases')}


# defining the RNN function

def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, time_steps])
    x = tf.split(x, time_steps, 1)
    #rnn_cell = rnn.BasicLSTMCell(n_hidden)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
    rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=0.8)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    with tf.name_scope('output_layer'):
        logit = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'], name='add')
    return logit

pred = RNN(x, weights, biases)
prediction = tf.nn.softmax(pred,name='prediction')

print("x : {}".format(x))  # the tensor is name  'x:0'
print("y : {}".format(y))  # the tensor is named 'y:0'
print("pred : {}".format(pred))  # the tensor is named 'add:0'

with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
with tf.name_scope('grads_and_vars'):
    grads_and_vars = optimizer.compute_gradients(cost)
with tf.name_scope('capped_gvs'):
    capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in grads_and_vars]
with tf.name_scope('training_op'):
    training_op = optimizer.apply_gradients(capped_gvs)
tf.summary.scalar('cost', cost)

# Model evaluation
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merge = tf.summary.merge_all()

# Storing the log for tensorboard
logs_path = './logs'

# initializing the variable

init = tf.global_variables_initializer()

start_time = time.time()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    sess.run(init)
    total_loss = 0
    total_acc = 0
    offset = 0
    r1 = offset
    r2 = offset + (batch_size * time_steps)

    for steps in range(1, training_iters + 1):
        if offset > 16200:
            offset = 0
            r1 = offset
            r2 = offset + (batch_size * time_steps)

        # getting the input sequence
        # print("steps :%d offset1: %d range: %d" %(steps,offset,batch_size*time_steps))
        # print(r1,r2,steps)
        X = [char_to_ix[str(training_data[i])] for i in range(r1, r2)]
        X = np.reshape(np.array(X), [batch_size, time_steps, 1])
        batch_size = X.shape[0]
        r1 = r2
        r2 = r2 + (batch_size * time_steps)

        # getting ready the True Y
        zeros_array = np.zeros((batch_size, vocab_size), dtype=float)
        for i in range(batch_size):
            zeros_array[i][char_to_ix[str(training_data[offset + time_steps])]] = 1.0
            offset += time_steps

        Y = np.reshape(zeros_array, [batch_size, -1])

        _, acc, loss, onehot_pred, summary = sess.run([training_op, accuracy, cost, pred, merge],
                                                      feed_dict={x: X, y: Y})
        total_loss += loss
        total_acc += acc
        writer.add_summary(summary, steps)

        if steps % display_step == 0:
            print("Iter= " + str(steps + 1) + ", Average Loss= " + "{:.6f}".format(
                total_loss / display_step) + ", Average Accuracy= " + "{:.2f}%".format(
                (100 * total_acc) / display_step))
            total_loss = 0
            total_acc = 0

    # Pick picking out the model input and output
    x_tensor = sess.graph.get_tensor_by_name("Input_X:0")
    pred_tensor = sess.graph.get_tensor_by_name("output_layer/add:0")

    model_input = build_tensor_info(x_tensor)
    model_output = build_tensor_info(pred_tensor)

    # Creating a model signature for using tf_serving
    signature_definition = signature_def_utils.build_signature_def(inputs={'x_input': model_input},
                                                                   outputs={'y_output': model_output},
                                                                   method_name=signature_constants.PREDICT_METHOD_NAME)

    # saving the model

    builder = saved_model_builder.SavedModelBuilder('./models/lstm/1')
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],strip_default_attrs=True,
                                         signature_def_map={
                                             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition},
                                         legacy_init_op=legacy_init_op)

    builder.save()

    # https://www.scribendi.ai/how-to-train-rnn-models-and-serve-them-in-production/