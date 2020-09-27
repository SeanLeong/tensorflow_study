import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
使用RNN做分类问题

"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_class = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_class])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}


def RNN(X, weights, biases):
    # 1.hidden layer for input to cell

    # 将X(128 batch, 28 steps, 28 inputs) ==> (128batch * 28steps, 28 hidden)
    X = tf.reshape(X, [-1, n_inputs])
    # 此时的X_in 的shape:[128*28, 128]
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 此时的X_in 由于原来的[128*28, 128] ==> [128, 28, 128]
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 2.cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # 3.hidden layer for oupt as the final results

    # result根据自己的需求进行return(a,b)
    # a.states[1], lstm cell 的state分为两个部分(c_state, m_state)
    ## result = tf.matmul(states[1], weights['out']) + biases['out']

    # 将tensor拆成一个list [(batch, ouputs)] * steps
    # b. states作为最后的结果
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    result = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return result


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
