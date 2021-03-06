import tensorflow as tf
import numpy as np
"""
显示tensorBorder
"""

def add_layer(input, in_size, out_size, n_layer, activation_function = None):
    #随机值
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            #在histogram记录
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('Weights'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
            # 在histogram记录
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(input, Weights), biases)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

#隐藏层
l1 = add_layer(xs, 1, 10, n_layer= 1, activation_function=tf.nn.relu)
#输出层
prediction = add_layer(l1, 10, 1, n_layer= 2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    #学习率0.1 减小误差
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
#初始化所有的参数
init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
# tensorboard --logdir logs