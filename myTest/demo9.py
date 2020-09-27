import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
一层神经网络识别MNIST手写数字
"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义网络
def add_layer(input, in_size, out_size, activation_function=None):
    # 随机值
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
    Wx_plus_b = tf.add(tf.matmul(input, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 定义placeholder
# None表示不规定样本的大小，784则是单个样本的大小
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# 添加一层网络
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
#反向传播
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#初始化
sess = tf.Session()
sess.run(tf.initialize_all_variables())


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

#训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
