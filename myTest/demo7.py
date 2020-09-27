import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#添加层

def add_layer(input, in_size, out_size, activation_function = None):
    #随机值
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

"""
np.linspace是生成要给等差数列，-1到1之间，样本数量是300
后面的[]，np.newaxis写在','之前是变为列扩展的二维数组
写在','之后时，变为行扩展的二维数组
"""
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
#隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#输出层
prediction = add_layer(l1, 10, 1, activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1])) #reduction_indices是
#学习率0.1 减小误差
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
#初始化所有的参数
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#训练
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#
plt.show()
for i in range(100000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])  # 删除第一条线
        except Exception:
            pass
        prediction_val = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)
        plt.pause(0.1)

