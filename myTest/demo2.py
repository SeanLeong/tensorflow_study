import tensorflow as tf
import numpy as np

#创建训练的dateset
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.5 + 0.6

#创建参数W,b
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Biases = tf.Variable(tf.zeros([1]))

y = x_data * Weights + Biases
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

#开始训练
sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(Biases), sess.run(loss))
