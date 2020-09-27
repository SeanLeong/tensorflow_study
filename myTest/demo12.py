import tensorflow as tf
import numpy as np
#Save to file
#记得定义dtype和shape

"""
W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weight')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.initialize_all_variables()

#定义tensorflow的saver，用于存储变量参数情况
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'my_net/save_net.ckpt')
    print("Save to path:", save_path)
"""

#导入变量
#记得与之前的dtype和shape相同

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weight')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

#不需要init
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'my_net/save_net.ckpt')
    print("weight:", sess.run(W))
    print("biases:", sess.run(b))
