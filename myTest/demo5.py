import tensorflow as tf

"""
placeholder的使用，hold住对应的变量，具体的值有外界输入
输入的值使用feed_dict字典进行设置
"""

#给定一个具体的类型,第二个参数可以规定结构
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    #用feed_dict传入对应的值
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))
