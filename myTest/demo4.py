import tensorflow as tf


#tf中的变量情况

#定义变量
state = tf.Variable(0, name='counter')
# print(state.name)#counter:0

one = tf.constant(1)
new_value = tf.add(state, one)
#将new_value赋值给state
update = tf.assign(state, new_value)

#初始化所有的变量
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))