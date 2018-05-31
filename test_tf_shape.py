import tensorflow as tf
 
input_x  = tf.placeholder(tf.float32, shape=[None, 4])
shape_x = tf.shape(input_x)
var = tf.get_variable("var", shape=[shape_x[1], 3], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
y = tf.matmul(input_x, var)
shape = tf.shape(y)
y_reshaped = tf.reshape(y, [shape[1]*shape[0], -1])
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y_reshaped, feed_dict={input_x: [[1, 2, 3, 4], [5, 6, 7, 8]]}))
