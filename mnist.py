import tensorflow as tf 
import numpy as np 
import dataset
slim = tf.contrib.slim 

num_epoch = 100
def mnist(learning_rate, initializer_mode, num_conv_layers, num_fc_layers):
    if num_conv_layers not in [1, 2]:
        raise ValueError("num_conv_layers should be 1 or 2")
    if num_fc_layers not in [1, 2]:
        raise ValueError("num_fc_layers should be 1 or 2")

    def make_hyperparameter_string(learning_rate, initializer_mode, num_conv_layers, num_fc_layers):
        hyperparameter = "lr_%.e_" % learning_rate
        if initializer_mode == 0:
            hyperparameter += "xavier_constant"
        else:
            hyperparameter += "truncated_normal_constant"
        hyperparameter += "_%d_conv_%d_fc" % (num_conv_layers, num_fc_layers)
        return hyperparameter

    learning_rate = learning_rate 
    if initializer_mode == 0:
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.1)
    else:
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)
        biases_initializer = tf.constant_initializer(0.1)
    logdir = "logs/mnist/" + make_hyperparameter_string(learning_rate, initializer_mode, num_conv_layers, num_fc_layers)

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)

    def mnist_net(x):
        endpoints = {}
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                                activation_fn=tf.nn.relu,
                                weights_initializer=weights_initializer,
                                biases_initializer=biases_initializer):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding="SAME"):
                net = slim.conv2d(x, 32, [5, 5], scope="conv1")
                net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool1")
                endpoints["block1"] = net
                if num_conv_layers == 2:
                    net = slim.conv2d(net, 64, [5, 5], scope="conv2")
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool2")
                    endpoints["block2"] = net
                    net = tf.reshape(net, shape=[-1, 7*7*64])
                elif num_conv_layers == 1:
                    net = tf.reshape(net, shape=[-1, 14*14*32])
                if num_fc_layers == 1:
                    logits = slim.fully_connected(net, 10, scope="fc")
                else:
                    logits = slim.stack(net, slim.fully_connected, [1024, 10], scope="fc")
                endpoints["logits"] = logits
        return logits, endpoints
    
    # ValueError: Variable conv1/weights already exists, disallowed. 
    # Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope("input"):
            images = tf.placeholder(tf.float32, shape=[None, 784], name="images")
            images_3d = tf.reshape(images, shape=[-1, 28, 28, 1], name="images_3d")
            labels = tf.placeholder(tf.uint8, shape=[None], name="labels")
            onehot_labels = tf.one_hot(indices=labels, depth=10, name="onehot_labels")

        logits, endpoints = mnist_net(images_3d)

        with tf.name_scope("loss"):
            # loss = slim.losses.softmax_cross_entropy(logits, onehot_labels)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
            # 要注释掉这一行，否则会在 tensorboard 中出现两次 softmax_cross_entropy_loss，应该是上面的一行已经加了一次了
            # tf.losses.add_loss(loss) # Letting TF-Slim know about the additional loss. 
            total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
            # tf.add_to_collection('EXTRA_LOSSES', total_loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(labels, tf.cast(tf.argmax(logits, axis=1), tf.uint8))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        with tf.name_scope("optimize"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # create_train_op ensures that each time we ask for the loss, the update_ops
            # are run and the gradients being computed are applied too.
            train_op = slim.learning.create_train_op(total_loss, optimizer)

        # batch size 100 要比 30 好很多，也要稳很多
        train_set = dataset.train("MNIST-data").cache().shuffle(buffer_size=1000).batch(100).repeat(num_epoch)
        test_set = dataset.test("MNIST-data").cache().batch(30).repeat()

        iterator = train_set.make_one_shot_iterator()
        one_element = iterator.get_next()
        iterator_test = test_set.make_one_shot_iterator()
        one_element_test = iterator_test.get_next()

        init_op = tf.global_variables_initializer()
        log_writer  = tf.summary.FileWriter(logdir)

        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries = set()
        for key in endpoints:
            summaries.add(tf.summary.histogram("block/" + key, endpoints[key]))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))
        # for loss in tf.get_collection('EXTRA_LOSSES'):
            # summaries.add(tf.summary.scalar(loss.op.name, loss))
        # summaries.add(tf.summary.scalar("accuracy", accuracy))
        accuracy_train_summary_op = tf.summary.scalar("accuracy_train", accuracy)
        accuracy_test_summary_op = tf.summary.scalar("accuracy_test", accuracy)
        summaries.add(tf.summary.image("image", images_3d, 4))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        step = 0
        with tf.Session() as sess:
            log_writer.add_graph(sess.graph)
            sess.run(init_op)
            try:
                while True:
                    images_, labels_ = sess.run(one_element)
                    sess.run(train_op, feed_dict={
                            images: images_,
                            labels: labels_
                        })
                    if step % 10 == 0:
                        summary_, accuracy_train_summary = sess.run([summary_op, accuracy_train_summary_op], 
                            feed_dict={
                                images: images_,
                                labels: labels_
                        })
                        images_, labels_ = sess.run(one_element_test)
                        accuracy_test_summary = sess.run(accuracy_test_summary_op, feed_dict={
                            images: images_,
                            labels: labels_
                        })
                        log_writer.add_summary(summary_, step)
                        log_writer.add_summary(accuracy_train_summary, step)
                        log_writer.add_summary(accuracy_test_summary, step)
                    step += 1
            except tf.errors.OutOfRangeError:
                print("Finished")
        log_writer.close()
for learning_rate in [1e-3, 1e-4, 1e-5]:
    for initializer_mode in [0, 1]:
        for num_conv_layers in [1, 2]:
            for num_fc_layers in [1, 2]:
                mnist(learning_rate, initializer_mode, num_conv_layers, num_fc_layers)
