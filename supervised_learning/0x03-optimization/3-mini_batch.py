#!/usr/bin/env python3
"""Mini-Batch"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains a loaded neural network model using
    mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_acc = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
            step = 0
            for t in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffle[t:t+batch_size]
                Y_batch = Y_shuffle[t:t+batch_size]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step += 1
                if step % 100 == 0:
                    step_cost = sess.run(loss, feed_dict={x: X_batch,
                                                          y: Y_batch})
                    step_accuracy = sess.run(accuracy, feed_dict={x: X_batch,
                                                                  y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
