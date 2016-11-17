import tensorflow as tf
import numpy as np
import data_helpers
from tensorflow.contrib import learn
from cnn import CNN
from deep_cnn import DeepCNN


if __name__ == "__main__":
    """
    Load data
    """
    x_text, y = data_helpers.load_data_and_labels()


    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    vocab_size = len(vocab_processor.vocabulary_)
    embedding_size = 128

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]



    # Split train/cv/test set
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

    # create an embedding matrix
    with tf.device('/cpu:0'), tf.variable_scope("embedding"):
        embedding_mtx = tf.get_variable("embedding_mtx", [vocab_size, embedding_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
    """
    Initialize a CNN object
    """
    cnn = CNN(sequence_length=x_train.shape[1],
              num_classes=y_train.shape[1],
              vocab_size=vocab_size,
              embedding_size=128,
              filter_sizes=[1],
              num_filters=embedding_size,
              learning_rate=0.001,
              l2_reg_lambda=3)

    """
    Train
    """
    batch_size = 64
    num_epochs = 10

    # Training loop. For each batch...
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.keep_prob: 0.5
            }
            # print cnn.loss.eval(feed_dict=feed_dict)
            cnn.train_op.run(feed_dict=feed_dict)

        # EVALUATION
        print "~~~~~~~~~~~~"
        print "EVALUATION"
        print "~~~~~~~~~~~~"

        accuracy_list = []
        dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
        for dev_batch in dev_batches:
            x_batch, y_batch = zip(*dev_batch)
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.keep_prob: 1
            }
            accuracy_list.append(cnn.accuracy.eval(feed_dict=feed_dict))
        print np.mean(np.array(accuracy_list))

        # ~~~~~~~~~~~~
        # EVALUATION
        # ~~~~~~~~~~~~
        # 0.730859            Note: can achieve 0.749219 with unigram
    #
    # # """
    # # Initialize a DeepCNN object
    # # """
    # # deep_cnn = DeepCNN(sequence_length=x_train.shape[1],
    # #                    num_classes=y_train.shape[1],
    # #                    vocab_size=len(vocab_processor.vocabulary_),
    # #                    embedding_size=128,
    # #                    num_semantic_channels=128,
    # #                    filter_pairs=[(3, 64), (3, 64), (3, 32)],
    # #                    num_fully_connected=32,
    # #                    learning_rate=0.001)
    # # """
    # # Train
    # # """
    # # batch_size = 64
    # # num_epochs = 10
    # #
    # # # Training loop. For each batch...
    # # with tf.Session() as session:
    # #     # Generate batches
    # #     batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
    # #     session.run(tf.initialize_all_variables())
    # #     for batch in batches:
    # #         x_batch, y_batch = zip(*batch)
    # #         feed_dict = {
    # #             deep_cnn.input_x: x_batch,
    # #             deep_cnn.input_y: y_batch,
    # #             deep_cnn.keep_prob: 0.5
    # #         }
    # #
    # #         # print deep_cnn.loss.eval(feed_dict=feed_dict)
    # #         deep_cnn.train_op.run(feed_dict=feed_dict)
    # #
    # #     # EVALUATION
    # #     print "~~~~~~~~~~~~"
    # #     print "EVALUATION"
    # #     print "~~~~~~~~~~~~"
    # #
    # #     accuracy_list = []
    # #     dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
    # #     for dev_batch in dev_batches:
    # #         x_batch, y_batch = zip(*dev_batch)
    # #         feed_dict = {
    # #             deep_cnn.input_x: x_batch,
    # #             deep_cnn.input_y: y_batch,
    # #             deep_cnn.keep_prob: 1
    # #         }
    # #         accuracy_list.append(deep_cnn.accuracy.eval(feed_dict=feed_dict))
    # #     print np.mean(np.array(accuracy_list))
    # #
    # #     # ~~~~~~~~~~~~
    # #     # EVALUATION
    # #     # ~~~~~~~~~~~~
    # #     # 0.733984
