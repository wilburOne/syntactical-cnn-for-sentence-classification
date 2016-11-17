import tensorflow as tf
import numpy as np
import data_helpers
from cnn import CNN
from deep_cnn import DeepCNN
from embedding.process_embedding import get_embedding_matrix_and_word_index_dict


if __name__ == "__main__":
    """
    Load embedding
    """
    embed_file_name = 'embedding/glove.6B.300d.txt'
    embedding_mtx, word_index_dict = get_embedding_matrix_and_word_index_dict(embed_file_name, 400000, 300)
    vocab_size, embedding_size = embedding_mtx.shape

    """
    Load data
    """
    x_text, y = data_helpers.load_data_and_labels()
    max_document_length = max([len(x.split(" ")) for x in x_text])

    # ignores unknown words, just assign 0
    for i in range(len(x_text)):
        word_list = x_text[i].split(' ')
        x_text[i] = np.zeros(max_document_length)
        for j in range(len(word_list)):
            word = word_list[j]
            if word in word_index_dict:
                x_text[i][j] = word_index_dict[word]
    x = np.array(x_text)

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
        embedding_mtx = tf.get_variable("embedding_mtx", initializer=tf.constant(embedding_mtx, dtype=tf.float32),
                                        trainable=False)

    """
    Initialize a CNN object
    """
    cnn = CNN(sequence_length=x_train.shape[1],
              num_classes=y_train.shape[1],
              vocab_size=vocab_size,
              embedding_size=embedding_size,
              filter_sizes=[2],
              num_filters=128,
              learning_rate=0.001,
              l2_reg_lambda=2)
    """
    Train
    """
    batch_size = 64
    num_epochs = 200

    # Training loop. For each batch...
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        # early stopping
        max_accuracy = 0
        max_i = 0
        for i in range(num_epochs):
            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, 1)

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.keep_prob: 0.5
                }
                # print cnn.loss.eval(feed_dict=feed_dict)
                cnn.train_op.run(feed_dict=feed_dict)
            # evaluate each 3 epochs
            if i % 3 == 0:
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
                accuracy = np.mean(np.array(accuracy_list))
                print accuracy, i
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_i = i

        print 'final result'
        print max_accuracy, max_i

        # final result
        # 0.795117 24 (note: can achieve 0.800391 135 with just unigram)

    # """
    # Initialize a DeepCNN object
    # """
    # deep_cnn = DeepCNN(sequence_length=x_train.shape[1],
    #                    num_classes=y_train.shape[1],
    #                    vocab_size=vocab_size,
    #                    embedding_size=embedding_size,
    #                    num_semantic_channels=128,
    #                    filter_pairs=[(3, 64), (3, 64), (3, 32)],
    #                    num_fully_connected=32,
    #                    learning_rate=0.001,
    #                    l2_reg_lambda=2)
    # """
    # Train
    # """
    # batch_size = 64
    # num_epochs = 200
    #
    # # Training loop. For each batch...
    # with tf.Session() as session:
    #     session.run(tf.initialize_all_variables())
    #     # early stopping
    #     max_accuracy = 0
    #     max_i = 0
    #     for i in range(num_epochs):
    #         # Generate batches
    #         batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, 1)
    #
    #         for batch in batches:
    #             x_batch, y_batch = zip(*batch)
    #             feed_dict = {
    #                 deep_cnn.input_x: x_batch,
    #                 deep_cnn.input_y: y_batch,
    #                 deep_cnn.keep_prob: 0.5
    #             }
    #             deep_cnn.train_op.run(feed_dict=feed_dict)
    #         # evaluate each 3 epochs
    #         if i % 3 == 0:
    #             accuracy_list = []
    #             dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
    #             for dev_batch in dev_batches:
    #                 x_batch, y_batch = zip(*dev_batch)
    #                 feed_dict = {
    #                     deep_cnn.input_x: x_batch,
    #                     deep_cnn.input_y: y_batch,
    #                     deep_cnn.keep_prob: 1
    #                 }
    #                 accuracy_list.append(deep_cnn.accuracy.eval(feed_dict=feed_dict))
    #             accuracy = np.mean(np.array(accuracy_list))
    #             print accuracy, i
    #             if accuracy > max_accuracy:
    #                 max_accuracy = accuracy
    #                 max_i = i
    #
    #     print 'final result'
    #     print max_accuracy, max_i
    #
    #     # final result: 0.803906 12
