import tensorflow as tf


class DeepCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, num_semantic_channels, filter_pairs, num_fully_connected, learning_rate,
                 l2_reg_lambda=0):

        # Add placeholders
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=True):
            embedding_mtx = tf.get_variable("embedding_mtx", [vocab_size, embedding_size])
            self.embedded_chars = tf.nn.embedding_lookup(embedding_mtx, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # member variable to store different hidden states
        self.h_pool = dict()

        # First convolution layer
        with tf.name_scope("conv-1"):
            filter_shape1 = [1, embedding_size, 1, num_semantic_channels]
            W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_semantic_channels]), name="b1")
            conv1 = tf.nn.conv2d(
                self.embedded_chars_expanded,
                W1,
                strides=[1, 1, embedding_size, 1],
                padding="SAME",
                name="conv1")
            # Apply nonlinearity
            h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
            # Note we do not do maxpooling on the first layer
            self.h_pool[1] = h1

        # Add more convolutional layers
        prev_num_filter = num_semantic_channels
        input_height = sequence_length
        i = 2
        for pair in filter_pairs:
            with tf.name_scope("conv-%s" % i):
                filter_size, num_filters = pair
                filter_shape = [filter_size, 1, prev_num_filter, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.h_pool[i-1],
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Apply max_pooling
                self.h_pool[i] = tf.nn.max_pool(
                        h,
                        ksize=[1, filter_size, 1, 1],
                        strides=[1, filter_size, 1, 1],
                        padding='SAME',
                        name="pool")
            prev_num_filter = num_filters
            # note we used zero padding here
            if input_height % filter_size == 0:
                input_height = input_height/filter_size
            else:
                input_height = input_height/filter_size + 1
            i += 1

        self.h_pool_flat = tf.reshape(self.h_pool[i-1], [-1, input_height*prev_num_filter])

        # add fully connected layer
        with tf.name_scope("fully-connected"):
            W_fc = tf.get_variable(
                "W_fc",
                shape=[input_height*prev_num_filter, num_fully_connected],
                initializer=tf.contrib.layers.xavier_initializer())
            b_fc = tf.Variable(tf.constant(0.1, shape=[num_fully_connected]), name="b_fc")
            h_fc = tf.nn.relu(tf.matmul(self.h_pool_flat, W_fc) + b_fc)
            self.h_drop = tf.nn.dropout(h_fc, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_fully_connected, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

