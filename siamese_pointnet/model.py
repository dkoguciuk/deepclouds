import tensorflow as tf


class Model(object):
    """
    A MLP based deep Siamese network for pointcloud classification.
    """

    POINTCLOUD_SIZE = 2048
    """
    Number of points in the pointcloud.
    """

    MODEL_NAME = "MLP_basic"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    def __init__(self, layers_sizes, batch_size, learning_rate,
                 initialization_method, hidden_activation, output_activation,
                 margin, normalize_embedding=True):
        """
        Build a model.
        Args:
            layers_sizes (list of ints): List of hidden units of mlp, but without the first input layer which is POINTCLOUD_SIZE.
            batch_size (int): Batch size of SGD.
            learning_rate (float): Learning rate of SGD.
            initialization_method (str): Weights initialization method: xavier and hu are supported for now.
            hidden_activation (str): Activation method of hidden layers, supported methods:
                        relu, sigmoid, tanh
            output_activation (str): Activation method of last layer, supported layers:
                        relu, sigmoid, sigmoid_my
            margin (float): Learning margin.
        """

        # Placeholders for input clouds
        self.input_a = tf.placeholder(tf.float32, [self.POINTCLOUD_SIZE * 3, batch_size], name="input_a")
        self.input_p = tf.placeholder(tf.float32, [self.POINTCLOUD_SIZE * 3, batch_size], name="input_p")
        self.input_n = tf.placeholder(tf.float32, [self.POINTCLOUD_SIZE * 3, batch_size], name="input_n")
        
        # Initalize parameters
        self.parameters = {}
        self._initialize_parameters(self.POINTCLOUD_SIZE * 3, layers_sizes, initialization_method)
        
        # Build forward propagation
        with tf.name_scope("anchor_embedding"):
            self.embedding_a = self._forward_propagation(self.input_a, hidden_activation, output_activation)
            if normalize_embedding:
                self.embedding_a = self._normalize_embedding(self.embedding_a) 
            tf.summary.histogram("anchor_embedding", self.embedding_a)
        with tf.name_scope("positive_embedding"):
            self.embedding_p = self._forward_propagation(self.input_p, hidden_activation, output_activation)
            if normalize_embedding:
                self.embedding_p = self._normalize_embedding(self.embedding_p)
            tf.summary.histogram("positive_embedding", self.embedding_p)
        with tf.name_scope("negative_embedding"):
            self.embedding_n = self._forward_propagation(self.input_n, hidden_activation, output_activation)
            if normalize_embedding:
                self.embedding_n = self._normalize_embedding(self.embedding_n)
            tf.summary.histogram("negative_embedding", self.embedding_n)

        # Define Loss function
        with tf.name_scope("batch_training"):
            self.loss = self._tripplet_loss(self.embedding_a, self.embedding_p, self.embedding_n, margin, batch_size)
            tf.summary.scalar("batch_cost", self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Summary merge
        self.summary = tf.summary.merge_all()

    def get_summary(self):
        """
        Get tf summary -- one can use this function and pass it to sess.run method,
        because it would preserve all tf namespaces.

        Returns:
            (tf.summary): tf summary of the model with the whole tf model.
        """
        return self.summary

    def get_loss_function(self):
        """
        Get lost to perform evaluation.

        Returns:
            Loss function.
        """
        return self.loss
        

    @classmethod
    def get_model_name(cls):
        """
        Get name of the model -- each model class would have such method implemented.

        Args:
            (str): Model name of the class.
        """
        return cls.MODEL_NAME

    def _initialize_parameters(self, n_x, layers_shapes, initialization_method):
        """
        Initializes parameters to build a neural network with tensorflow.
        Args:
            n_x (int): size of an input layer
            hidden_shapes (list of int): hidden layers sizes
            method (str): net's weights initialization method: xavier and hu are supported for now 
        """
        # First layer
        if initialization_method == "xavier":
            self.parameters["W1"] = tf.get_variable("W1", [layers_shapes[0], n_x], initializer=tf.contrib.layers.xavier_initializer())
        elif initialization_method == "hu":
            self.parameters["W1"] = tf.Variable(tf.random_normal([layers_shapes[0], n_x]) * tf.sqrt(2.0 / n_x), name="W1")
        else:
            raise ValueError("I don't know this method of net's weights initialization..")
        self.parameters["b1"] = tf.get_variable("b1", [layers_shapes[0], 1], initializer=tf.zeros_initializer())
        tf.summary.histogram("W1", self.parameters["W1"])
        tf.summary.histogram("b1", self.parameters["b1"])
    
        # Other layers
        for idx, _ in enumerate(layers_shapes[1:]):
            leyers_shape_idx = idx + 1
            layers_param_idx = str(idx + 2)
            if initialization_method == "xavier":
                self.parameters["W" + layers_param_idx] = tf.get_variable("W" + layers_param_idx, [layers_shapes[leyers_shape_idx], layers_shapes[leyers_shape_idx - 1]], initializer=tf.contrib.layers.xavier_initializer())
            elif initialization_method == "hu":
                self.parameters["W" + layers_param_idx] = tf.Variable(tf.random_normal([layers_shapes[leyers_shape_idx], layers_shapes[leyers_shape_idx - 1]]) * tf.sqrt(2.0 / layers_shapes[leyers_shape_idx - 1]), name="W" + layers_param_idx)
            self.parameters["b" + layers_param_idx] = tf.get_variable("b" + layers_param_idx, [layers_shapes[leyers_shape_idx], 1], initializer=tf.zeros_initializer())
            tf.summary.histogram("W" + layers_param_idx, self.parameters["W" + layers_param_idx])
            tf.summary.histogram("b" + layers_param_idx, self.parameters["b" + layers_param_idx])

    def _forward_propagation(self, X, hidden_activation, output_activation):
        """
        Implements the forward propagation for the model defined in parameters.
        
        Arguments:
            X (tf.placeholder): input dataset placeholder, of shape (input size, number of examples)
            parameters (dict): python dictionary containing your parameters "WX", "bX",
                        the shapes are given in initialize_parameters
            hidden_activation (str): Activation method of hidden layers, supported methods:
                        relu, sigmoid, tanh
            output_activation (str): Activation method of last layer, supported layers:
                        relu, sigmoid, sigmoid_my
        Returns:
            Y_hat: the output of the last layer (estimation)
        """
        AX = X
        
        # Hidden layers
        for idx in range(1, len(self.parameters) / 2):
            with tf.name_scope("layer_" + str(idx)):
                ZX = tf.add(tf.matmul(self.parameters["W" + str(idx)], AX), self.parameters["b" + str(idx)])
                if hidden_activation == "sigmoid":
                    AX = tf.nn.sigmoid(ZX, name="sigmoid" + str(idx))
                elif hidden_activation == "relu":
                    AX = tf.nn.relu(ZX, name="relu" + str(idx))
                elif hidden_activation == "tanh":
                    AX = 1.7159 * tf.nn.tanh(2 * ZX / 3, name="tanh" + str(idx))  # LeCunn tanh
                else:
                    raise ValueError("I don't know this activation method...")
    
        # Output layer
        idx = len(self.parameters) / 2
        with tf.name_scope("layer_" + str(idx)):
            ZX = tf.add(tf.matmul(self.parameters["W" + str(idx)], AX), self.parameters["b" + str(idx)])
            if output_activation == "sigmoid":
                AX = tf.nn.sigmoid(ZX, name="sigmoid" + str(idx))
            elif output_activation == "relu":
                AX = tf.nn.relu(ZX, name="relu" + str(idx))
    
        return AX

    def _tripplet_loss(self, embedding_a, embedding_p, embedding_n, margin, batch_size):
        """
        Define tripplet loss tensor.

        Args:
            embedding_a (tensor): Output tensor of the anchor cloud.
            embedding_p (tensor): Output tensor of the positive cloud.
            embedding_n (tensor): Output tensor of the negative cloud.
            margin (float): Loss margin.
            batch_size (float): Barch size.

        Returns:
            (tensor): Loss function.
        """ 
        pos_dist = tf.reduce_sum(tf.squared_difference(embedding_a, embedding_p), axis=0)
        neg_dist = tf.reduce_sum(tf.squared_difference(embedding_a, embedding_n), axis=0)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))  # / batch_size

    def _normalize_embedding(self, embedding):
        """
        Normalize embedding of a pointcloud.

        Args:
            embedding (tensor): Embedding tensor of a pointcloud to be normalized.
        Returns:
            (tensor): Normalized embedding tensor of a pointcloud.
        """
        return tf.nn.l2_normalize(embedding, dim=0, epsilon=1e-10, name='embeddings')
