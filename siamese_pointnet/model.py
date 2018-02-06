import os
import time
import numpy as np
import tensorflow as tf
import siamese_pointnet.defines as df

class GenericModel(object):
    """
    A generic model of deep Siamese network for pointcloud classification.
    """
    def __init__(self):
        """
        This is pure virtual class - one cannot make an instance of it.
        """
        raise NotImplementedError()
    
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

    def get_optimizer(self):
        """
        Get optimizer to perform learning.

        Returns:
            Optimizer.
        """
        return self.optimizer

    @classmethod
    def get_model_name(cls):
        """
        Get name of the model -- each model class would have such method implemented.

        Args:
            (str): Model name of the class.
        """
        return cls.MODEL_NAME

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
        with tf.name_scope("triplet_loss"):
            with tf.name_scope("dist_pos"):
                pos_dist = tf.reduce_sum(tf.square(embedding_a - embedding_p), axis=1)
            with tf.name_scope("dist_neg"):
                neg_dist = tf.reduce_sum(tf.square(embedding_a - embedding_n), axis=1)
            with tf.name_scope("copute_loss"):
                basic_loss = tf.maximum(margin + pos_dist - neg_dist, 0.0)
                final_loss = tf.reduce_mean(basic_loss)
            return final_loss
        
    def _tripplet_cosine_loss(self, embedding_a, embedding_p, embedding_n, margin):
        """
        Define tripplet loss tensor.

        Args:
            embedding_a (tensor of shape [B, E]): Embedding tensor of the anchor cloud of size
            B: batch_size, E: embedding vector size.
            embedding_p (tensor of shape [B, E]): Embedding tensor of the positive cloud of size
            B: batch_size, E: embedding vector size.
            embedding_n (tensor of shape [B, E]): Embedding tensor of the negative cloud of size
            B: batch_size, E: embedding vector size.
            margin (float): Loss margin.

        Returns:
            (tensor): Loss function.
        """ 
        with tf.name_scope("triplet_loss"):
            with tf.name_scope("dist_pos"):
                pos_dist = tf.losses.cosine_distance(embedding_a, embedding_p, axis=1)
            with tf.name_scope("dist_neg"):
                neg_dist = tf.losses.cosine_distance(embedding_a, embedding_n, axis=1)
            with tf.name_scope("copute_loss"):
                basic_loss = tf.maximum(margin + pos_dist - neg_dist, 0.0)
                final_loss = tf.reduce_mean(basic_loss)
            return final_loss

    def _normalize_embedding(self, embedding):
        """
        Normalize embedding of a pointcloud.

        Args:
            embedding (tensor): Embedding tensor of a pointcloud to be normalized.
        Returns:
            (tensor): Normalized embedding tensor of a pointcloud.
        """
        return tf.nn.l2_normalize(embedding, axis=1, epsilon=1e-10, name='embeddings')

class MLPModel(GenericModel):
    """
    A MLP based deep Siamese network for pointcloud classification.
    """

    MODEL_NAME = "MLP_basic"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    def __init__(self, layers_sizes, batch_size, learning_rate,
                 initialization_method, hidden_activation, output_activation,
                 margin, normalize_embedding=True, pointcloud_size=2048):
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
            normalize_embedding (bool): Should I normalize the embedding of a pointcloud to [0,1].
            pointcloud_size (int): Number of 3D points in the pointcloud.
        """
        raise NotImplemented("For a long time this class was not tested!")

        # Placeholders for input clouds
        self.input_a = tf.placeholder(tf.float32, [batch_size, pointcloud_size * 3], name="input_a")
        self.input_p = tf.placeholder(tf.float32, [batch_size, pointcloud_size * 3], name="input_p")
        self.input_n = tf.placeholder(tf.float32, [batch_size, pointcloud_size * 3], name="input_n")
        
        # Initalize parameters
        self.parameters = {}
        self._initialize_parameters(pointcloud_size * 3, layers_sizes, initialization_method)
        
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
            self.parameters["W1"] = tf.get_variable("W1", [n_x, layers_shapes[0]], initializer=tf.contrib.layers.xavier_initializer())
        elif initialization_method == "hu":
            self.parameters["W1"] = tf.Variable(tf.random_normal([n_x, layers_shapes[0]]) * tf.sqrt(2.0 / n_x), name="W1")
        else:
            raise ValueError("I don't know this method of net's weights initialization..")
        self.parameters["b1"] = tf.get_variable("b1", [layers_shapes[0]], initializer=tf.zeros_initializer())
        tf.summary.histogram("W1", self.parameters["W1"])
        tf.summary.histogram("b1", self.parameters["b1"])
    
        # Other layers
        for idx, _ in enumerate(layers_shapes[1:]):
            leyers_shape_idx = idx + 1
            layers_param_idx = str(idx + 2)
            if initialization_method == "xavier":
                self.parameters["W" + layers_param_idx] = tf.get_variable("W" + layers_param_idx, [layers_shapes[leyers_shape_idx - 1], layers_shapes[leyers_shape_idx]], initializer=tf.contrib.layers.xavier_initializer())
            elif initialization_method == "hu":
                self.parameters["W" + layers_param_idx] = tf.Variable(tf.random_normal([layers_shapes[leyers_shape_idx - 1], layers_shapes[leyers_shape_idx]]) * tf.sqrt(2.0 / layers_shapes[leyers_shape_idx - 1]), name="W" + layers_param_idx)
            self.parameters["b" + layers_param_idx] = tf.get_variable("b" + layers_param_idx, [layers_shapes[leyers_shape_idx]], initializer=tf.zeros_initializer())
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
                ZX = tf.add(tf.matmul(AX, self.parameters["W" + str(idx)]), self.parameters["b" + str(idx)])
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
            ZX = tf.add(tf.matmul(AX, self.parameters["W" + str(idx)]), self.parameters["b" + str(idx)])
            if output_activation == "sigmoid":
                AX = tf.nn.sigmoid(ZX, name="sigmoid" + str(idx))
            elif output_activation == "relu":
                AX = tf.nn.relu(ZX, name="relu" + str(idx))
    
        return AX

class RNNBidirectionalModel(GenericModel):
    """
    A bidirectional RNN model with LSTM cell with triple loss function for pointcloud classification.
    """

    MODEL_NAME = "RNN_bidirectional"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    CLASSES_COUNT = 40
    """
    How many classes do we have in the modelnet dataset.
    """

    def __init__(self, rnn_layers_sizes, mlp_layers_sizes,
                 batch_size, learning_rate, margin, normalize_embedding=True, pointcloud_size=16):
        """
        Build a model.
        Args:
            rnn_layers_sizes (list of ints): List of hidden units of rnn.
            mlp_layers_sizes (list of ints): List of hidden units of mlp built on top of rnn -- first
                val of mlp_layers_sizes must match 2*rnn_last_value*pointcloud_size
            batch_size (int): Batch size of SGD.
            learning_rate (float): Learning rate of SGD.
            margin (float): Learning margin.
            normalize_embedding (bool): Should I normalize the embedding of a pointcloud to [0,1].
            pointcloud_size (int): Number of 3D points in the pointcloud.
        """
        if mlp_layers_sizes[0] != 2*rnn_layers_sizes[-1]*pointcloud_size:
            raise ValueError("first val of mlp_layers_sizes must match 2*rnn_last_value*pointcloud_size")
        
        # Save params
        self.batch_size = batch_size
        self.rnn_layers_sizes = rnn_layers_sizes
        self.mlp_layers_sizes = mlp_layers_sizes
        self.pointcloud_size = pointcloud_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.normalize_embedding = normalize_embedding
        self.summaries = []
        
        # Placeholders for input clouds - we will interpret numer of points in the cloud as timestep with 3 coords as an input number
        with tf.name_scope("placeholders"):
            self.placeholder_embdg = tf.placeholder(tf.float32, [self.batch_size, 1, self.pointcloud_size, 3], name="input_embedding")
            self.placeholder_train = tf.placeholder(tf.float32, [self.batch_size, 3, self.pointcloud_size, 3], name="input_training")

        # init RNN and MLP params
        with tf.name_scope("params"):
            self._init_params()
        
        # calculate embedding
        with tf.name_scope("find_hard_triplets"):
            self.embeddings = tf.squeeze(self._calculate_embeddings(self.placeholder_embdg))
        
        # calculate loss & optimizer
        with tf.name_scope("train"):
            self.train_embd = self._calculate_embeddings(self.placeholder_train)
            self.loss = self._calculate_loss(self.train_embd)
            self.optimizer = self._define_optimizer(self.loss)
            self.summaries.append(tf.summary.scalar('loss', self.loss))
        
        # merge summaries and write        
        self.summary = tf.summary.merge(self.summaries)

    def get_embeddings(self):
        """
        Get embeddings to be run with a batch of a single pointclouds to find hard triplets to train. 
        """
        return self.embeddings

    def save_model(self, session):
        """
        Save the model in the model dir.

        Args:
            session (tf.Session): Session which one want to save model.
        """
        saver = tf.train.Saver()
        name = self.MODEL_NAME + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + ".ckpt"
        return saver.save(session, os.path.join("models_feature_extractor", name))      

    def _init_params(self):
        """
        Initialize params for RNN and MLP networks used later.
        """
        
        # Define RNN params
        self.rnn_cells = { 'fw' : [], 'bw' : []}
        self.rnn_states = { 'fw' : [], 'bw' : []}
        for layers in self.rnn_layers_sizes:

            # Layer
            cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(layers, forget_bias=1.0)
            cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(layers, forget_bias=1.0)
            
            self.rnn_cells['fw'].append(cell_fw)
            self.rnn_cells['bw'].append(cell_bw)
            
            # State
            state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            
            self.rnn_states['fw'].append(state_fw)
            self.rnn_states['bw'].append(state_bw)
        
        # Define MLP params
        self.mlp_params = {}
        for layer_idx in range(1, len(self.mlp_layers_sizes)):
            self.mlp_params['W' + str(layer_idx)] = tf.get_variable('W' + str(layer_idx), 
                                                                [self.mlp_layers_sizes[layer_idx-1], self.mlp_layers_sizes[layer_idx]], 
                                                                initializer = tf.contrib.layers.xavier_initializer(), )
            self.mlp_params["b" + str(layer_idx)] = tf.get_variable("b" + str(layer_idx),
                                                                [self.mlp_layers_sizes[layer_idx]],
                                                                initializer = tf.zeros_initializer())

    def _calculate_embeddings(self, input):
        """
        Calculate embeddings for clouds stored in the input variable. This method would be called
        for both embeddings calculation for futher hard triples finding and for final training step
        so one can pass here either np.ndarray of shape [batch_size, 1, N, 3] or [batch_size, 3, N, 3].

        Args:
            input (np.ndarray of shape [B, X, N, 3]): input pointclouds, where B: batch size, X: 1
                (one pointcloud) or 3 (triplet of pointclouds), N: number of points in each pointcloud.
        Returns:
            (np.ndaray of shape [B, X, E]): embedding of each cloud, where B: batch_size, X: 1 (one
                pointcloud) or 3 (triplet of pointclouds), E: size of an embedding of a pointcloud.
        """

        with tf.name_scope("calculate_embeddings"):

            # rnn forward prop
            rnn_output = self._forward_rnn(input)
            
#             # bare rnn output
#             inputs = tf.unstack(rnn_output, axis=1)
#             for input_idx in range(len(inputs)):
#                 inputs[input_idx] = tf.reshape(inputs[input_idx], [self.batch_size, -1])
#             ret = tf.stack(inputs, axis=1)
#             return ret
            
            # mlp forward prop
            mlp_output = self._forward_mlp(rnn_output)
            
            # normalize embedding
            if self.normalize_embedding:
                return tf.nn.l2_normalize(mlp_output, axis=2, epsilon=1e-10, name='embeddings')

            return mlp_output

    def _forward_rnn(self, input):
        """
        Calculate forward propagation of a RNN.

        Args:
            input (np.ndarray of shape [B, X, N, 3]): input pointclouds, where B: batch size, X: 1
                (one pointcloud) or 3 (triplet of pointclouds), N: number of points in each pointcloud.
        Returns:
            (np.ndaray of shape [B, X, N, R]): embedding of each cloud, where B: batch_size, X: 1 (one
                pointcloud) or 3 (triplet of pointclouds), N: number of points in each pointcloud,
                R: number of features for every point.  
        """

        with tf.name_scope("rnn"):

            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None
            
            if len(inputs) == 1:
                out, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cells['fw'], self.rnn_cells['bw'],
                                                                           initial_states_fw=self.rnn_states['fw'],
                                                                           initial_states_bw=self.rnn_states['bw'],
                                                                           inputs=inputs[0], dtype=tf.float32,
                                                                           scope="rnn")
                ret = tf.stack([out], axis=1)
            elif len(inputs) == 3:
                anr, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cells['fw'], self.rnn_cells['bw'],
                                                                           initial_states_fw=self.rnn_states['fw'],
                                                                           initial_states_bw=self.rnn_states['bw'],
                                                                           inputs=inputs[0], dtype=tf.float32,
                                                                           scope="anchor")
                pos, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cells['fw'], self.rnn_cells['bw'],
                                                                           initial_states_fw=self.rnn_states['fw'],
                                                                           initial_states_bw=self.rnn_states['bw'],
                                                                           inputs=inputs[1], dtype=tf.float32,
                                                                           scope="positive")
                neg, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.rnn_cells['fw'], self.rnn_cells['bw'],
                                                                           initial_states_fw=self.rnn_states['fw'],
                                                                           initial_states_bw=self.rnn_states['bw'],
                                                                           inputs=inputs[2], dtype=tf.float32,
                                                                           scope="negative")
                ret = tf.stack([anr, pos, neg], axis=1)
            else:
                raise ValueError("I cannot handle this!")
            
            # return
            return ret 

    def _forward_mlp(self, input):
        """
        Calculate forward propagation of a MLP.

        Args:
            input (np.ndaray of shape [B, X, N, R]): embedding of each cloud, where B: batch_size,
                X: 1 (one pointcloud) or 3 (triplet of pointclouds), N: number of points in each
                pointcloud, R: number of features for every point. 
        Returns:
            (np.ndaray of shape [B, X, E]): embedding of each cloud, where B: batch_size, X: 1 (one
                pointcloud) or 3 (triplet of pointclouds), E: size of an embedding of a pointcloud.
        """

        with tf.name_scope("mlp"):

            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None
            
            if len(inputs) == 1:
                AX = tf.reshape(inputs[0], [self.batch_size, -1])
                for layer_idx in range(1, len(self.mlp_layers_sizes)):
                    with tf.name_scope("layer_" + str(layer_idx)):
                        AX = tf.nn.tanh(tf.matmul(AX, self.mlp_params['W' + str(layer_idx)]) + self.mlp_params['b' + str(layer_idx)])
                ret = tf.stack([AX], axis=1)
            elif len(inputs) == 3:
                outputs = []
                with tf.name_scope("anchor"):
                    AX = tf.reshape(inputs[0], [self.batch_size, -1])
                    for layer_idx in range(1, len(self.mlp_layers_sizes)):
                        with tf.name_scope("layer_" + str(layer_idx)):
                            AX = tf.nn.tanh(tf.matmul(AX, self.mlp_params['W' + str(layer_idx)]) + self.mlp_params['b' + str(layer_idx)])
                    outputs.append(AX)
                with tf.name_scope("positive"):
                    AX = tf.reshape(inputs[1], [self.batch_size, -1])
                    for layer_idx in range(1, len(self.mlp_layers_sizes)):
                        with tf.name_scope("layer_" + str(layer_idx)):
                            AX = tf.nn.tanh(tf.matmul(AX, self.mlp_params['W' + str(layer_idx)]) + self.mlp_params['b' + str(layer_idx)])
                    outputs.append(AX)
                with tf.name_scope("negative"):
                    AX = tf.reshape(inputs[2], [self.batch_size, -1])
                    for layer_idx in range(1, len(self.mlp_layers_sizes)):
                        with tf.name_scope("layer_" + str(layer_idx)):
                            AX = tf.nn.tanh(tf.matmul(AX, self.mlp_params['W' + str(layer_idx)]) + self.mlp_params['b' + str(layer_idx)])
                    outputs.append(AX)
                ret = tf.stack(outputs, axis=1)
            else:
                raise ValueError("I cannot handle this!")
        
            # return
            return ret 
    
    def _calculate_loss(self, embeddings):
        """
        Calculate loss.

        Args:
            embeddings (np.ndaray of shape [B, 3, E]): embedding of each cloud, where
                B: batch_size, E: size of an embedding of a pointcloud.
        Returns:
            (float): Loss of current batch.
        """
        with tf.name_scope("loss"):
            embeddings_list = tf.unstack(embeddings, axis=1)
            return self._tripplet_cosine_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], self.margin)
            #return self._tripplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], self.margin, self.batch_size)

    def _define_optimizer(self, loss_function):
        """
        Define optimizer operation.
        """
        with tf.name_scope("optimizer"):
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_function)
