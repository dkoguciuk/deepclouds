import os
import sys
import time
import numpy as np
import tensorflow as tf
import deepclouds.defines as df
import deepclouds.tf_util as tf_util
from deepclouds.my_lstm_cell import MyLSTMCell


class GenericModel(object):
    """
    A generic model of deepclouds network for pointcloud classification.
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

    def get_embeddings(self):
        """
        Get embeddings to be run with a batch of a single pointclouds to find hard triplets to train. 
        """
        return self.cloud_embedding_embdg

    @classmethod
    def get_model_name(cls):
        """
        Get name of the model -- each model class would have such method implemented.

        Args:
            (str): Model name of the class.
        """
        return cls.MODEL_NAME

    @staticmethod
    def save_model(session, model_name):
        """
        Save the model in the model dir.

        Args:
            session (tf.Session): Session which one want to save model.
        """
        saver = tf.train.Saver()
        name = model_name + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + ".ckpt"
        return saver.save(session, os.path.join("models_feature_extractor", name)) 

    def _triplet_loss(self, embedding_a, embedding_p, embedding_n):
        """
        Define tripplet loss tensor.

        Args:
            embedding_a (np.ndaray of shape [B, E]): Output tensor of the anchor cloud.
            embedding_p (np.ndaray of shape [B, E]): Output tensor of the positive cloud.
            embedding_n (np.ndaray of shape [B, E]): Output tensor of the negative cloud.
            margin (float): Loss margin.
        Returns:
            (tensor): Loss function.
        """ 
        with tf.name_scope("triplet_loss"):
            with tf.name_scope("dist_pos"):
                self.pos_dist = tf.reduce_sum(tf.square(embedding_a - embedding_p), axis=-1)
            with tf.name_scope("dist_neg"):
                self.neg_dist = tf.reduce_sum(tf.square(embedding_a - embedding_n), axis=-1)
            with tf.name_scope("copute_loss"):
                self.basic_loss = tf.maximum(self.margin + self.pos_dist - self.neg_dist, 0.0)
                self.non_zero_triplets = tf.count_nonzero(self.basic_loss)
                self.summaries.append(tf.summary.scalar('non_zero_triplets', self.non_zero_triplets))
                final_loss = tf.reduce_mean(self.basic_loss)
            return final_loss
        
    def _triplet_cosine_loss(self, embedding_a, embedding_p, embedding_n):
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
                e = tf.stack([embedding_a, embedding_p, embedding_n], axis=1)
                pos_nom = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])), e, dtype=tf.float32)
                pos_den = tf.multiply(tf.norm(embedding_a, axis=-1), tf.norm(embedding_p, axis=-1))
                self.pos_dist = 1 - pos_nom / pos_den
            with tf.name_scope("dist_neg"):
                neg_nom = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0], x[2])), e, dtype=tf.float32)
                neg_den = tf.multiply(tf.norm(embedding_a, axis=-1), tf.norm(embedding_n, axis=-1))
                self.neg_dist = 1 - neg_nom / neg_den
            with tf.name_scope("copute_loss"):
                self.basic_loss = tf.maximum(self.pos_dist + self.margin - self.neg_dist, 0.0)
                self.non_zero_triplets = tf.count_nonzero(self.basic_loss)
                self.summaries.append(tf.summary.scalar('non_zero_triplets', self.non_zero_triplets))
                final_loss = tf.reduce_mean(self.basic_loss)
            return final_loss

    def _normalize_embedding(self, embedding):
        """
        Normalize embedding of a pointcloud.

        Args:
            embedding (tensor): Embedding tensor of a pointcloud to be normalized.
        Returns:
            (tensor): Normalized embedding tensor of a pointcloud.
        """
        return tf.nn.l2_normalize(embedding, dim=-1, epsilon=1e-10, name='embeddings')

class MLPModel(GenericModel):
    """
    A MLP based network for pointcloud classification.
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
            self.loss = self._triplet_loss(self.embedding_a, self.embedding_p, self.embedding_n, margin, batch_size)
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
            self.cloud_embedding_embdg = tf.squeeze(self._calculate_embeddings(self.placeholder_embdg))
        
        # calculate loss & optimizer
        with tf.name_scope("train"):
            self.train_embd = self._calculate_embeddings(self.placeholder_train)
            self.loss = self._calculate_loss(self.train_embd)
            self.optimizer = self._define_optimizer(self.loss)
            self.summaries.append(tf.summary.scalar('loss', self.loss))
        
        # merge summaries and write        
        self.summary = tf.summary.merge(self.summaries)

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
                return tf.nn.l2_normalize(mlp_output, dim=2, epsilon=1e-10, name='embeddings')

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
            return self._triplet_cosine_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], self.margin)
            #return self._triplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], self.margin, self.batch_size)

    def _define_optimizer(self, loss_function):
        """
        Define optimizer operation.
        """
        with tf.name_scope("optimizer"):
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_function)

class DeepCloudsModel(GenericModel):
    """
    Feature extraction model similar to the one described in Order Matters paper.
    """

    MODEL_NAME = "OrderMattersModel"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    CLASSES_COUNT = 40
    """
    How many classes do we have in the modelnet dataset.
    """

    def __init__(self, train, 
                 batch_size, pointcloud_size,
                 read_block_units, process_block_steps,
                 normalize_embedding=True, verbose=True,
                 learning_rate=0.0001, gradient_clip=10.0,
                 t_net=True, read_block_method='birnn', #birnn or pointnet
                 distance='euclidian'):
        """
        Build a model.
        Args:
            train (bool): Are we training or evaluating the model?
            batch_size (int): Batch size of SGD.
            pointcloud_size (int): Number of 3D points in the pointcloud.
            read_block_units (list of ints): List of hidden units -- each number
                is a layer of the read block RNN.
            process_block_steps (int): How many steps should perform process block?
            normalize_embedding (bool): Should I normalize the embedding of a pointcloud to <0,1>.
            margin (float): Learning margin. 
            learning_rate (float): Learning rate of SGD.
            verbose (bool): Should we print additional info?
            distance (str): Distance measure of two embeddings: euclidian or cosine.
        """

        # Save params
        self.train = train
        self.batch_size = batch_size
        self.pointcloud_size = pointcloud_size
        self.read_block_units = read_block_units
        self.process_block_steps = process_block_steps
        self.normalize_embedding = normalize_embedding
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        self.non_zero_triplets = []
        self.summaries = []
        self.distance = distance
        self.t_net = t_net
        self.read_block_method = read_block_method
        
        # Variable decays
        self.global_step = tf.Variable(1, trainable=False, name='global_step')
        DECAY_STEP = 200000
        DECAY_RATE = 0.7
        BN_INIT_DECAY = 0.5
        BN_DECAY_DECAY_RATE = 0.5
        BN_DECAY_DECAY_STEP = float(DECAY_STEP)
        BN_DECAY_CLIP = 0.99
        
        # bn_decay
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, self.global_step*self.batch_size, BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE, staircase=True)
        self.bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        tf.summary.scalar('bn_decay', self.bn_decay)
        
        # Learning rate
        BASE_LEARNING_RATE = learning_rate
        learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, self.global_step*self.batch_size,
                                                   DECAY_STEP, DECAY_RATE, staircase=True)
        self.learning_rate = tf.maximum(learning_rate, 0.00001)
        tf.summary.scalar('learning_rate', self.learning_rate)

        # Placeholders for input clouds - we will interpret numer of points in the cloud as timestep with 3 coords as an input number
        with tf.name_scope("placeholders"):
            self.placeholder_embdg = tf.placeholder(tf.float32, [self.batch_size, 1, self.pointcloud_size, 3], name="input_embedding")
            self.placeholder_is_tr = tf.placeholder(tf.bool, shape=(), name="input_is_training")
        
        if self.train:
            with tf.name_scope("placeholders"):
                self.placeholder_train = tf.placeholder(tf.float32, [self.batch_size, 3, self.pointcloud_size, 3], name="input_training")
                self.margin = tf.placeholder(tf.float32, shape=1, name="input_margin")

        # init params
        with tf.name_scope("params"):
            self._init_params()

        # Get hard triplets
        with tf.name_scope("get_embedding"):
            
            # Placeholder
            self.read_block_input_embd = self.placeholder_embdg
            
            # T net?
            if self.t_net:
                self.read_block_input_embd = self._define_input_transform_net(self.read_block_input_embd, self.placeholder_is_tr)
            
            # Read block
            if self.read_block_method == 'birnn':
                self.memory_vector_embdg = self._define_read_block_birnn(self.read_block_input_embd)
            elif self.read_block_method == 'pointnet':
                self.memory_vector_embdg = self._define_read_block_pointnet(self.read_block_input_embd, self.placeholder_is_tr, self.bn_decay)
            else:
                raise ValueError('Don\'t know this method of read block implementation..')
            
#             # T net
#             if self.t_net and False:
#                 with tf.variable_scope('transform_net2') as sc:
#                     feature_transform = tf.expand_dims(tf.squeeze(self.memory_vector_embdg, axis=1), -1)
#                     transform_2 = feature_transform_net(feature_transform, is_training=self.placeholder_is_tr, bn_decay=None, K=2*read_block_units[0])
#                 self.memory_vector_embdg = tf.matmul(tf.squeeze(feature_transform, axis=-1), transform_2)
#                 self.memory_vector_embdg = tf.expand_dims(self.memory_vector_embdg, 1)

            # Process block
            with tf.name_scope("process_block"):
                #self.cloud_embedding_embdg_pre = tf.squeeze(self._define_process_block(self.memory_vector_embdg), axis=1)
                self.cloud_embedding_embdg = self._define_process_block(self.memory_vector_embdg)
                if self.normalize_embedding:
                    self.cloud_embedding_embdg = tf.nn.l2_normalize(self.cloud_embedding_embdg, axis=-1, epsilon=1e-10)
        
        # Train procedure
        if self.train:
            with tf.name_scope("train"):

                # Placeholder
                self.read_block_input_train = self.placeholder_train
                
                # T net?
                if self.t_net:
                    self.read_block_input_train = self._define_input_transform_net(self.read_block_input_train, self.placeholder_is_tr)

                # Read block
                if self.read_block_method == 'birnn':
                    self.memory_vector_train = self._define_read_block_birnn(self.read_block_input_train)
                elif self.read_block_method == 'pointnet':
                    self.memory_vector_train = self._define_read_block_pointnet(self.read_block_input_train, self.placeholder_is_tr, self.bn_decay)
                else:
                    raise ValueError('Don\'t know this method of read block implementation..')
                
#                 # T net
#                 if self.t_net and False:
#                     with tf.variable_scope('transform_net2') as sc:
#                         parts = tf.unstack(self.memory_vector_train, axis=1)
#                         outs = []
#                         for idx in range(len(parts)):
#                             with tf.variable_scope(str(idx)) as sc:
#                                 placeholder = tf.expand_dims(parts[idx], axis=-2)
#                                 t_net_mat = feature_transform_net(placeholder, is_training=self.placeholder_is_tr, bn_decay=None, K=2*read_block_units[0])
#                                 outs.append(tf.matmul(tf.squeeze(placeholder, axis=-2), t_net_mat))
#                         self.memory_vector_train = tf.stack(outs, axis=1)
    
                # Process block
                with tf.name_scope("process_block"):
                    self.cloud_embedding_train_pre = self._define_process_block(self.memory_vector_train)
                    if self.normalize_embedding:
                        self.cloud_embedding_train = tf.nn.l2_normalize(self.cloud_embedding_train_pre, axis=-1, epsilon=1e-10)

                # Optimizer
                with tf.name_scope("optimizer"):
                    self.loss = self._calculate_loss(self.cloud_embedding_train)
                    self.optimizer = self._define_optimizer(self.loss)
                    self.summaries.append(tf.summary.scalar('loss', self.loss))

            # merge summaries and write
            self.summary = tf.summary.merge(self.summaries)

    def _init_params(self):
        """
        Initialize params for both RNN networks used later.
        """

        if self.verbose:
            sys.stdout.write("Initializing params...")
            sys.stdout.flush()
            
        if self.read_block_method == 'birnn':
            
            # Define read block params
            self.read_block_cells = { 'fw' : [], 'bw' : []}
            self.read_block_states = { 'fw' : [], 'bw' : []}
            for layers in self.read_block_units:
            
                cell_fw = tf.contrib.rnn.LSTMCell(layers)
                cell_bw = tf.contrib.rnn.LSTMCell(layers)

                self.read_block_cells['fw'].append(cell_fw)
                self.read_block_cells['bw'].append(cell_bw)

                # State
                state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
                state_bw = cell_bw.zero_state(self.batch_size, tf.float32)

                self.read_block_states['fw'].append(state_fw)
                self.read_block_states['bw'].append(state_bw)
                
        elif self.read_block_method == 'pointnet':
            
#             self.params_conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=64, kernel_size = [1,3], scope='conv1')
#             self.params_conv_1_bn = tf_util.BatchNormVars(scope='convbc1')
#             self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=64, kernel_size = [1,1], scope='conv2')
#             self.params_conv_2_bn = tf_util.BatchNormVars(scope='convbc2')
#             self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=64, kernel_size = [1,1], scope='conv3')
#             self.params_conv_3_bn = tf_util.BatchNormVars(scope='convbc3')
#             self.params_conv_4 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='conv4')
#             self.params_conv_4_bn = tf_util.BatchNormVars(scope='convbc4')
#             self.params_conv_5 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=1024, kernel_size = [1,1], scope='conv5')
#             self.params_conv_5_bn = tf_util.BatchNormVars(scope='convbc5')
            self.params_conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=8, kernel_size = [1,3], scope='conv1')
            self.params_conv_1_bn = tf_util.BatchNormVars(scope='convbc1')
            self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=32, kernel_size = [1,1], scope='conv2')
            self.params_conv_2_bn = tf_util.BatchNormVars(scope='convbc2')
            self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=32, num_out_channels=128, kernel_size = [1,1], scope='conv3')
            self.params_conv_3_bn = tf_util.BatchNormVars(scope='convbc3')
            self.params_conv_4 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=256, kernel_size = [1,1], scope='conv4')
            self.params_conv_4_bn = tf_util.BatchNormVars(scope='convbc4')
            self.params_conv_5 = tf_util.Conv2DVars(num_in_channels=256, num_out_channels=1024, kernel_size = [1,1], scope='conv5')
            self.params_conv_5_bn = tf_util.BatchNormVars(scope='convbc5')

        # Define process block params
        self.process_block_cells = []
        self.process_block_state_starts = []
        for layer_idx in range(len(self.process_block_steps)):
            self.process_block_cells.append(MyLSTMCell(num_units = self.read_block_units[-1]*4,
                                                       num_out = self.read_block_units[-1]*2, name = 'process_layer_' + str(layer_idx)))
            self.process_block_state_starts.append(self.process_block_cells[-1].zero_state(self.batch_size, tf.float32))

        # Define input t-net-1 params
        if self.t_net:
            self.params_t1conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=64, kernel_size = [1,3], scope='t1con1')
            self.params_t1conv_1_bn = tf_util.BatchNormVars(scope='t1convbc1')
            self.params_t1conv_2 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='t1con2')
            self.params_t1conv_2_bn = tf_util.BatchNormVars(scope='t1convbc2')
            self.params_t1conv_3 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=1024, kernel_size = [1,1], scope='t1con3')
            self.params_t1conv_3_bn = tf_util.BatchNormVars(scope='t1convbc3')
            self.params_t1fc1 = tf_util.FullyConnVars(num_inputs=1024, num_outputs=512, scope='t1fc1')
            self.params_t1fc1_bn = tf_util.BatchNormVars(scope='t1fc1bn')
            self.params_t1fc2 = tf_util.FullyConnVars(num_inputs=512, num_outputs=256, scope='t1fc2')
            self.params_t1fc2_bn = tf_util.BatchNormVars(scope='t1fc2bn')

            self.params_t1xyz_weights = tf.get_variable('weights', [256, 9], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            self.params_t1xyz_biases = tf.get_variable('biases', initializer=np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32))

        if self.verbose:
            print "OK!"

    def _define_input_transform_net_inner(self, input, is_training, bn_decay=None):
        batch_size = input.get_shape()[0].value
        num_point = input.get_shape()[1].value

        input_image = tf.expand_dims(input, -1)
        net = tf_util.conv2d(input_image, self.params_t1conv_1, self.params_t1conv_1_bn,
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t1conv_2, self.params_t1conv_2_bn,
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t1conv_3, self.params_t1conv_3_bn,
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point,1],
                                 padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, self.params_t1fc1, self.params_t1fc1_bn,
                                      bn=True, is_training=is_training,
                                      bn_decay=bn_decay)
        net = tf_util.fully_connected(net, self.params_t1fc2, self.params_t1fc2_bn,
                                      bn=True, is_training=is_training,
                                      bn_decay=bn_decay)

        with tf.variable_scope('transform_XYZ') as sc:
            transform = tf.matmul(net, self.params_t1xyz_weights)
            transform = tf.nn.bias_add(transform, self.params_t1xyz_biases)

        transform = tf.reshape(transform, [batch_size, 3, 3])
        return transform

    def _define_input_transform_net(self, input, is_training, bn_decay=None, scope='transform_net_1'):

        with tf.name_scope(scope):
            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:
                
                if self.verbose:
                    sys.stdout.write("Defining transform net 1...")
                    sys.stdout.flush()
                    
                transform_1 = self._define_input_transform_net_inner(inputs[0], is_training=is_training, bn_decay=bn_decay)
                out = tf.matmul(inputs[0], transform_1)

                if self.verbose:
                    print "OK!"

                ret = tf.stack([out], axis=1)

            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining transform net 1 for anchor...")
                    sys.stdout.flush()

                transform_1 = self._define_input_transform_net_inner(inputs[0], is_training=is_training, bn_decay=bn_decay)
                anr = tf.matmul(inputs[0], transform_1)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining transform net 1 for positive...")
                    sys.stdout.flush()

                transform_1 = self._define_input_transform_net_inner(inputs[1], is_training=is_training, bn_decay=bn_decay)
                pos = tf.matmul(inputs[1], transform_1)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining transform net 1 for negative...")
                    sys.stdout.flush()

                transform_1 = self._define_input_transform_net_inner(inputs[2], is_training=is_training, bn_decay=bn_decay)
                neg = tf.matmul(inputs[2], transform_1)

                if self.verbose:
                    print "OK!"

                ret = tf.stack([anr, pos, neg], axis=1)

            return ret

    def _define_read_block_birnn(self, input, scope='read_block'):
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
        with tf.name_scope(scope):
            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:
                
                if self.verbose:
                    sys.stdout.write("Defining read block for embedding...")
                    sys.stdout.flush()

                out, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.read_block_cells['fw'], self.read_block_cells['bw'],
                                                                           initial_states_fw=self.read_block_states['fw'],
                                                                           initial_states_bw=self.read_block_states['bw'],
                                                                           inputs=inputs[0], dtype=tf.float32,
                                                                           scope="rnn")

                if self.verbose:
                    print "OK!"

                ret = tf.stack([out], axis=1)
                
            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining read block for anchor...")
                    sys.stdout.flush()

                anr, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.read_block_cells['fw'], self.read_block_cells['bw'],
                                                                           initial_states_fw=self.read_block_states['fw'],
                                                                           initial_states_bw=self.read_block_states['bw'],
                                                                           inputs=inputs[0], dtype=tf.float32,
                                                                           scope="anchor")

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for positive...")
                    sys.stdout.flush()

                pos, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.read_block_cells['fw'], self.read_block_cells['bw'],
                                                                           initial_states_fw=self.read_block_states['fw'],
                                                                           initial_states_bw=self.read_block_states['bw'],
                                                                           inputs=inputs[1], dtype=tf.float32,
                                                                           scope="positive")

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for negative...")
                    sys.stdout.flush()

                neg, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.read_block_cells['fw'], self.read_block_cells['bw'],
                                                                           initial_states_fw=self.read_block_states['fw'],
                                                                           initial_states_bw=self.read_block_states['bw'],
                                                                           inputs=inputs[2], dtype=tf.float32,
                                                                           scope="negative")

                if self.verbose:
                    print "OK!"

                ret = tf.stack([anr, pos, neg], axis=1)
            else:
                raise ValueError("I cannot handle this!")

            # return
            return ret

    def _define_read_block_pointnet_inner(self, input, is_training, bn_decay=None):
        
        batch_size = input.get_shape()[0].value
        num_point = input.get_shape()[1].value
        input_image = tf.expand_dims(input, -1)
        net = tf_util.conv2d(input_image, self.params_conv_1, self.params_conv_1_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_2, self.params_conv_2_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
#         #HERE FEATURE TRANSFORM?
        net = tf_util.conv2d(net, self.params_conv_3, self.params_conv_3_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_4, self.params_conv_4_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_5, self.params_conv_5_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf.squeeze(net, axis=-2)
        return net
        
    def _define_read_block_pointnet(self, input, is_training, bn_decay=None, scope='read_block'):

        with tf.name_scope(scope):
            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:
                
                if self.verbose:
                    sys.stdout.write("Defining read block for embedding [pointnet]...")
                    sys.stdout.flush()

                out = self._define_read_block_pointnet_inner(inputs[0], is_training, bn_decay)

                if self.verbose:
                    print "OK!"

                ret = tf.stack([out], axis=1)
                
            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining read block for anchor...")
                    sys.stdout.flush()

                anr = self._define_read_block_pointnet_inner(inputs[0], is_training, bn_decay)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for positive...")
                    sys.stdout.flush()

                pos = self._define_read_block_pointnet_inner(inputs[1], is_training, bn_decay)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for negative...")
                    sys.stdout.flush()

                neg = self._define_read_block_pointnet_inner(inputs[2], is_training, bn_decay)

                if self.verbose:
                    print "OK!"

                ret = tf.stack([anr, pos, neg], axis=1)
            else:
                raise ValueError("I cannot handle this!")

            # return
            return ret

    def _define_process_block_step(self, process_block_cell, M, process_block_state):

        if self.verbose:
            sys.stdout.write('.')
            sys.stdout.flush()

        # Equation 3
        out, (c, q) = tf.nn.static_rnn(cell = process_block_cell,
                                       inputs = [tf.zeros([self.batch_size, 0])],
                                       initial_state=process_block_state)

        # Equation 4
        m = tf.unstack(M, axis=0)
        q = tf.unstack(q, axis=0)
        e = []
        for i in range(len(m)):
            e.append(tf.einsum('ij,j', m[i], q[i]))
        e = tf.stack(e, axis=0)

        # Equation 5
        a = tf.exp(e) / tf.reduce_sum(tf.exp(e), axis=1, keepdims=True)

        # Equation 6
        a_sum = tf.reduce_sum(a, axis=1, keepdims=True)
        r = tf.reduce_mean(M * tf.expand_dims(a, -1), axis=1) / a_sum

        # Equation 7
        state = tf.contrib.rnn.LSTMStateTuple(c, tf.concat([q, r], axis=1))

        # Return last hidden state
        return out, state

    def _define_process_block_layer(self, layer_no, M, process_block_state):
        
        # Process block
        outs = []
        out, state = self._define_process_block_step(self.process_block_cells[layer_no], M, process_block_state)
        outs.append(out)
        
        for _ in range(1, self.process_block_steps[layer_no]):
            out, state = self._define_process_block_step(self.process_block_cells[layer_no], M, process_block_state)
            outs.append(out)

        return tf.squeeze(tf.stack(outs, axis=2), axis=0), state

    def _define_process_block_inner(self, input):
        
        # first layer
        states = []
        out, state = self._define_process_block_layer(0, input, self.process_block_state_starts[0])
        states.append(state)
        
        # all other layes
        for layer_idx in range(1, len(self.process_block_steps)):
            out, state = self._define_process_block_layer(layer_idx, out, self.process_block_state_starts[layer_idx])
            states.append(state)

        # Return
        ret = tf.stack([state[1] for state in states], axis=1)
        ret = tf.reshape(ret, (ret.shape[0], 1, -1))
        return ret  

    def _define_process_block(self, input):
        """
        Calculate forward propagation of a RNN.

        Args:
            input (np.ndaray of shape [B, X, N, D]): embedding of each cloud, where B: batch_size,
                X: 1 (one pointcloud) or 3 (triplet of pointclouds), N: number of points in each
                pointcloud, D: number of features for every point.
        Returns:
            (np.ndaray of shape [B, X, R]): embedding of each cloud, where B: batch_size,
                X: 1 (one pointcloud) or 3 (triplet of pointclouds), R: number of features for
                every point cloud
        """

        with tf.name_scope("rnn"):

            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:

                if self.verbose:
                    sys.stdout.write("Defining process block for embedding")
                    sys.stdout.flush()

                ret = self._define_process_block_inner(inputs[0])

                if self.verbose:
                    print "OK!"         

            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining process block for anchor")
                    sys.stdout.flush()

                anchor_ret = self._define_process_block_inner(inputs[0])

                if self.verbose:
                    sys.stdout.write("OK!\nDefining process block for positive")
                    sys.stdout.flush()

                positive_ret = self._define_process_block_inner(inputs[1])

                if self.verbose:
                    sys.stdout.write("OK!\nDefining process block for negative")
                    sys.stdout.flush()

                negative_ret = self._define_process_block_inner(inputs[2])

                if self.verbose:
                    print "OK!"

                # Return
                ret = tf.concat([anchor_ret, positive_ret, negative_ret], axis=1)
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

        if self.verbose:
            sys.stdout.write("Defining loss function...")
            sys.stdout.flush()

        with tf.name_scope("loss"):
            embeddings_list = tf.unstack(embeddings, axis=1)
            if self.distance == 'cosine':
                ret = self._triplet_cosine_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2])
            elif self.distance == 'euclidian':
                ret = self._triplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2])
            else:
               raise ValueError("I don't know this embeddings distance..") 

        if self.verbose:
            print "OK!"
        return ret

    def _define_optimizer(self, loss_function):
        """
        Define optimizer operation.
        """

        with tf.name_scope("optimizer"):
            # Compute grads
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(loss_function)
            # Clip
            if self.gradient_clip > 0:
                grads, vars = zip(*grads_and_vars)
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.gradient_clip)
                grads_and_vars = zip(clipped_grads, vars)
            # Summaryssssssssss
            for grad, var in grads_and_vars:
                if grad == None:
                    print("NULL GRADS: ", var.name)
                    #exit()
                else:
                    self.summaries.append(tf.summary.scalar(var.name, tf.norm(grad)))
            return optimizer.apply_gradients(grads_and_vars, self.global_step)
