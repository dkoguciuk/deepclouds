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

    def _triplet_loss(self, embedding_a, embedding_p, embedding_n, transformation_matrix, regularization_weight):
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
                
                K = transformation_matrix.get_shape()[1].value
                mat_diff = tf.matmul(transformation_matrix, tf.transpose(transformation_matrix, perm=[0,2,1]))
                mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                batch_size = transformation_matrix.get_shape()[0]
                reg_loss = mat_diff_loss * regularization_weight * tf.to_float(self.non_zero_triplets) / tf.to_float(batch_size)
                self.summaries.append(tf.summary.scalar('reg_loss', reg_loss))
                
                #final_loss = tf.reduce_mean(self.basic_loss)
                final_loss = tf.reduce_mean(self.basic_loss) + reg_loss
            return final_loss
        
    def _triplet_cosine_loss(self, embedding_a, embedding_p, embedding_n,
                             transformation_matrix, regularization_weight):#, labels_a, classes_learning_weights):
        """
        Define tripplet loss tensor.

        Args:
            embedding_a (tensor of shape [B, E]): Embedding tensor of the anchor cloud of size
            B: batch_size, E: embedding vector size.

            embedding_p (tensor of shape [B, E]): Embedding tensor of the 
positive cloud of size
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
                self.pos_dist = 1 - pos_nom / (pos_den + 1e-9)
            with tf.name_scope("dist_neg"):
                neg_nom = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0], x[2])), e, dtype=tf.float32)
                neg_den = tf.multiply(tf.norm(embedding_a, axis=-1), tf.norm(embedding_n, axis=-1))
                self.neg_den = neg_den
                self.neg_dist = 1 - neg_nom / (neg_den + 1e-9)
            with tf.name_scope("copute_loss"):
                self.basic_loss = tf.maximum(self.pos_dist + self.margin - self.neg_dist, 0.0)
                #loss_num = tf.gather(classes_learning_weights, labels_a)
                #loss_den = tf.reduce_sum(tf.where(self.basic_loss > 0, loss_num, tf.zeros(loss_num.get_shape())))
                #self.weighted_loss = tf.multiply((loss_num / loss_den), self.basic_loss)

                self.non_zero_triplets = tf.count_nonzero(self.basic_loss)
                self.summaries.append(tf.summary.scalar('non_zero_triplets', self.non_zero_triplets))

                K = transformation_matrix.get_shape()[1].value
                mat_diff = tf.matmul(transformation_matrix, tf.transpose(transformation_matrix, perm=[0,2,1]))
                mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                batch_size = transformation_matrix.get_shape()[0]
                reg_loss = mat_diff_loss * regularization_weight * tf.to_float(self.non_zero_triplets) / tf.to_float(batch_size)
                self.summaries.append(tf.summary.scalar('reg_loss', reg_loss))

                final_loss = tf.reduce_mean(self.basic_loss) + reg_loss
                #final_loss = tf.reduce_mean(self.weighted_loss) + reg_loss
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

class DeepCloudsModel(GenericModel):
    """
    Feature extraction model similar to the one described in Order Matters paper.
    """

    MODEL_NAME = "DeepCloudsModel"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    def __init__(self, train, 
                 batch_size, pointcloud_size,
                 read_block_units, process_block_steps,
                 normalize_embedding=True, verbose=True,
                 learning_rate=0.0001, gradient_clip=10.0,
                 input_t_net=False, feature_t_net=False,
                 read_block_method='birnn', #birnn or pointnet
                 process_block_method='max-pool', #max-pool or attention-rnn
                 distance='cosine', regularization_weight=0.001):
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
        self.input_t_net = input_t_net
        self.feature_t_net = feature_t_net
        self.read_block_method = read_block_method
        self.process_block_method = process_block_method
        self.regularization_weight = regularization_weight
        self.CLASSES_COUNT = batch_size
        
        # Variable decays
        self.global_step = tf.Variable(1, trainable=False, name='global_step')
#         DECAY_STEP = 200000
#         DECAY_RATE = 0.7
#         BN_INIT_DECAY = 0.5
#         BN_DECAY_DECAY_RATE = 0.5
#         BN_DECAY_DECAY_STEP = float(DECAY_STEP)
#         BN_DECAY_CLIP = 0.99
#          
#         # bn_decay
#         bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY, self.global_step*self.batch_size, BN_DECAY_DECAY_STEP,
#                                                  BN_DECAY_DECAY_RATE, staircase=True)
#         self.bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
#         tf.summary.scalar('bn_decay', self.bn_decay)
        
#         # Learning rate
#         BASE_LEARNING_RATE = learning_rate
#         learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, self.global_step*self.batch_size,
#                                                    DECAY_STEP, DECAY_RATE, staircase=True)
#         self.learning_rate = tf.maximum(learning_rate, 0.00001)
#         tf.summary.scalar('learning_rate', self.learning_rate)
        self.learning_rate = learning_rate

        # Placeholders for input clouds - we will interpret numer of points in the cloud as timestep with 3 coords as an input number
        with tf.name_scope("placeholders"):
            self.placeholder_embdg = tf.placeholder(tf.float32, [self.batch_size, 1, self.pointcloud_size, 3], name="input_embedding")
            #self.placeholder_label = tf.placeholder(tf.int32, [self.batch_size], name="input_labels")
            self.placeholder_is_tr = tf.placeholder(tf.bool, shape=(), name="input_is_training")
            self.classes_learning_weights = tf.placeholder(tf.float32, [self.CLASSES_COUNT], name="classes_weights")
        
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
            if self.input_t_net:
                self.read_block_input_embd = self._define_input_transform_net(self.read_block_input_embd, self.placeholder_is_tr, bn_decay=None)
            
            # Read block
            if self.read_block_method == 'birnn':
                self.memory_vector_embdg = self._define_read_block_birnn(self.read_block_input_embd)
            elif self.read_block_method == 'pointnet':
                self.memory_vector_embdg = self._define_read_block_pointnet(self.read_block_input_embd, self.placeholder_is_tr, bn_decay=None)#self.bn_decay)
            else:
                raise ValueError('Don\'t know this method of read block implementation..')

#             # T net
#             if self.feature_t_net:
#                 self.memory_vector_embdg = self._define_feature_transform_net(self.memory_vector_embdg, is_training=self.placeholder_is_tr, bn_decay=None)

            # Process block
            with tf.name_scope("process_block"):
                self.cloud_embedding_embdg = self._define_process_block(self.memory_vector_embdg, self.process_block_method)
                if self.normalize_embedding:
                    self.cloud_embedding_embdg = tf.nn.l2_normalize(self.cloud_embedding_embdg, axis=-1, epsilon=1e-10)
        
        # Train procedure
        if self.train:
            with tf.name_scope("train"):

                # Placeholder
                self.read_block_input_train = self.placeholder_train
                
                # T net?
                if self.input_t_net:
                    self.read_block_input_train = self._define_input_transform_net(self.read_block_input_train, self.placeholder_is_tr)

                # Read block
                if self.read_block_method == 'birnn':
                    self.memory_vector_train = self._define_read_block_birnn(self.read_block_input_train)
                elif self.read_block_method == 'pointnet':
                    self.memory_vector_train = self._define_read_block_pointnet(self.read_block_input_train, self.placeholder_is_tr, bn_decay=None)#self.bn_decay)
                else:
                    raise ValueError('Don\'t know this method of read block implementation..')
                
#                 # T net
#                 if self.feature_t_net:
#                     self.memory_vector_train = self._define_feature_transform_net(self.memory_vector_train, is_training=self.placeholder_is_tr, bn_decay=None)

                # Process block
                with tf.name_scope("process_block"):
                    self.cloud_embedding_train = self._define_process_block(self.memory_vector_train, self.process_block_method)
                    if self.normalize_embedding:
                        self.cloud_embedding_train = tf.nn.l2_normalize(self.cloud_embedding_train, axis=-1, epsilon=1e-10)

                # Optimizer
                with tf.name_scope("optimizer"):
                    self.loss = self._calculate_loss(self.cloud_embedding_train)
                    self.optimizer = self._define_optimizer(self.loss)
                    self.summaries.append(tf.summary.scalar('loss', self.loss))

            # merge summaries and write
            self.summary = tf.summary.merge(self.summaries)

    def find_semihard_triples(self, embeddings, labels):
        """
        embeddings (tf.tensor of size [B, 1, E])
        labels (tf.tensor of size [B]) 
        """
        raise ValueError("This method is not implemented yet!")
    
        i0 = tf.constant(0)
        d0 = tf.zeros([1, 1])
        
        while_condition = lambda i, d: tf.less(i, embeddings.get_shape()[0])
        
        def body(i, d):

            # Calc distances
            how_many = tf.constant([self.batch_size])
            dupa = tf.reshape(tf.tile(embeddings[i, 0], how_many), [self.batch_size, -1])
            cipa = tf.stack([embeddings[:, 0], dupa], axis=1)
            nom = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1])), cipa, dtype=tf.float32)
            den = tf.multiply(tf.norm(embeddings[:, 0], axis=-1), tf.norm(dupa, axis=-1))
            dist = 1 - nom / den

            # Find hardes positive            
            class_idxs = tf.where(labels == labels[i])          # All indexes within the same class
            class_dist = tf.gather(dist, class_idxs)            # All distances within the same class
            class_dmax = tf.argmax(class_dist)                  # Index of the biggest distance in the class_dist/class_idx 
            posit_hard = tf.gather(class_idxs, class_dmax)      # Index of the biggest distance in the batch

            return [tf.add(i, 1), tf.concat([d, d], axis=0)]
        
        # do the loop:
        r = tf.while_loop(while_condition, body, [i0, d0], shape_invariants=[i0.get_shape(), tf.TensorShape([None, 1])])

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
                
#         elif self.read_block_method == 'pointnet':
#             self.params_conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=8, kernel_size = [1,3], scope='conv1')
#             #self.params_conv_1_bn = tf_util.BatchNormVars(scope='convbc1')
#             #self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=32, kernel_size = [1,1], scope='conv2')
#             self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=32, kernel_size = [1,1], scope='conv2')
#             #self.params_conv_2_bn = tf_util.BatchNormVars(scope='convbc2')
#             #self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=32, num_out_channels=128, kernel_size = [1,1], scope='conv3')
#             self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=32, num_out_channels=64, kernel_size = [1,1], scope='conv3')
#             #self.params_conv_3_bn = tf_util.BatchNormVars(scope='convbc3')
#             self.params_conv_4 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='conv4')
#             #self.params_conv_4_bn = tf_util.BatchNormVars(scope='convbc4')
#             self.params_conv_5 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=self.read_block_units[-1]*2, kernel_size = [1,1], scope='conv5')
#             #self.params_conv_5_bn = tf_util.BatchNormVars(scope='convbc5')
# 
#         # Define process block params
#         if self.process_block_method == 'attention-rnn':
#             self.process_block_cells = []
#             self.process_block_state_starts = []
#             for layer_idx in range(len(self.process_block_steps)):
#                 self.process_block_cells.append(MyLSTMCell(num_units = self.read_block_units[-1]*4,
#                                                            num_out = self.read_block_units[-1]*2, name = 'process_layer_' + str(layer_idx)))
#                 self.process_block_state_starts.append(self.process_block_cells[-1].zero_state(self.batch_size, tf.float32))
# 
#         # Define input t-net-1 params
#         if self.input_t_net:
#             self.params_t1conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=8, kernel_size = [1,3], scope='t1con1')
#             #self.params_t1conv_1_bn = tf_util.BatchNormVars(scope='t1convbc1')
#             self.params_t1conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=64, kernel_size = [1,1], scope='t1con2')
#             #self.params_t1conv_2_bn = tf_util.BatchNormVars(scope='t1convbc2')
#             self.params_t1conv_3 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=256, kernel_size = [1,1], scope='t1con3')
#             #self.params_t1conv_3_bn = tf_util.BatchNormVars(scope='t1convbc3')
#             self.params_t1fc1 = tf_util.FullyConnVars(num_inputs=256, num_outputs=128, scope='t1fc1')
#             #self.params_t1fc1_bn = tf_util.BatchNormVars(scope='t1fc1bn')
#             self.params_t1fc2 = tf_util.FullyConnVars(num_inputs=128, num_outputs=64, scope='t1fc2')
#             #self.params_t1fc2_bn = tf_util.BatchNormVars(scope='t1fc2bn')
#  
#             self.params_t1xyz_weights = tf.get_variable('t1weights', [64, 9], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
#             self.params_t1xyz_biases = tf.get_variable('t1biases', initializer=np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32))
# 
#         # Define input t-net-2 params
#         if self.feature_t_net:
#             #self.params_t2conv_1 = tf_util.Conv2DVars(num_in_channels=2*self.read_block_units[-1], num_out_channels=256, kernel_size = [1,1], scope='t2con1')
#             self.params_t2conv_1 = tf_util.Conv2DVars(num_in_channels=32, num_out_channels=64, kernel_size = [1,1], scope='t2con1')
#             #self.params_t2conv_1_bn = tf_util.BatchNormVars(scope='t2convbc1')
#             #self.params_t2conv_2 = tf_util.Conv2DVars(num_in_channels=256, num_out_channels=512, kernel_size = [1,1], scope='t2con2')
#             self.params_t2conv_2 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='t2con2')
#             #self.params_t2conv_2_bn = tf_util.BatchNormVars(scope='t2convbc2')
#             #self.params_t2conv_3 = tf_util.Conv2DVars(num_in_channels=512, num_out_channels=1024, kernel_size = [1,1], scope='t2con3')
#             self.params_t2conv_3 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=256, kernel_size = [1,1], scope='t2con3')
#             #self.params_t2conv_3_bn = tf_util.BatchNormVars(scope='t2convbc3')
#             self.params_t2fc1 = tf_util.FullyConnVars(num_inputs=256, num_outputs=128, scope='t2fc1')
#             #self.params_t2fc1_bn = tf_util.BatchNormVars(scope='t2fc1bn')
#             self.params_t2fc2 = tf_util.FullyConnVars(num_inputs=128, num_outputs=64, scope='t2fc2')
#             #self.params_t2fc2_bn = tf_util.BatchNormVars(scope='t2fc2bn')
 
        elif self.read_block_method == 'pointnet':
            self.params_conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=8, kernel_size = [1,3], scope='conv1')
            #self.params_conv_1_bn = tf_util.BatchNormVars(scope='convbc1')
            #self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=32, kernel_size = [1,1], scope='conv2')
            self.params_conv_2 = tf_util.Conv2DVars(num_in_channels=8, num_out_channels=64, kernel_size = [1,1], scope='conv2')
            #self.params_conv_2_bn = tf_util.BatchNormVars(scope='convbc2')
            #self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=32, num_out_channels=128, kernel_size = [1,1], scope='conv3')
            self.params_conv_3 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='conv3')
            #self.params_conv_3_bn = tf_util.BatchNormVars(scope='convbc3')
            self.params_conv_4 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=256, kernel_size = [1,1], scope='conv4')
            #self.params_conv_4_bn = tf_util.BatchNormVars(scope='convbc4')
            self.params_conv_5 = tf_util.Conv2DVars(num_in_channels=256, num_out_channels=self.read_block_units[-1]*2, kernel_size = [1,1], scope='conv5')
            #self.params_conv_5_bn = tf_util.BatchNormVars(scope='convbc5')
 
        # Define process block params
        if self.process_block_method == 'attention-rnn':
            self.process_block_cells = []
            self.process_block_state_starts = []
            for layer_idx in range(len(self.process_block_steps)):
                self.process_block_cells.append(MyLSTMCell(num_units = self.read_block_units[-1]*4,
                                                           num_out = self.read_block_units[-1]*2, name = 'process_layer_' + str(layer_idx)))
                self.process_block_state_starts.append(self.process_block_cells[-1].zero_state(self.batch_size, tf.float32))
 
        # Define input t-net-1 params
        if self.input_t_net:
            self.params_t1conv_1 = tf_util.Conv2DVars(num_in_channels=1, num_out_channels=64, kernel_size = [1,3], scope='t1con1')
            #self.params_t1conv_1_bn = tf_util.BatchNormVars(scope='t1convbc1')
            self.params_t1conv_2 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=128, kernel_size = [1,1], scope='t1con2')
            #self.params_t1conv_2_bn = tf_util.BatchNormVars(scope='t1convbc2')
            self.params_t1conv_3 = tf_util.Conv2DVars(num_in_channels=128, num_out_channels=1024, kernel_size = [1,1], scope='t1con3')
            #self.params_t1conv_3_bn = tf_util.BatchNormVars(scope='t1convbc3')
            self.params_t1fc1 = tf_util.FullyConnVars(num_inputs=1024, num_outputs=512, scope='t1fc1')
            #self.params_t1fc1_bn = tf_util.BatchNormVars(scope='t1fc1bn')
            self.params_t1fc2 = tf_util.FullyConnVars(num_inputs=512, num_outputs=256, scope='t1fc2')
            #self.params_t1fc2_bn = tf_util.BatchNormVars(scope='t1fc2bn'
  
            self.params_t1xyz_weights = tf.get_variable('t1weights', [256, 9], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            self.params_t1xyz_biases = tf.get_variable('t1biases', initializer=np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32))
 
        # Define input t-net-2 params
        if self.feature_t_net:
            self.params_t2conv_1 = tf_util.Conv2DVars(num_in_channels=64, num_out_channels=256, kernel_size = [1,1], scope='t2con1')        
            #self.params_t2conv_1_bn = tf_util.BatchNormVars(scope='t2convbc1')
            self.params_t2conv_2 = tf_util.Conv2DVars(num_in_channels=256, num_out_channels=512, kernel_size = [1,1], scope='t2con2')
            #self.params_t2conv_2_bn = tf_util.BatchNormVars(scope='t2convbc2')
            self.params_t2conv_3 = tf_util.Conv2DVars(num_in_channels=512, num_out_channels=1024, kernel_size = [1,1], scope='t2con3')
            #self.params_t2conv_3_bn = tf_util.BatchNormVars(scope='t2convbc3')
            self.params_t2fc1 = tf_util.FullyConnVars(num_inputs=1024, num_outputs=512, scope='t2fc1')
            #self.params_t2fc1_bn = tf_util.BatchNormVars(scope='t2fc1bn')
            self.params_t2fc2 = tf_util.FullyConnVars(num_inputs=512, num_outputs=256, scope='t2fc2')
            #self.params_t2fc2_bn = tf_util.BatchNormVars(scope='t2fc2bn')
 
            #K = 2*self.read_block_units[-1]
            K = 64
            self.params_t2xyz_weights = tf.get_variable('t2weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            self.params_t2xyz_biases = tf.get_variable('t2biases', initializer=np.eye(K, dtype=np.float32).flatten())

        #self.parameters = {}
        #self.parameters["W1"] = tf.get_variable("W1", [self.read_block_units[-1]*2, self.read_block_units[-1]*2], initializer=tf.contrib.layers.xavier_initializer())
        #self.parameters["b1"] = tf.get_variable("b1", [self.read_block_units[-1]*2], initializer=tf.zeros_initializer())
        #self.parameters["W2"] = tf.get_variable("W2", [self.read_block_units[-1]*2, self.read_block_units[-1]*2], initializer=tf.contrib.layers.xavier_initializer())
        #self.parameters["b2"] = tf.get_variable("b2", [self.read_block_units[-1]*2], initializer=tf.zeros_initializer())
        

        if self.verbose:
            print ("OK!")

    def _define_input_transform_net_inner(self, input, is_training, bn_decay=None):
        batch_size = input.get_shape()[0].value
        num_point = input.get_shape()[1].value

        input_image = tf.expand_dims(input, -1)
        net = tf_util.conv2d(input_image, self.params_t1conv_1, None,#self.params_t1conv_1_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t1conv_2, None, #self.params_t1conv_2_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t1conv_3, None, #self.params_t1conv_3_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='t1maxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, self.params_t1fc1, None, #self.params_t1fc1_bn,
                                      bn=False, is_training=is_training,
                                      bn_decay=bn_decay)
        net = tf_util.fully_connected(net, self.params_t1fc2, None, #self.params_t1fc2_bn,
                                      bn=False, is_training=is_training,
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
                    print ("OK!")

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
                    print ("OK!")

                ret = tf.stack([anr, pos, neg], axis=1)

            return ret

    def _define_feature_transform_net_inner(self, input, is_training, bn_decay=None):
        batch_size = input.get_shape()[0].value
        num_point = input.get_shape()[1].value
    
        input_image = tf.expand_dims(input, -2)
        net = tf_util.conv2d(input_image, self.params_t2conv_1, None, #self.params_t2conv_1_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t2conv_2, None, #self.params_t2conv_2_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_t2conv_3, None, #self.params_t2conv_3_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='t2maxpool')
    
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, self.params_t2fc1, None, #self.params_t2fc1_bn, 
                                      bn=False, is_training=is_training,
                                      bn_decay=bn_decay)
        net = tf_util.fully_connected(net, self.params_t2fc2, None, #self.params_t1fc2_bn,
                                      bn=False, is_training=is_training,
                                      bn_decay=bn_decay)
    
        with tf.variable_scope('transform_feat') as sc:
            transform = tf.matmul(net, self.params_t2xyz_weights)
            transform = tf.nn.bias_add(transform, self.params_t2xyz_biases)
    
        #transform = tf.reshape(transform, [batch_size, 2*self.read_block_units[0], 2*self.read_block_units[0]])
        transform = tf.reshape(transform, [batch_size, 64, 64])
        return transform

    def _define_feature_transform_net(self, input, is_training, bn_decay=None, scope='transform_net_2'):

        with tf.name_scope(scope):
            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:
                
                if self.verbose:
                    sys.stdout.write("Defining transform net 2...")
                    sys.stdout.flush()
                
                transform = self._define_feature_transform_net_inner(inputs[0], is_training=is_training, bn_decay=bn_decay)
                out = tf.matmul(inputs[0], transform)

                if self.verbose:
                    print ("OK!")

                ret = tf.stack([out], axis=1)

            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining transform net 2 for anchor...")
                    sys.stdout.flush()

                self.transform_anr = self._define_feature_transform_net_inner(inputs[0], is_training=is_training, bn_decay=bn_decay)
                anr = tf.matmul(inputs[0], self.transform_anr)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining transform net 2 for positive...")
                    sys.stdout.flush()

                transform_pos = self._define_feature_transform_net_inner(inputs[1], is_training=is_training, bn_decay=bn_decay)
                pos = tf.matmul(inputs[1], transform_pos)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining transform net 2 for negative...")
                    sys.stdout.flush()

                transform_neg = self._define_feature_transform_net_inner(inputs[2], is_training=is_training, bn_decay=bn_decay)
                neg = tf.matmul(inputs[2], transform_neg)

                if self.verbose:
                    print ("OK!")

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
                    print ("OK!")

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
                    print ("OK!")

                ret = tf.stack([anr, pos, neg], axis=1)
            else:
                raise ValueError("I cannot handle this!")

            # return
            return ret

    def _define_read_block_pointnet_inner(self, input, is_training, bn_decay=None):
        
        batch_size = input.get_shape()[0].value
        num_point = input.get_shape()[1].value
        input_image = tf.expand_dims(input, -1)
        net = tf_util.conv2d(input_image, self.params_conv_1, None, #self.params_conv_1_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_2, None, #self.params_conv_2_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
          
        # T net
        if self.feature_t_net:
            net = tf.squeeze(net, axis=-2)
            transform = self._define_feature_transform_net_inner(net, is_training=is_training, bn_decay=bn_decay)
            net = tf.matmul(net, transform)
            net = tf.expand_dims(net, axis=-2)
        
        #HERE FEATURE TRANSFORM?
        net = tf_util.conv2d(net, self.params_conv_3, None, #self.params_conv_3_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_4, None, #self.params_conv_4_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf_util.conv2d(net, self.params_conv_5, None, #self.params_conv_5_bn,
                             padding='VALID', stride=[1,1],
                             bn=False, is_training=is_training,
                             bn_decay=bn_decay)
        net = tf.squeeze(net, axis=-2)

        return net, transform
        
    def _define_read_block_pointnet(self, input, is_training, bn_decay=None, scope='read_block'):

        with tf.name_scope(scope):
            # Get layer output
            inputs = tf.unstack(input, axis=1)
            ret = None

            if len(inputs) == 1:
                
                if self.verbose:
                    sys.stdout.write("Defining read block for embedding [pointnet]...")
                    sys.stdout.flush()

                out, _ = self._define_read_block_pointnet_inner(inputs[0], is_training, bn_decay)

                if self.verbose:
                    print ("OK!")

                ret = tf.stack([out], axis=1)
                
            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining read block for anchor...")
                    sys.stdout.flush()

                anr, self.transform_anr = self._define_read_block_pointnet_inner(inputs[0], is_training, bn_decay)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for positive...")
                    sys.stdout.flush()

                pos, self.transform_pos = self._define_read_block_pointnet_inner(inputs[1], is_training, bn_decay)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining read block for negative...")
                    sys.stdout.flush()

                neg, self.transform_neg = self._define_read_block_pointnet_inner(inputs[2], is_training, bn_decay)

                if self.verbose:
                    print ("OK!")

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

    def max_pool2d(self, inputs,
                   kernel_size,
                   stride=[2, 2],
                   padding='VALID'):
      """ 2D max pooling.

      Args:
          inputs: 4-D tensor BxHxWxC
          kernel_size: a list of 2 ints
          stride: a list of 2 ints
  
      Returns:
          Variable tensor
      """
#      with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      stride_h, stride_w = stride
      outputs = tf.nn.max_pool(inputs,
                                   ksize=[1, kernel_h, kernel_w, 1],
                                   strides=[1, stride_h, stride_w, 1],
                                   padding=padding)
#                                   name=sc.name)
      return outputs

    def _define_process_block(self, input, method):
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

                if method == 'attention-rnn':
                    ret = self._define_process_block_inner(inputs[0])
                if method == 'max-pool':
                    ret = tf.squeeze(self.max_pool2d(tf.expand_dims(inputs[0], axis=-2), kernel_size=[inputs[0].shape[1], 1]))
                    #ret = tf.nn.relu(tf.add(tf.matmul(ret, self.parameters["W1"]), self.parameters["b1"]))
                    #ret = tf.nn.relu(tf.add(tf.matmul(ret, self.parameters["W2"]), self.parameters["b2"]))
                    ret = tf.expand_dims(ret, axis=1)
                if self.verbose:
                    print ("OK!")         

            elif len(inputs) == 3:

                if self.verbose:
                    sys.stdout.write("Defining process block for anchor")
                    sys.stdout.flush()


                if method == 'attention-rnn':
                    anchor_ret = self._define_process_block_inner(inputs[0])
                if method == 'max-pool':
                    anchor_ret = tf.squeeze(self.max_pool2d(tf.expand_dims(inputs[0], axis=-2), kernel_size=[inputs[0].shape[1], 1]))
                    #anchor_ret = tf.nn.relu(tf.add(tf.matmul(anchor_ret, self.parameters["W1"]), self.parameters["b1"]))
                    #anchor_ret = tf.nn.relu(tf.add(tf.matmul(anchor_ret, self.parameters["W2"]), self.parameters["b2"]))
                    anchor_ret = tf.expand_dims(anchor_ret, axis=1)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining process block for positive")
                    sys.stdout.flush()

                if method == 'attention-rnn':
                    positive_ret = self._define_process_block_inner(inputs[1])
                if method == 'max-pool':
                    positive_ret = tf.squeeze(self.max_pool2d(tf.expand_dims(inputs[1], axis=-2), kernel_size=[inputs[1].shape[1], 1]))
                    #positive_ret = tf.nn.relu(tf.add(tf.matmul(positive_ret, self.parameters["W1"]), self.parameters["b1"]))
                    #positive_ret = tf.nn.relu(tf.add(tf.matmul(positive_ret, self.parameters["W2"]), self.parameters["b2"]))
                    positive_ret = tf.expand_dims(positive_ret, axis=1)

                if self.verbose:
                    sys.stdout.write("OK!\nDefining process block for negative")
                    sys.stdout.flush()

                if method == 'attention-rnn':
                    negative_ret = self._define_process_block_inner(inputs[2])
                if method == 'max-pool':
                    negative_ret = tf.squeeze(self.max_pool2d(tf.expand_dims(inputs[2], axis=-2), kernel_size=[inputs[2].shape[1], 1]))
                    #negative_ret = tf.nn.relu(tf.add(tf.matmul(negative_ret, self.parameters["W1"]), self.parameters["b1"]))
                    #negative_ret = tf.nn.relu(tf.add(tf.matmul(negative_ret, self.parameters["W2"]), self.parameters["b2"]))
                    negative_ret = tf.expand_dims(negative_ret, axis=1)

                if self.verbose:
                    print ("OK!")

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
                ret = self._triplet_cosine_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2],
                                                self.transform_anr, self.regularization_weight)#, self.placeholder_label, self.classes_learning_weights)
            elif self.distance == 'euclidian':
                ret = self._triplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], self.transform_anr, self.regularization_weight)
            else:
               raise ValueError("I don't know this embeddings distance..") 

        if self.verbose:
            print ("OK!")
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
