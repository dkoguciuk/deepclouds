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
                self.pos_dist = tf.norm(embedding_a - embedding_p, axis=-1)
            with tf.name_scope("dist_neg"):
                self.neg_dist = tf.norm(embedding_a - embedding_n, axis=-1)
            with tf.name_scope("copute_loss"):
                #self.soft_loss = tf.nn.softplus(self.pos_dist - self.neg_dist)
                self.soft_loss = tf.log(tf.exp(self.pos_dist - self.neg_dist) + 1)
                #self.basic_loss = tf.maximum(self.margin + self.pos_dist - self.neg_dist, 0.0)
                #self.non_zero_triplets = tf.count_nonzero(self.basic_loss)
                #self.summaries.append(tf.summary.scalar('non_zero_triplets', self.non_zero_triplets))
                
#                 K = transformation_matrix.get_shape()[1].value
#                 mat_diff = tf.matmul(transformation_matrix, tf.transpose(transformation_matrix, perm=[0, 2, 1]))
#                 mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
#                 mat_diff_loss = tf.nn.l2_loss(mat_diff)
#                 batch_size = transformation_matrix.get_shape()[0]
#                 reg_loss = mat_diff_loss * regularization_weight * tf.to_float(self.non_zero_triplets) / tf.to_float(batch_size)
#                 self.summaries.append(tf.summary.scalar('reg_loss', reg_loss))
                
                # final_loss = tf.reduce_mean(self.basic_loss)
                #final_loss = tf.reduce_sum(self.basic_loss)# + reg_loss
                final_loss = tf.reduce_sum(self.soft_loss)
            return final_loss
        
    def _triplet_cosine_loss(self, embedding_a, embedding_p, embedding_n,
                             transformation_matrix, regularization_weight):  # , labels_a, classes_learning_weights):
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
                # self.pos_dist = 1 - pos_nom / (pos_den + 1e-9)
                self.pos_dist = 1 - pos_nom / (pos_den)
            with tf.name_scope("dist_neg"):
                neg_nom = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0], x[2])), e, dtype=tf.float32)
                neg_den = tf.multiply(tf.norm(embedding_a, axis=-1), tf.norm(embedding_n, axis=-1))
                self.neg_den = neg_den
                # self.neg_dist = 1 - neg_nom / (neg_den + 1e-9)
                self.neg_dist = 1 - neg_nom / (neg_den)
            with tf.name_scope("copute_loss"):
                self.basic_loss = tf.maximum(self.pos_dist + self.margin - self.neg_dist, 0.0)
                # loss_num = tf.gather(classes_learning_weights, labels_a)
                # loss_den = tf.reduce_sum(tf.where(self.basic_loss > 0, loss_num, tf.zeros(loss_num.get_shape())))
                # self.weighted_loss = tf.multiply((loss_num / loss_den), self.basic_loss)

                self.non_zero_triplets = tf.count_nonzero(self.basic_loss)
                self.summaries.append(tf.summary.scalar('non_zero_triplets', self.non_zero_triplets))

                K = transformation_matrix.get_shape()[1].value
                mat_diff = tf.matmul(transformation_matrix, tf.transpose(transformation_matrix, perm=[0, 2, 1]))
                mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                batch_size = transformation_matrix.get_shape()[0]
                reg_loss = mat_diff_loss * regularization_weight * tf.to_float(self.non_zero_triplets) / tf.to_float(batch_size)
                self.summaries.append(tf.summary.scalar('reg_loss', reg_loss))

                final_loss = tf.reduce_mean(self.basic_loss) + reg_loss
                # final_loss = tf.reduce_mean(self.weighted_loss) + reg_loss
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
                 classes_no, instances_no, pointcloud_size,
                 read_block_units, process_block_steps,
                 normalize_embedding=True, verbose=True,
                 learning_rate=0.0001, gradient_clip=10.0,
                 input_t_net=False, feature_t_net=False,
                 read_block_method='birnn',  # birnn or pointnet
                 process_block_method='max-pool',  # max-pool or attention-rnn
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
        self.classes_no = classes_no
        self.instances_no = instances_no
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
        #self.CLASSES_COUNT = batch_size
        
        # Variable decays
        self.global_step = tf.Variable(1, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        
        # input placeholder 
        with tf.variable_scope("placeholders"):
            self.margin = tf.placeholder(tf.float32, shape=1, name="input_margin")
            self.input_point_clouds = tf.placeholder(tf.float32, [self.classes_no * self.instances_no, self.pointcloud_size, 3], name="input_point_clouds")
            self.placeholder_is_tr = tf.placeholder(tf.bool, shape=(), name="input_is_training")
        
        # input t-net
        with tf.variable_scope("tnet-input"):
            self.tnet_input = self._define_tnet_input(self.input_point_clouds, is_training=self.placeholder_is_tr, bn_decay=None)
            self.data_after_step_1 = tf.matmul(self.input_point_clouds, self.tnet_input)        # C*I x N x 3
            # (C*I x N x 3) 
        
        # Read block part 1
        with tf.variable_scope("read_block_part_1"):
            self.data_after_step_2 = self._define_read_block_part_1(self.data_after_step_1, is_training=self.placeholder_is_tr, bn_decay=None)
            # (C*I x N x 64)      
        
        # Feature t-net
        with tf.variable_scope("tnet-feature"):
            self.tnet_feature = self._define_tnet_feature(self.data_after_step_2, is_training=self.placeholder_is_tr, bn_decay=None)
            self.data_after_step_3 = tf.matmul(self.data_after_step_2, self.tnet_feature)
            # (C*I x N x 64)
        
        # Read block part 2
        with tf.variable_scope("read_block_part_2"):
            self.data_after_step_4 = self._define_read_block_part_2(self.data_after_step_3, is_training=self.placeholder_is_tr, bn_decay=None)
        # (C*I x N x 512)
    
        # Process block
        with tf.variable_scope("process_block"):
            self.data_after_step_5 = tf.squeeze(self.max_pool2d(tf.expand_dims(self.data_after_step_4, axis=-2), kernel_size=[self.data_after_step_4.shape[1], 1]))
            #self.data_after_step_5 = tf.nn.l2_normalize(self.data_after_step_5, axis=-1, epsilon=1e-10)
        # (C*I x 64)
        
        # Find hard indices
        with tf.variable_scope("batch_hard_triplets"):
            self.hard_indices = self.find_batch_hard_triples(self.data_after_step_5)
            
            # Create triplets to train on        
            self.embds_positive = tf.identity(self.data_after_step_5)
            self.embds_positive = tf.gather(self.embds_positive, self.hard_indices[0])
            self.embds_positive = tf.reshape(self.embds_positive, (self.classes_no * self.instances_no, -1))
            
            self.embds_negative = tf.identity(self.data_after_step_5)
            self.embds_negative = tf.gather(self.embds_negative, self.hard_indices[1])
            self.embds_negative = tf.reshape(self.embds_negative, (self.classes_no * self.instances_no, -1))
            
            self.triplets = tf.stack([self.data_after_step_5, self.embds_positive, self.embds_negative], axis=-2)
        
        # Optimizer
        with tf.variable_scope("optimizer"):
            self.loss = self._calculate_loss(self.triplets, self.tnet_feature)
            self.optimizer = self._define_optimizer(self.loss)
            self.summaries.append(tf.summary.scalar('loss', self.loss))

        # merge summaries and write
        self.summary = tf.summary.merge(self.summaries)

    def find_batch_hard_triples(self, input_data):
        """
        Find triplets with batch hard approach.
        
        Args:
            input_data (tf.tensor of size [B, E])
            
        Returns:
            HMMM?
        """
        
        #######################################################################
        # Compute euclidian distance between each pairs of embeddings in the batch
        #######################################################################
        
        def calc_euclidian_distances(elem):
            """
            Compute distance between querry embedding and all embeddings in the batch.
            
            Args:
                elem (tf.tensor of size E)
            
            Returns:
                (tf.tensor of size B)
            """
            diff = input_data - elem
            norm = tf.norm(diff, axis=-1)
            return norm
        
        distances = tf.map_fn(calc_euclidian_distances, input_data)      # C*I x C*I
        
        #######################################################################
        # Find hardes positive in the batch
        #######################################################################
        
        mask_pos_np = np.zeros(shape=(self.classes_no, self.instances_no, self.classes_no, self.instances_no), dtype=np.float64)
        for c in range(self.classes_no):
            mask_pos_np[c,:,c,:] = 1.
        mask_pos = tf.convert_to_tensor(mask_pos_np, dtype=tf.float32)  # C x I x C x I
        mask_pos = tf.reshape(mask_pos, (self.classes_no*self.instances_no, self.classes_no*self.instances_no))
        
        pos_hard = tf.argmax(tf.multiply(distances, mask_pos), axis=-1)
        pos_hard = tf.reshape(pos_hard, (self.classes_no, self.instances_no))
        
        #######################################################################
        # Find hardes negative in the batch
        #######################################################################

        mask_neg_np = np.ones(shape=(self.classes_no, self.instances_no, self.classes_no, self.instances_no), dtype=np.float64)
        for c in range(self.classes_no):
            mask_neg_np[c,:,c,:] = np.inf
        mask_neg = tf.convert_to_tensor(mask_neg_np, dtype=tf.float32)
        mask_neg = tf.reshape(mask_neg, (self.classes_no*self.instances_no, self.classes_no*self.instances_no))
        
        neg_hard = tf.argmin(tf.multiply(distances, mask_neg), axis=-1)
        neg_hard = tf.reshape(neg_hard, (self.classes_no, self.instances_no))
        
        #######################################################################
        # Return pos / neg indices
        #######################################################################
        
        # return
        return (pos_hard, neg_hard)

    def find_semihard_triples(self, input_data):
        """
        Find triplets with batch hard approach.
        
        Args:
            input_data (tf.tensor of size [C, I, E])
            
        Returns:
            HMMM?
        """
    
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
            class_idxs = tf.where(labels == labels[i])  # All indexes within the same class
            class_dist = tf.gather(dist, class_idxs)  # All distances within the same class
            class_dmax = tf.argmax(class_dist)  # Index of the biggest distance in the class_dist/class_idx 
            posit_hard = tf.gather(class_idxs, class_dmax)  # Index of the biggest distance in the batch

            return [tf.add(i, 1), tf.concat([d, d], axis=0)]
        
        # do the loop:
        r = tf.while_loop(while_condition, body, [i0, d0], shape_invariants=[i0.get_shape(), tf.TensorShape([None, 1])])

    def _define_tnet_input(self, input_data, is_training, bn_decay=None):
        """
        Define input tnet.
        
        Args:
            input_data (tf.tensor of size B, N, 3)
        
        Returns:
            tf.tensor of size B, 3, 3
        """

        # Define tnet's convs
        with tf.variable_scope("t1_prep"):
            batch_size = input_data.get_shape()[0].value
            num_point = input_data.get_shape()[1].value
            input_image = tf.expand_dims(input_data, -1)
        
        with tf.variable_scope("t1_conv1"):
            net = tf_util.conv2d_on_the_fly(input_image, num_in_channels=1, num_out_channels=64, kernel_size=[1, 3],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)

        with tf.variable_scope("t1_conv2"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=64, num_out_channels=128, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("t1_conv3"):  
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=128, num_out_channels=1024, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
            
        with tf.variable_scope("t1_max_pool"): 
            net = tf_util.max_pool2d_on_the_fly(net, [num_point, 1], padding='VALID')
            net = tf.reshape(net, [batch_size, -1])

        # Define tnet's FC
        with tf.variable_scope("t1_fc_1"):
            net = tf_util.fully_connected_on_the_fly(net, num_inputs=1024, num_outputs=512,
                                                     bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("t1_fc_2"):
            net = tf_util.fully_connected_on_the_fly(net, num_inputs=512, num_outputs=256,
                                                     bn=False, is_training=is_training, bn_decay=bn_decay)

        # Define, reshape and return
        with tf.variable_scope('t1_transform') as sc:
            
            params_t1xyz_weights = tf.get_variable('t1_weights', [256, 9], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            params_t1xyz_biases = tf.get_variable('t1_biases', initializer=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))
            
            transform = tf.matmul(net, params_t1xyz_weights)
            transform = tf.nn.bias_add(transform, params_t1xyz_biases)
            transform = tf.reshape(transform, [batch_size, 3, 3])

        return transform

    def _define_tnet_feature(self, input_data, is_training, bn_decay=None):
        """
        Define input tnet.
        
        Args:
            input_data (tf.tensor of size B, N, F)
        
        Returns:
            tf.tensor of size B, F, F
        """
        
        # Define tnet's convs
        with tf.variable_scope("t2_prep"):
            batch_size = input_data.get_shape()[0].value
            num_point = input_data.get_shape()[1].value
            input_image = tf.expand_dims(input_data, -2)

        # Fnet convs
        with tf.variable_scope("t2_conv1"):
            net = tf_util.conv2d_on_the_fly(input_image, num_in_channels=64, num_out_channels=256, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
            
        with tf.variable_scope("t2_conv2"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=256, num_out_channels=512, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
            
        with tf.variable_scope("t2_conv3"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=512, num_out_channels=1024, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("t1_max_pool"):
            net = tf_util.max_pool2d_on_the_fly(net, [num_point, 1], padding='VALID')
            net = tf.reshape(net, [batch_size, -1])
    
        # Fnet FC
        with tf.variable_scope("t2_fc_1"):
            net = tf_util.fully_connected_on_the_fly(net, num_inputs=1024, num_outputs=512,
                                                     bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("t2_fc_2"):
            net = tf_util.fully_connected_on_the_fly(net, num_inputs=512, num_outputs=256,
                                                     bn=False, is_training=is_training, bn_decay=bn_decay)
    
        # Create, reshape and return
        with tf.variable_scope('t2_transform') as sc:
            
            K = 64
            params_t2xyz_weights = tf.get_variable('t2_weights', [256, K * K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            params_t2xyz_biases = tf.get_variable('t2_biases', initializer=np.eye(K, dtype=np.float32).flatten())
            
            transform = tf.matmul(net, params_t2xyz_weights)
            transform = tf.nn.bias_add(transform, params_t2xyz_biases)
            transform = tf.reshape(transform, [batch_size, 64, 64])

        return transform

    def _define_read_block_part_1(self, input_data, is_training, bn_decay=None):
        """
        Define input read block part 1.
        
        Args:
            input_data (tf.tensor of size B, N, 3)
        
        Returns:
            tf.tensor of size B, N, F
        """
        
        # Prep
        with tf.variable_scope("prep_in"):
            batch_size = input_data.get_shape()[0].value
            num_point = input_data.get_shape()[1].value
            input_image = tf.expand_dims(input_data, -1)
        
        # Read block convs
        with tf.variable_scope("conv1"):
            net = tf_util.conv2d_on_the_fly(input_image, num_in_channels=1, num_out_channels=8, kernel_size=[1, 3],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("conv2"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=8, num_out_channels=64, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        # Return
        with tf.variable_scope("prep_out"):
            net = tf.squeeze(net, axis=-2)
        return net

    def _define_read_block_part_2(self, input_data, is_training, bn_decay=None):
        """
        Define input read block part 2.
        
        Args:
            input_data (tf.tensor of size B, N, F)
        
        Returns:
            tf.tensor of size B, N, E
        """
        
        # Prep
        with tf.variable_scope("prep_in"):
            net = tf.expand_dims(input_data, axis=-2)
        
        # Read block part 2 convs 
        with tf.variable_scope("conv3"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=64, num_out_channels=128, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("conv4"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=128, num_out_channels=256, kernel_size=[1, 1],
                                            padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)
        
        with tf.variable_scope("conv5"):
            net = tf_util.conv2d_on_the_fly(net, num_in_channels=256, num_out_channels=self.read_block_units[-1] * 2, kernel_size=[1, 1],
                                 padding='VALID', stride=[1, 1], bn=False, is_training=is_training, bn_decay=bn_decay)

        with tf.variable_scope("prep_out"):
            net = tf.squeeze(net, axis=-2)

        # Return
        return net

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

    def _calculate_loss(self, embeddings, tnet_feature):
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
                                                self.transform_anr, self.regularization_weight)  # , self.placeholder_label, self.classes_learning_weights)
            elif self.distance == 'euclidian':
                ret = self._triplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2], tnet_feature, self.regularization_weight)
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
                    # exit()
                else:
                    self.summaries.append(tf.summary.scalar(var.name, tf.norm(grad)))
            return optimizer.apply_gradients(grads_and_vars, self.global_step)


