import os
import time
import inspect
import importlib
import numpy as np
import tensorflow as tf
from deepclouds.backbones.pointnet import PointNet


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
        
    def _triplet_cosine_loss(self, embedding_a, embedding_p, embedding_n):
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

                # K = transformation_matrix.get_shape()[1].value
                # mat_diff = tf.matmul(transformation_matrix, tf.transpose(transformation_matrix, perm=[0, 2, 1]))
                # mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
                # mat_diff_loss = tf.nn.l2_loss(mat_diff)
                # batch_size = transformation_matrix.get_shape()[0]
                # reg_loss = mat_diff_loss * regularization_weight * tf.to_float(self.non_zero_triplets) / tf.to_float(batch_size)
                # self.summaries.append(tf.summary.scalar('reg_loss', reg_loss))

                final_loss = tf.reduce_mean(self.basic_loss) #+ reg_loss
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


class SiamesePointClouds(GenericModel):
    """
    Feature extraction model similar to the one described in Order Matters paper.
    """

    MODEL_NAME = "DeepCloudsModel"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    def __init__(self, setting):

        # Save params
        self.classes_no = setting.classes_no_in_batch
        self.instances_no = setting.instances_no_in_batch
        self.points_no = setting.points_num
        self.distance_metric = setting.distance_metric
        self.learning_rate = setting.learning_rate
        self.gradient_clip = setting.gradient_clip
        self.non_zero_triplets = []
        self.summaries = []
        self.global_step = tf.Variable(1, trainable=False, name='global_step')

        # Placeholders
        self.input_point_cloud = tf.placeholder(tf.float32, shape=(self.classes_no*self.instances_no, self.points_no,
                                                                   3))
        self.is_training = tf.placeholder(tf.bool, shape=())

        # Import backbone module
        backbone_module = importlib.import_module(setting.backbone_model)
        for name, obj in inspect.getmembers(backbone_module):
            if inspect.isclass(obj):
                if setting.backbone_model in str(obj):
                    backbone_class = obj
                    break

        # Get features
        backbone_model = backbone_class(input_point_cloud=self.input_point_cloud, is_training=self.is_training,
                                        setting=setting.__dict__)
        features = backbone_model.get_features()

        # Find hard indices
        with tf.variable_scope("batch_hard_triplets"):
            self.hard_indices = self.find_batch_hard_triples(features)
            
            # Create triplets to train on        
            self.embds_positive = tf.identity(features)
            self.embds_positive = tf.gather(self.embds_positive, self.hard_indices[0])
            self.embds_positive = tf.reshape(self.embds_positive, (self.classes_no * self.instances_no, -1))
            
            self.embds_negative = tf.identity(features)
            self.embds_negative = tf.gather(self.embds_negative, self.hard_indices[1])
            self.embds_negative = tf.reshape(self.embds_negative, (self.classes_no * self.instances_no, -1))
            
            self.triplets = tf.stack([features, self.embds_positive, self.embds_negative], axis=-2)
        
        # Optimizer
        with tf.variable_scope("optimizer"):
            self.loss = self._calculate_loss(self.triplets)
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
        
        mask_pos_np = np.zeros(shape=(self.classes_no, self.instances_no, self.classes_no, self.instances_no),
                               dtype=np.float64)
        for c in range(self.classes_no):
            mask_pos_np[c, :, c, :] = 1.
        mask_pos = tf.convert_to_tensor(mask_pos_np, dtype=tf.float32)  # C x I x C x I
        mask_pos = tf.reshape(mask_pos, (self.classes_no*self.instances_no, self.classes_no*self.instances_no))
        
        pos_hard = tf.argmax(tf.multiply(distances, mask_pos), axis=-1)
        pos_hard = tf.reshape(pos_hard, (self.classes_no, self.instances_no))
        
        #######################################################################
        # Find hardes negative in the batch
        #######################################################################

        mask_neg_np = np.ones(shape=(self.classes_no, self.instances_no, self.classes_no, self.instances_no),
                              dtype=np.float64)
        for c in range(self.classes_no):
            mask_neg_np[c, :, c, :] = np.inf
        mask_neg = tf.convert_to_tensor(mask_neg_np, dtype=tf.float32)
        mask_neg = tf.reshape(mask_neg, (self.classes_no*self.instances_no, self.classes_no*self.instances_no))
        
        neg_hard = tf.argmin(tf.multiply(distances, mask_neg), axis=-1)
        neg_hard = tf.reshape(neg_hard, (self.classes_no, self.instances_no))
        
        #######################################################################
        # Return pos / neg indices
        #######################################################################
        
        # return
        return pos_hard, neg_hard

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

    def _calculate_loss(self, embeddings):#, tnet_feature):
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
            if self.distance_metric == 'cosine':
                ret = self._triplet_cosine_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2])
            elif self.distance_metric == 'euclidian':
                ret = self._triplet_loss(embeddings_list[0], embeddings_list[1], embeddings_list[2])
            else:
               raise ValueError("I don't know this embeddings distance..") 

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


