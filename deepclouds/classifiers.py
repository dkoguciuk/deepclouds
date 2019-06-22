import os
import time
import tensorflow as tf
import deepclouds.model as md

class MLPClassifier(md.GenericModel):

    MODEL_NAME = "MLPClassifier"
    """
    Name of the model, which will be used as a directory for tensorboard logs. 
    """

    CLASSES_COUNT = 40
    """
    How many classes do we have in the modelnet dataset.
    """

    def __init__(self, name, mlp_layers_sizes, batch_size, learning_rate, reg_weight, drop_keepprob):
        """
        Build a model.
        Args:
            input_features_sizes (int): 
            mlp_layers_sizes (list of ints): List of hidden units of mlp, where the first
                value is considered to be a size of feature vector for each pointcloud.
            batch_size (int): Batch size of SGD.
            learning_rate (float): Learning rate of SGD.
        """        
        # Save params
        self.mlp_layers_sizes = mlp_layers_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.drop_keepprob = drop_keepprob
        self.summaries = []
        self.name = name
        
        with tf.variable_scope(self.name):
            
            # Placeholders for input clouds - we will interpret numer of points in the cloud as timestep with 3 coords as an input number
            with tf.name_scope("placeholders"):
                self.placeholder_embed = tf.placeholder(tf.float32, [self.batch_size, self.mlp_layers_sizes[0]], name="input_embedding")
                self.placeholder_label = tf.placeholder(tf.float32, [self.batch_size, self.mlp_layers_sizes[-1]], name="true_labels")
    
            # init MLP params
            with tf.name_scope("params"):
                self._init_params()
            
            # calculate loss & optimizer
            with tf.name_scope("train"):
                self.classification_pred = self._define_classifier(self.placeholder_embed)
                self.loss = self._calculate_loss(self.classification_pred, self.placeholder_label)
                self.optimizer = self._define_optimizer(self.loss)
                self.summaries.append(tf.summary.scalar('loss', self.loss))
            
            # merge summaries and write        
            self.summary = tf.summary.merge(self.summaries)

    def get_classification_prediction(self):
        """
        Get classification prediction to be run with a batch of a sifnle pointclouds.
        """
        return self.classification_pred

    def save_model(self, session, model_name):
        """
        Save the model in the model dir.

        Args:
            session (tf.Session): Session which one want to save model.
        """    
        saver = tf.train.Saver()
        name = model_name + time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + ".ckpt"
        return saver.save(session, os.path.join("models_feature_extractor", name))       

    def _init_params(self):
        """
        Initialize params for MLP network classifier.
        """
        # Define MLP params
        self.mlp_params = {}
        for layer_idx in range(1, len(self.mlp_layers_sizes)):
            self.mlp_params['class_W' + str(layer_idx)] = tf.get_variable('class_W' + str(layer_idx), 
                                                                          [self.mlp_layers_sizes[layer_idx-1], self.mlp_layers_sizes[layer_idx]], 
                                                                          initializer = tf.contrib.layers.xavier_initializer(),
                                                                          dtype=tf.float32)
            self.mlp_params["class_b" + str(layer_idx)] = tf.get_variable("class_b" + str(layer_idx),
                                                                          [self.mlp_layers_sizes[layer_idx]],
                                                                          initializer = tf.zeros_initializer(),
                                                                          dtype=tf.float32)

    def _define_classifier(self, embeddings):
        """
        Define classifier on embeddings vector.

        Args:
            embeddings (np.ndaray of shape [B, E]): embedding of each cloud, where
                B: batch_size, E: size of an embedding of a pointcloud.
        Returns:
            (np.ndarray of shape [B, C]): Prediction probability for each class.
        """
        with tf.name_scope("mlp"):
            AX = embeddings
            for layer_idx in range(1, len(self.mlp_layers_sizes)-1):
                with tf.name_scope("layer_" + str(layer_idx)):
                    ZX = tf.nn.relu(tf.matmul(AX, self.mlp_params['class_W' + str(layer_idx)]) + self.mlp_params['class_b' + str(layer_idx)])
                    AX = tf.nn.dropout(ZX, self.drop_keepprob)            

            with tf.name_scope("layer_" + str(len(self.mlp_layers_sizes)-1)):
                return tf.matmul(AX, self.mlp_params['class_W' + str(len(self.mlp_layers_sizes)-1)]) + self.mlp_params['class_b' + str(len(self.mlp_layers_sizes)-1)]

    def _calculate_loss(self, predictions, true_labels):
        """
        Calculate loss.

        Args:
            predictions (np.ndaray of shape [B, C]): prediction of a class for each cloud, where:
                B: batch_size, C: prediction probability for each class.
            true_labels (np.ndaray of shape [B, C]): true labels of for each cloud, where:
                B: batch_size, C: true label for each cloud
        Returns:
            (float): Loss of current batch.
        """
        with tf.name_scope("loss"):
            basic_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels, logits=predictions))
            reglr_loss = []
            for layer_idx in range(1, len(self.mlp_layers_sizes)):
                reglr_loss.append(tf.nn.l2_loss(self.mlp_params['class_W' + str(layer_idx)]))
            reglr_loss = tf.reduce_sum(reglr_loss)
            return basic_loss + reglr_loss*self.reg_weight

    def _define_optimizer(self, loss_function):
        """
        Define optimizer operation.
        """
        with tf.name_scope("optimizer"):
            classifier_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_function, var_list=classifier_vars)
        
