#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
import deepclouds.defines as df
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

CLOUD_SIZE = 1024
SYNTHETIC = False
SAMPLING_METHOD = 'random'
READ_BLOCK_UNITS = [256]
DISTANCE = 'cosine'
READ_BLOCK_METHOD = 'pointnet'

def save_embeddings(device):

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    #data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)
    data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

    ##################################################################################################
    ################################## FEATURES EXTRACTION MODEL #####################################
    ##################################################################################################

    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            features_model = DeepCloudsModel(train=False,
                                             batch_size = 40,
                                             pointcloud_size = CLOUD_SIZE,
                                             read_block_units=READ_BLOCK_UNITS, process_block_steps=[4],
                                             normalize_embedding=True, verbose=True,
                                             input_t_net=True, feature_t_net=True,
                                             distance=DISTANCE, read_block_method=READ_BLOCK_METHOD)
            features_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="end-to-end")
            features_model_saver = tf.train.Saver(features_vars)

    # Saver
    saver = tf.train.Saver()
    
    if os.path.exists("embeddings"):
        shutil.rmtree("embeddings")
    os.mkdir("embeddings")

    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        # Run the initialization
        sess.run(tf.global_variables_initializer())
         
        # saver    
        saver.restore(sess, tf.train.latest_checkpoint('models'))
        
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print "Variable: ", k
            print "Shape: ", v.shape
        
        # Do the training loop
        global_batch_idx = 1
        summary_skip_batch = 1

        # loop for all batches
        index = 1
        for clouds, labels in data_gen.generate_representative_batch(train=False,
                                                                     instances_number=1,
                                                                     shuffle_points=True,
                                                                     jitter_points=True,
                                                                     rotate_pointclouds=False,
                                                                     rotate_pointclouds_up=True,
                                                                     sampling_method=SAMPLING_METHOD):

            # count embeddings
            embedding_input = np.stack([clouds], axis=1)
            embeddings = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                              features_model.placeholder_is_tr : False})

            for class_idx in range(40):
                data_filapath = "embeddings/data_%04d.npy" % (index*40 + class_idx)
                label_filapath = "embeddings/label_%04d.npy" % (index*40 + class_idx)
                np.save(data_filapath, embeddings[class_idx])
                np.save(label_filapath, labels[class_idx])
            index = index+1
            

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    args = vars(parser.parse_args())

    # save embeddings
    save_embeddings(device=args["device"])

    # Print all settings at the end of learning
    print "DONE!"

if __name__ == "__main__":
    main(sys.argv[1:])
