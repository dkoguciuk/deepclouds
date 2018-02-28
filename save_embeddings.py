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
import siamese_pointnet.defines as df
import siamese_pointnet.modelnet_data as modelnet
from siamese_pointnet.classifiers import MLPClassifier
from siamese_pointnet.model import RNNBidirectionalModel, MLPModel, OrderMattersModel

CLOUD_SIZE = 32

def save_embeddings(device):

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model = OrderMattersModel(batch_size = 1, pointcloud_size = CLOUD_SIZE,
                                  read_block_units = [2**(np.floor(np.log2(3*CLOUD_SIZE)) + 1)],
                                  process_block_steps=32)

    # Saver
    saver = tf.train.Saver()
    
    if os.path.exists("embeddings"):
        shutil.rmtree("embeddings")
    os.mkdir("embeddings")

    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:

        # Run the initialization
        sess.run(tf.global_variables_initializer())
         
        # saver    
        saver.restore(sess, tf.train.latest_checkpoint('models_17'))
        
        # Do the training loop
        global_batch_idx = 1
        summary_skip_batch = 1

        # loop for all batches
        index = 1
        for clouds, labels in data_gen.generate_representative_batch(train=True,
                                                                     batch_size=1,
                                                                     shuffle_points=False,
                                                                     jitter_points=True,
                                                                     rotate_pointclouds=True):

            # count embeddings
            embedding_input = np.stack([clouds], axis=1)
            embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input})
            
            data_filapath = "embeddings/data_%04d.npy" % index
            label_filapath = "embeddings/label_%04d.npy" % index
            np.save(data_filapath, embeddings)
            np.save(label_filapath, labels)
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
