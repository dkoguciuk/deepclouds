#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import argparse
import tensorflow as tf
import siamese_pointnet.defines as df
from siamese_pointnet.model import Model
import siamese_pointnet.modelnet_data as modelnet

def train_pointnet(name, batch_size, epochs, learning_rate, margin,
                   layers_sizes=[2048, 1024, 512, 256, 128],
                   initialization_method="xavier", hidden_activation="relu", output_activation="relu"):
    """
    Train siamese pointnet.
    """

    # Define model
    model = Model(layers_sizes, batch_size, learning_rate,
                  initialization_method, hidden_activation, output_activation, margin)

    # Init all vars
    init = tf.initialize_all_variables()

    # Session
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        log_model_dir = os.path.join(df.LOGS_DIR, model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
        writer.add_graph(sess.graph)

        # Do the training loop
        global_batch_idx = 1
        for epoch in range(epochs):

            # modelnet data object
            modelnet_data = modelnet.ModelnetData()

            # loop for all batches
            index = 1
            for clouds in modelnet_data.generate_train_tripples(batch_size, shuffle_files=False, shuffle_pointclouds=False,
                                                                jitter_pointclouds=True, rotate_pointclouds=False,
                                                                reshape_flags=["flatten_pointclouds",
                                                                                 "transpose_pointclouds"]):

                # Info
                print "Epoch: ", epoch + 1, " cloud: ", index
                index += 1

                # run optimizer
                summary_train_batch = sess.run(model.get_summary(), feed_dict={model.input_a: clouds[0],
                                                                               model.input_p: clouds[1],
                                                                               model.input_n: clouds[2]})
                writer.add_summary(summary_train_batch, global_batch_idx)
                global_batch_idx += 1

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=32)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=100)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-m", "--margin", help="Triple loss margin value", type=float, required=False, default=0.2)
    args = vars(parser.parse_args())

    # train
    train_pointnet(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
                   learning_rate=args["learning_rate"], margin=args["margin"])

    # Print all settings at the end of learning
    print "MLP-basic model:"
    print "name          = ", args["name"]
    print "batch_size    = ", args["batch_size"]
    print "epochs        = ", args["epochs"]
    print "learning rate = ", args["learning_rate"]
    print "margin        = ", args["margin"]

if __name__ == "__main__":
    main(sys.argv[1:])
