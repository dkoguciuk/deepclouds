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
import deepclouds.defines as df
from deepclouds.model import MLPModel, RNNBidirectionalModel
import deepclouds.modelnet_data as modelnet

def train_deepclouds(name, batch_size, epochs, learning_rate, margin, device,
                     layers_sizes=[2048, 512, 128],
                     initialization_method="xavier", hidden_activation="relu", output_activation="relu"):
    """
    Train deepclouds.
    """

    # Reset
    tf.reset_default_graph()

    # Define model
    with tf.device(device):
        model = RNNBidirectionalModel(layers_sizes, batch_size, learning_rate,
                                      # initialization_method, hidden_activation, output_activation,
                                      margin, pointcloud_size=2048)

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:

        # Run the initialization
        sess.run(tf.global_variables_initializer())
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
            for pointclouds, labels in modelnet_data.generate_representative_batch(batch_size=batch_size, shuffle_files=False, shuffle_pointclouds=False,
                                                                                   jitter_pointclouds=False, rotate_pointclouds_up=False):
                                                                                   #reshape_flags=["flatten_pointclouds"]):
 
                # count embeddings
                embeddings = sess.run(model.count_embeddings(), feed_dict={model.input_a: pointclouds})
                print type(embeddings), embeddings.shape
                exit()
#  
#                 # run optimizer
#                 summary_train_batch, loss = sess.run([model.get_summary(), model.get_loss_function()],
#                                                      feed_dict={model.input_a: clouds[0],
#                                                                 model.input_p: clouds[1],
#                                                                 model.input_n: clouds[2]})
#                 writer.add_summary(summary_train_batch, global_batch_idx)
#                 global_batch_idx += 1
#                  
#                 # Info
#                 print "Epoch: ", epoch + 1, " batch: ", index  # , " loss: ", loss
#                 index += 1

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=64)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=100)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    parser.add_argument("-m", "--margin", help="Triple loss margin value", type=float, required=False, default=0.2)
    args = vars(parser.parse_args())

    # train
    train_deepclouds(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
                     learning_rate=args["learning_rate"], margin=args["margin"], device=args["device"])

    # Print all settings at the end of learning
    print "MLP-basic model:"
    print "name          = ", args["name"]
    print "batch_size    = ", args["batch_size"]
    print "epochs        = ", args["epochs"]
    print "learning rate = ", args["learning_rate"]
    print "margin        = ", args["margin"]

if __name__ == "__main__":
    main(sys.argv[1:])
