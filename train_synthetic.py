#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import siamese_pointnet.defines as df
from siamese_pointnet.model import RNNBidirectionalModel, MLPModel
import siamese_pointnet.modelnet_data as modelnet

CLOUD_SIZE = 32


def find_hard_triples_to_train(embeddings, labels):
    hard_positives_indices = []
    hard_negatives_indices = []
    for cloud_idx in range(embeddings.shape[0]):
        # calc distances
        distances = np.linalg.norm(embeddings-embeddings[cloud_idx], axis=1)
        
        # find hard positive
        class_idx = labels[cloud_idx]
        class_indices = np.squeeze(np.argwhere(labels == class_idx))
        hard_positives_indices.append(class_indices[np.argmax(np.take(distances, class_indices))])
        
        # find hard negative
        other_indices = np.squeeze(np.argwhere(labels != class_idx))
        hard_negatives_indices.append(other_indices[np.argmin(np.take(distances, other_indices))])
        
        if cloud_idx == 0:
            print "POSITIVE MAX DIST ", distances[class_indices[np.argmax(np.take(distances, class_indices))]]
            print "NEGATIVE MIN DIST ", distances[other_indices[np.argmin(np.take(distances, other_indices))]]

    return hard_positives_indices, hard_negatives_indices

def train_synthetic(name, batch_size, epochs, learning_rate, margin, device,
                    layers_sizes=[CLOUD_SIZE * 3, CLOUD_SIZE, 1],
                    initialization_method="xavier", hidden_activation="relu", output_activation="relu"):
    """
    Train siamese pointnet with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData()
    data_gen_size = data_gen.check_generated_size()
    if not data_gen_size:
        data_gen.regenerate_files(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model = RNNBidirectionalModel([CLOUD_SIZE * 3, CLOUD_SIZE, 8], [2*8*CLOUD_SIZE, CLOUD_SIZE/2], batch_size, learning_rate,
        # model = MLPModel(layers_sizes, batch_size, learning_rate,
        #                 initialization_method, hidden_activation, output_activation,
                         margin, pointcloud_size=CLOUD_SIZE)

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
        summary_skip_batch = 1
        for epoch in range(epochs):
 
            # loop for all batches
            index = 1
            for clouds, labels in data_gen.generate_representative_batch(batch_size):
                                                                         #reshape_flags=["flatten_pointclouds"]):

                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input})
                
                # Find hard examples to train on
                pos_indices, neg_indices = find_hard_triples_to_train(embeddings, labels)
                
                # Print info
                #print embeddings[0], embeddings[pos_indices[0]], embeddings[neg_indices[0]]
                print np.sum((embeddings[0]-embeddings[pos_indices[0]])**2), np.sum((embeddings[0]-embeddings[neg_indices[0]])**2)
                print np.sum((embeddings[0]-embeddings[pos_indices[0]])**2) - np.sum((embeddings[0]-embeddings[neg_indices[0]])**2) + 0.2
                
                # Create triples to train
                pos_clouds = np.copy(clouds)
                pos_clouds = pos_clouds[pos_indices, ...]
                neg_clouds = np.copy(clouds)
                neg_clouds = neg_clouds[neg_indices, ...]
                training_input = np.stack([clouds, pos_clouds, neg_clouds], axis =1)
                
                # run optimizer
                _, loss, summary = sess.run([model.get_optimizer(), model.get_loss_function(), model.get_summary()],
                                   feed_dict={model.placeholder_train: training_input})
                
                # Tensorboard vis
                if global_batch_idx % summary_skip_batch == 0:
                    writer.add_summary(summary, global_batch_idx)
                global_batch_idx += 1

                # Info
                print "Epoch: %06d batch: %03d loss: %06f" % (epoch + 1, index, loss)
                index += 1

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=80)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=100)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    parser.add_argument("-m", "--margin", help="Triple loss margin value", type=float, required=False, default=0.2)
    args = vars(parser.parse_args())

    # train
    train_synthetic(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
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
