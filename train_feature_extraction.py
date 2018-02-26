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

def find_semi_hard_triples_to_train_1(embeddings, labels, margin):
    
    def find_nearest_idx(array, value):
        return (np.abs(array-value)).argmin()
    
    hard_positives_indices = []
    hard_negatives_indices = []
    for cloud_idx in range(embeddings.shape[0]):
        # calc distances
        distances = np.linalg.norm(embeddings-embeddings[cloud_idx], axis=1)
        
        # find hard positive
        class_idx = labels[cloud_idx]
        class_indices = np.squeeze(np.argwhere(labels == class_idx))
        hard_positives_indices.append(class_indices[np.argmax(np.take(distances, class_indices))])
        positive_distance = distances[hard_positives_indices[-1]]
        
        # Help vars
        other_indices = np.squeeze(np.argwhere(labels != class_idx))                                            # Find indices of all other class instances
        negatives_distances = np.take(distances, other_indices)                                                 # Find distances to all other class distances
#        negatives_distances[negatives_distances <= positive_distance] = float('inf')                           # Ifinity distance when it's smaller than positive dist
#        negative_idx = find_nearest_idx(negatives_distances, positive_distance + margin/10)                    # Find index of elem in the half of margin range
        
#        negatives_distances[negatives_distances <= (positive_distance + 1e-6)] = float('inf')                            # Ifinity distance when it's smaller than positive dist plus eps
        negatives_distances[negatives_distances <= positive_distance] = float('inf')                            # Ifinity distance when it's smaller than positive dist plus eps
        negative_idx = np.argmin(negatives_distances)                                                           # Smallest
        
        hard_negatives_indices.append(other_indices[negative_idx])                                              # Find negative embedding index 
    
    return hard_positives_indices, hard_negatives_indices 

def train_synthetic_features_extraction(name, batch_size, epochs, learning_rate, margin, device,
                                        rnn_layer_sizes=[CLOUD_SIZE * 3, CLOUD_SIZE * 3, CLOUD_SIZE * 3],
                                        mlp_layer_sizes=[CLOUD_SIZE * 3 * 2 * CLOUD_SIZE, 128]):
    """
    Train siamese pointnet with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
#         model = RNNBidirectionalModel(rnn_layer_sizes, mlp_layer_sizes,
#                                       batch_size, learning_rate, margin, normalize_embedding=True,
#                                       pointcloud_size=CLOUD_SIZE)
        model = OrderMattersModel(batch_size = batch_size, pointcloud_size = CLOUD_SIZE, margin=margin,
                                  read_block_units = [2**(np.floor(np.log2(3*CLOUD_SIZE)) + 1)],
                                  process_block_steps=32, learning_rate=learning_rate)
        # pierwsze testy z 32 krokami process_block_steps
        # oraz z log2 3*cloud_size (96) 

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:
 
        # Run the initialization
        sess.run(tf.global_variables_initializer())
        
        log_model_dir = os.path.join(df.LOGS_DIR, model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
#         writer.add_graph(sess.graph)
 
        # Do the training loop
        global_batch_idx = 1
        #summary_skip_batch = 1
        for epoch in range(epochs):
 
            # loop for all batches
            index = 1
            for clouds, labels in data_gen.generate_representative_batch(train=True,
                                                                         batch_size=batch_size,
                                                                         shuffle_points=False,
                                                                         jitter_points=True,
                                                                         rotate_pointclouds=True):

                ##################################################################################################
                ################################# FIND SEMI HARD TRIPLETS TO LEARN ###############################
                ##################################################################################################
                
                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input})
                
                # Find hard examples to train on
                pos_indices, neg_indices = find_semi_hard_triples_to_train_1(embeddings, labels, margin)
                 
                # Create triples to train
                pos_clouds = np.copy(clouds)
                pos_clouds = pos_clouds[pos_indices, ...]
                neg_clouds = np.copy(clouds)
                neg_clouds = neg_clouds[neg_indices, ...]
                training_input = np.stack([clouds, pos_clouds, neg_clouds], axis =1)
                 
                ##################################################################################################
                ############################################# TRAIN ##############################################
                ##################################################################################################
                 
                # run optimizer
                _, loss, summary_train = sess.run([model.get_optimizer(), model.get_loss_function(), model.get_summary()],
                                                  feed_dict={model.placeholder_train: training_input})
                 
                ##################################################################################################
                ############################## GET POS-NEG DIST DIFF ON TEST SET #################################
                ##################################################################################################
                 
                # Get test embeddings
                test_embeddings = { k : [] for k in range(40)}
                for clouds, labels in data_gen.generate_random_batch(False, 80):# 400 test examples / 80 clouds = 5 batches
 
                    # count embeddings
                    test_embedding_input = np.stack([clouds], axis=1)
                    test_embedding = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: test_embedding_input})
 
                    # add embeddings
                    for cloud_idx in range(labels.shape[0]):
                        test_embeddings[labels[cloud_idx]].append(test_embedding[cloud_idx].tolist())
                 
                # Convert to numpy
                class_embeddings = []
                for k in range(40):
                    class_embeddings.append(test_embeddings[k])
                class_embeddings = np.stack(class_embeddings, axis=0)
 
                # calc distances                
                pos = np.sqrt(np.sum((class_embeddings[:,:-1,:] - class_embeddings[:,1:,:]) **2, axis=2))
                neg = np.sqrt(np.sum((class_embeddings[:-1,:,:] - class_embeddings[1:,:,:]) **2, axis=2))
                 
                ##################################################################################################
                ############################################# SUMMARIES ##########################################
                ##################################################################################################
                 
                # Add summary
                summary_test = tf.Summary()
                summary_test.value.add(tag="%spos_neg_test_dist" % "", simple_value=np.mean(neg)-np.mean(pos))
                writer.add_summary(summary_test, global_batch_idx)
                writer.add_summary(summary_train, global_batch_idx)
                global_batch_idx += 1
 
                # Info
                print "Epoch: %06d batch: %03d loss: %06f dist_diff: %06f" % (epoch + 1, index, loss, np.mean(neg)-np.mean(pos))
                index += 1
        
        # Save model
        save_path = model.save_model(sess)
        print "Model saved in file: %s" % save_path

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=80)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=25)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    parser.add_argument("-m", "--margin", help="Triple loss margin value", type=float, required=False, default=0.2)
    args = vars(parser.parse_args())

    # train
    train_synthetic_features_extraction(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
                                        learning_rate=args["learning_rate"], margin=args["margin"], device=args["device"])

    # Print all settings at the end of learning
    print "Training params:"
    print "name          = ", args["name"]
    print "batch_size    = ", args["batch_size"]
    print "epochs        = ", args["epochs"]
    print "learning rate = ", args["learning_rate"]
    print "margin        = ", args["margin"]

if __name__ == "__main__":
    main(sys.argv[1:])
