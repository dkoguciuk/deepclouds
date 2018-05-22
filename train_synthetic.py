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
import deepclouds.defines as df
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import RNNBidirectionalModel, MLPModel

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
        negatives_distances = np.take(distances, other_indices)
        negatives_distances[negatives_distances <= np.max(np.take(distances, class_indices))] = float('inf')        
        hard_negatives_indices.append(other_indices[np.argmin(negatives_distances)])

    return hard_positives_indices, hard_negatives_indices

def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_semi_hard_triples_to_train(embeddings, labels):
    hard_positives_indices = []
    hard_negatives_indices = []
    MARGIN = 0.2
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
        #negatives_distances[negatives_distances <= positive_distance] = float('inf')                           # Ifinity distance when it's smaller than positive dist
        #negative_idx = find_nearest_idx(negatives_distances, positive_distance + MARGIN/10)                    # Find index of elem in the half of margin range
        
        negatives_distances[negatives_distances <= (positive_distance + 1e-6)] = float('inf')                   # Ifinity distance when it's smaller than positive dist plus eps
        negative_idx = np.argmin(negatives_distances)                                                           # Smallest
        
        hard_negatives_indices.append(other_indices[negative_idx])                                              # Find negative embedding index 
    
    return hard_positives_indices, hard_negatives_indices 

def train_synthetic_features_extraction(name, batch_size, epochs, learning_rate, margin, device,
                                        rnn_layer_sizes=[CLOUD_SIZE * 3, 40], mlp_layer_sizes=[2*40*CLOUD_SIZE, 40]):
    """
    Train deepclouds with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model = RNNBidirectionalModel(rnn_layer_sizes, mlp_layer_sizes,
                                      batch_size, learning_rate, margin, normalize_embedding=True,
                                      pointcloud_size=CLOUD_SIZE)

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
            for clouds, labels in data_gen.generate_representative_batch(batch_size):

                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input})
                
                # Find hard examples to train on
                pos_indices, neg_indices = find_semi_hard_triples_to_train(embeddings, labels)
                
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
                #if global_batch_idx % summary_skip_batch == 0:
                writer.add_summary(summary, global_batch_idx)
                global_batch_idx += 1

                # Info
                print "Epoch: %06d batch: %03d loss: %06f" % (epoch + 1, index, loss)
                index += 1
        
        # Save model
        save_path = model.save_model(sess)
        print "Model saved in file: %s" % save_path

def train_synthetic_classification(name, batch_size, epochs, learning_rate, device):
    """
    Train deepclouds classificator with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model_features = RNNBidirectionalModel([CLOUD_SIZE * 3, 40], [2*40*CLOUD_SIZE, 40],
                                               batch_size, learning_rate, 0.2, pointcloud_size=CLOUD_SIZE)  

    # Saver
    saver = tf.train.Saver()

    # Define model
    with tf.device(device):
        model_classifier = MLPClassifier([40, 40, 40, 40], batch_size, learning_rate)

    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:

        # Run the initialization
        sess.run(tf.global_variables_initializer())
         
        # saver    
        saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
        
        # Logs
        log_model_dir = os.path.join(df.LOGS_DIR, model_classifier.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
        writer.add_graph(sess.graph)
        
        # Do the training loop
        global_batch_idx = 1
        summary_skip_batch = 1
        for epoch in range(epochs):

            # loop for all batches
            index = 1
            for clouds, labels in data_gen.generate_train_batch(batch_size):
            
                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(model_features.get_embeddings(), feed_dict={model_features.placeholder_embdg: embedding_input})

                # run optimizer
                new_labels = sess.run(tf.one_hot(labels, 40))
                _, loss, pred, summary = sess.run([model_classifier.get_optimizer(), model_classifier.get_loss_function(),
                                             model_classifier.get_classification_prediction(), model_classifier.get_summary()],
                                   feed_dict={model_classifier.placeholder_embed: embeddings,
                                              model_classifier.placeholder_label: new_labels})
 
                
 
                #print pred[0], labels[0]
                #print "PREDICTED: %02d REAL: %02d" % (np.argmax(pred, axis=1)[0], labels[0])
                #exit()

                # Tensorboard vis
                if global_batch_idx % summary_skip_batch == 0:
                    writer.add_summary(summary, global_batch_idx)
                global_batch_idx += 1

                # Info
                print "Epoch: %06d batch: %03d loss: %06f" % (epoch + 1, index, loss)
                index += 1

        # Save model
        save_path = model_classifier.save_model(sess)
        print "Model saved in file: %s" % save_path

def save_embeddings(device):
    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model_features = RNNBidirectionalModel([CLOUD_SIZE * 3, 40], [2*40*CLOUD_SIZE, 40],
                                               1, 0.1, 0.2, pointcloud_size=CLOUD_SIZE)  

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
        saver.restore(sess, tf.train.latest_checkpoint('models'))
        
        # Do the training loop
        global_batch_idx = 1
        summary_skip_batch = 1

        # loop for all batches
        index = 1
        for clouds, labels in data_gen.generate_representative_batch(1):
        
            # count embeddings
            embedding_input = np.stack([clouds], axis=1)
            embeddings = sess.run(model_features.get_embeddings(), feed_dict={model_features.placeholder_embdg: embedding_input})
            
            data_filapath = "embeddings/data_%04d.npy" % index
            label_filapath = "embeddings/label_%04d.npy" % index
            np.save(data_filapath, embeddings)
            np.save(label_filapath, labels)
            index = index+1

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
#    train_synthetic_classification(args["name"], batch_size=args["batch_size"], epochs=args["epochs"], learning_rate=args["learning_rate"], device=args["device"])
#    save_embeddings(device=args["device"])

    # Print all settings at the end of learning
    print "MLP-basic model:"
    print "name          = ", args["name"]
    print "batch_size    = ", args["batch_size"]
    print "epochs        = ", args["epochs"]
    print "learning rate = ", args["learning_rate"]
    print "margin        = ", args["margin"]

if __name__ == "__main__":
    main(sys.argv[1:])
