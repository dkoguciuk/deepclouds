#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import datetime
from tqdm import tqdm
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepclouds.defines as df
import sklearn.metrics as metrics
import scipy.spatial.distance as cos_dist
from timeit import default_timer as timer
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel


CLOUD_SIZE = 1024
DISTANCE = 'cosine'
#DISTANCE = 'euclidian'
SAMPLING_METHOD = 'fps'
LOAD_MODEL = True
CALC_DIST = False
SYNTHETIC = False
READ_BLOCK_UNITS = [256]
ROTATE_CLOUDS_UP = True
SHUFFLE_CLOUDS = True
READ_BLOCK_METHOD = 'pointnet'
#PROCESS_BLOCK_METHOD = 'attention-rnn'
PROCESS_BLOCK_METHOD = 'max-pool'
INSTANCES_NUMBER = 80      # train = 889 for each cloud / 127 = 7 batches
BATCHES = 11               # train = 889 for each cloud / 127 = 7 batches
REGULARIZATION_WEIGHT = 0.0

# LOGGING
MODEL_SAVE_AFTER_EPOCHS = 100


def find_semi_hard_triples_to_train(embeddings, labels, margin):
    
    def find_nearest_idx(array, value):
        return (np.abs(array - value)).argmin()
    
    non_zero = 0
    hard_positives_indices = []
    hard_negatives_indices = []
    for cloud_idx in range(embeddings.shape[0]):
        # calc distances
        if DISTANCE == 'euclidian':
            distances = np.linalg.norm(embeddings - embeddings[cloud_idx], axis=-1)
        elif DISTANCE == 'cosine':
            numerator = np.squeeze(np.sum(np.multiply(embeddings[cloud_idx][0], embeddings), axis=-1))
            denominator = np.linalg.norm(embeddings[cloud_idx][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1))
            if denominator.shape[0] != np.count_nonzero(denominator):
                print (denominator.shape[0], np.count_nonzero(denominator),
                       np.count_nonzero(np.linalg.norm(embeddings[cloud_idx][0])),
                       np.count_nonzero(np.squeeze(np.linalg.norm(embeddings, axis=-1))))
            distances = 1 - np.divide(numerator, denominator)
        
        # find hard positive
        class_idx = labels[cloud_idx]
        class_indices = np.squeeze(np.argwhere(labels == class_idx))
        hard_positives_indices.append(class_indices[np.argmax(np.take(distances, class_indices))])
        positive_distance = distances[hard_positives_indices[-1]]
        
        # Help vars
        other_indices = np.squeeze(np.argwhere(labels != class_idx))  # Find indices of all other class instances
        negatives_distances = np.take(distances, other_indices)  # Find distances to all other class distances
        negatives_distances[negatives_distances <= positive_distance] = float('inf')  # Ifinity distance when it's smaller than positive dist plus eps
        negative_idx = np.argmin(negatives_distances)  # Smallest
        hard_negatives_indices.append(other_indices[negative_idx])  # Find negative embedding index 
    
        if max(positive_distance + margin - negatives_distances[negative_idx], 0) > 0:
            non_zero += 1
    
    return hard_positives_indices, hard_negatives_indices, non_zero

def calc_inner_class_distance(sess, data_gen, model, margin):
    # Inner class distance
    inner_class_distances = []
    for class_idx, class_name in enumerate(data_gen.class_names):
#         if class_idx != 39:
#             continue
        class_embeddings = []
        for clouds, labels in data_gen.generate_representative_batch_for_train(
                                                                     instances_number=INSTANCES_NUMBER,
                                                                     shuffle_points=SHUFFLE_CLOUDS,
                                                                     shuffle_clouds=True,
                                                                     jitter_points=True,
                                                                     rotate_pointclouds=False,
                                                                     rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                                     sampling_method=SAMPLING_METHOD):
             
            embeddings = []
            embedding_inputs = np.split(clouds, INSTANCES_NUMBER) #200/40 = 5
            for embedding_input in embedding_inputs:
                embedding_input = np.expand_dims(embedding_input, axis=1)         
                embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input,
                                                                              model.margin : [margin],
                                                                              model.placeholder_is_tr : True}))
            embeddings = np.concatenate(embeddings)
            
                        # INSTANCES = 40
            class_idcs = np.where(labels == class_idx)[0]
            #labels = labels[class_idcs]
            #clouds = clouds[class_idcs]
            embeddings = embeddings[class_idcs]
            class_embeddings.append(embeddings)

        class_embeddings = np.concatenate(class_embeddings)
#         np.save('bench_embeddings.npy', class_embeddings)
#         exit()
        distances = []
        if DISTANCE == 'cosine':
            for querry_embedding in class_embeddings:
                pos_num  = np.squeeze(np.sum(np.multiply(querry_embedding, class_embeddings), axis=-1))
                neg_num  = np.linalg.norm(querry_embedding) * np.squeeze(np.linalg.norm(class_embeddings, axis=-1))
                distances.append(1 - np.divide(pos_num, neg_num))
#    `    NOT TESTED
#         elif DISTANCE = 'euclidian':
#             for querry_embedding in class_embeddings:
#                 distances.append(np.linalg.norm(querry_embedding - class_embeddings, axis=1))
        else:
            print 'DON\'T KNOW THIS DISTANCE METRIC'
            exit()
        distances = np.stack(distances)
        distance = np.mean(distances[np.triu_indices(len(class_embeddings), 1)])
        inner_class_distances.append(distance)
        print ('CLASS ', class_name, ' distance: ', distance)
     
    # Scale things up
    inner_class_weights = np.array(inner_class_distances / np.sum(inner_class_distances), dtype=np.float32)
    return inner_class_weights

def train_features_extraction(synthetic, name, batch_size, epochs,
                                        learning_rate=None, gradient_clip=None,
                                        margin=None, margin_growth=None,
                                        device=None):
    """
    Train deepclouds with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    ##################################################################################################
    ########################################## DATA GENERATOR ########################################
    ##################################################################################################
                
    if synthetic:
        data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE, permuted=SHUFFLE_CLOUDS,
                                          rotated_up=ROTATE_CLOUDS_UP, rotated_rand=False)
    else:
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE, clusterize=False)

    ##################################################################################################
    ######################################### DEEPCLOUDS MODEL #######################################
    ##################################################################################################
    
    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            model = DeepCloudsModel(train=True,
                                    batch_size=data_gen.CLASSES_COUNT, pointcloud_size=CLOUD_SIZE,
                                    read_block_units=READ_BLOCK_UNITS, process_block_steps=[4],
                                    learning_rate=learning_rate, gradient_clip=gradient_clip,
                                    normalize_embedding=True, verbose=True,
                                    input_t_net=True, feature_t_net=True,
                                    distance=DISTANCE, read_block_method=READ_BLOCK_METHOD,
                                    process_block_method=PROCESS_BLOCK_METHOD,
                                    regularization_weight=REGULARIZATION_WEIGHT)

    if LOAD_MODEL:
        features_model_saver = tf.train.Saver()
        
    ##################################################################################################
    ######################################### TENSORFLOW STUFF #######################################
    ################################################################################################## 

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
 
        ##################################################################################################
        ########################################## INIT VARIABLES ########################################
        ##################################################################################################

        sess.run(tf.global_variables_initializer()) 
        
        if LOAD_MODEL:
            features_model_saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
            
        ##################################################################################################
        ########################################### LOG OPTIONS ##########################################
        ##################################################################################################
        
        log_model_dir = os.path.join(df.LOGS_DIR, model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
 
        histograms = []
        variables_names = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="end-to-end")
        for var in variables_names:
            histograms.append(tf.summary.histogram(var.name, var))
        hist_summary = tf.summary.merge(histograms)
        
        ##################################################################################################
        ########################################### EPOCHS LOOP ##########################################
        ##################################################################################################

        for epoch in range(epochs):
        
            ##################################################################################################
            ########################################## CLASS WEIGHTS #########################################
            ##################################################################################################  
            
#             # calc class weights
#             if epoch % 1 == 0:
#                 class_weights = calc_inner_class_distance(sess, data_gen, model, margin)
#                 print ("WEIGHTS COMPUTED...")
            class_weights = np.ones(40, dtype=np.float32) / 40
           # class_weights = np.ones(data_gen.CLASSES_COUNT, dtype=np.float32) / data_gen.CLASSES_COUNT

            ##################################################################################################
            ########################################### BATCHES LOOP #########################################
            ##################################################################################################
        
            for batch_in_epoch_idx, (clouds, labels) in enumerate(data_gen.generate_representative_batch_for_train(
                        instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
                        shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
                        rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD)):#, total=BATCHES)):
                
                clouds_a = clouds[labels==4]
                labels_a = labels[labels==4]
                clouds_b = clouds[labels==38]
                labels_b = labels[labels==38]
                clouds = np.concatenate((clouds_a, clouds_b))
                labels = np.concatenate((labels_a, labels_b))

                ##################################################################################################
                ################################# FIND SEMI HARD TRIPLETS TO LEARN ###############################
                ##################################################################################################

                # calc embeddings
                embeddings = []
                clouds_input = np.split(clouds, 4)  # 80*2 / 4 = 40
                for cloud_input in clouds_input:
                    embedding_input = np.expand_dims(cloud_input, axis=1)       
                    embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
                embeddings = np.concatenate(embeddings)
                    
                
                # Find hard examples to train on
                pos_indices, neg_indices, non_zero = find_semi_hard_triples_to_train(embeddings, labels, margin)
                  
                # Create triples to train
                pos_clouds = np.copy(clouds)
                pos_clouds = pos_clouds[pos_indices, ...]
                neg_clouds = np.copy(clouds)
                neg_clouds = neg_clouds[neg_indices, ...]
                training_inputs = np.stack([clouds, pos_clouds, neg_clouds], axis=1)
                  
                ##################################################################################################
                ############################################# TRAIN ##############################################
                ##################################################################################################
 
                training_inputs = np.split(training_inputs, 4)
                training_labels = np.split(labels, 4)
                for training_input, training_label in zip(training_inputs, training_labels):
                    global_batch_idx, _, training_loss, summary_train, training_non_zero = sess.run(
                            [model.global_step, model.get_optimizer(), model.get_loss_function(),
                             model.get_summary(), model.non_zero_triplets], feed_dict={model.placeholder_train: training_input,
                                                                                       model.margin : [margin],
                                                                                       model.placeholder_is_tr : True,
                                                                                       model.placeholder_label : training_label,
                                                                                       model.classes_learning_weights : class_weights})

            ##################################################################################################
            ############################################# LOG ################################################
            ##################################################################################################
 
#                if global_batch_idx % (BATCHES) == 0:

                # Loggin summary
                summary_log = tf.Summary()
                summary_log.value.add(tag="%smargin" % "", simple_value=margin)
                summary_log.value.add(tag="%snon_zero" % "", simple_value=training_non_zero)

                # pos/neg dist
                if CALC_DIST:# and (epoch % MODEL_SAVE_AFTER_EPOCHS == MODEL_SAVE_AFTER_EPOCHS - 1):
                    pos_man, neg_man = test_features_extraction(data_gen, model, sess, partial_score=synthetic)
                    summary_log.value.add(tag="%spos_neg_test_dist" % "", simple_value=neg_man - pos_man)
                    print "Epoch: %06d batch: %03d loss: %09f dist_diff: %09f non_zero: %03d margin: %09f learning_rate: %06f" % (epoch + 1, batch_in_epoch_idx, training_loss, neg_man - pos_man, training_non_zero, margin, learning_rate)
                else:
                    print "Epoch: %06d batch: %03d loss: %09f non_zero: %03d margin: %09f learning_rate: %06f" % (epoch + 1, batch_in_epoch_idx, training_loss, training_non_zero, margin, learning_rate)
               
                # Variables histogram
                summary_histograms = sess.run(hist_summary)
                writer.add_summary(summary_histograms, global_batch_idx)             
    
                # Write summary
                writer.add_summary(summary_log, global_batch_idx)
                writer.add_summary(summary_train, global_batch_idx)

            ##################################################################################################
            ########################################## SAVE MODEL ############################################
            ##################################################################################################
        
            if epoch % MODEL_SAVE_AFTER_EPOCHS == MODEL_SAVE_AFTER_EPOCHS - 1:
                save_path = model.save_model(sess, name)
                print "Model saved in file: %s" % save_path

def test_features_extraction(data_gen, model, sess, partial_score=True):
    """
    Train deepclouds with synthetic data.
    """

    # Get test embeddings
    batches = 0
    test_embeddings = { k : [] for k in range(40)}
    for clouds, labels in data_gen.generate_random_batch(False, 40, sampling_method=SAMPLING_METHOD):  # 400 test examples / 80 clouds = 5 batches

        # count embeddings
        test_embedding_input = np.stack([clouds], axis=1)
        test_embedding = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: test_embedding_input,
                                                                     model.placeholder_is_tr : False})
        test_embedding = np.squeeze(test_embedding, axis=1)
    
        # add embeddings
        for cloud_idx in range(labels.shape[0]):
            test_embeddings[labels[cloud_idx]].append(test_embedding[cloud_idx])
            
        # not the whole dataset
        if partial_score:
            batches += 1
            if batches == 5:
                break
    
    # Convert to numpy
    class_embeddings = []
    for k in range(40):
        class_embeddings.append(test_embeddings[k])
    
    # Calc distances between every embedding in one class 
    pos_man = []
    for class_idx in range(len(class_embeddings)):
        positive_dist_class = []
        for instance_idx_1 in range(len(class_embeddings[class_idx])):
            for instance_idx_2 in range(len(class_embeddings[class_idx])):
                if instance_idx_1 != instance_idx_2:
                    if DISTANCE == 'euclidian':
                        positive_dist_class.append(np.linalg.norm(class_embeddings[class_idx][instance_idx_1] - 
                                                                  class_embeddings[class_idx][instance_idx_2]))
                    elif DISTANCE == 'cosine':
                        positive_dist_class.append(cos_dist.cosine(class_embeddings[class_idx][instance_idx_1],
                                                                   class_embeddings[class_idx][instance_idx_2]))
        pos_man.append(positive_dist_class)
    pos_man_flat = [item for sublist in pos_man for item in sublist]
    
    # Calc distances between every embedding in one class and every other class
    neg_man = []
    for class_idx_1 in range(len(class_embeddings)):
        negative_dist_class = []
        for class_idx_2 in range(len(class_embeddings)):
            if class_idx_1 != class_idx_2:
                for instance_idx_1 in range(len(class_embeddings[class_idx_1])):
                    for instance_idx_2 in range(len(class_embeddings[class_idx_2])):
                        if instance_idx_1 != instance_idx_2:
                            if DISTANCE == 'euclidian':
                                negative_dist_class.append(np.linalg.norm(class_embeddings[class_idx_1][instance_idx_1] - 
                                                                          class_embeddings[class_idx_2][instance_idx_2]))
                            elif DISTANCE == 'cosine':
                                negative_dist_class.append(cos_dist.cosine(class_embeddings[class_idx_1][instance_idx_1],
                                                                           class_embeddings[class_idx_2][instance_idx_2]))
        neg_man.append(negative_dist_class)
    neg_man_flat = [item for sublist in neg_man for item in sublist]    
    
    return np.mean(pos_man_flat), np.mean(neg_man_flat)

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=80)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=25)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=False, default=0.0001)
    parser.add_argument("-g", "--gradient_clip", help="Max gradient value, gradient clipping disabled when smaller than zero", type=float, required=False, default=10.0)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    parser.add_argument("-m", "--margin", help="Triple loss margin value", type=float, required=False, default=0.2)
    parser.add_argument("-t", "--margin_growth", help="Allow margin growth in time", type=bool, required=False, default=False)
    args = vars(parser.parse_args())

    # train
    train_features_extraction(SYNTHETIC,
                              args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
                              learning_rate=args["learning_rate"], gradient_clip=args["gradient_clip"],
                              margin=args["margin"], device=args["device"], margin_growth=args["margin_growth"])

    # Print all settings at the end of learning
    print "Training params:"
    print "name          = ", args["name"]
    print "batch_size    = ", args["batch_size"]
    print "epochs        = ", args["epochs"]
    print "learning rate = ", args["learning_rate"]
    print "margin        = ", args["margin"]

if __name__ == "__main__":
    main(sys.argv[1:])
