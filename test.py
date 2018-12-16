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
INPUT_CLOUD_DROPOUT_KEEP = 0.75

#DISTANCE = 'cosine'
DISTANCE = 'euclidian'
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

INSTANCES_NUMBER = 10   # 720 / 10 = 72 batches
BATCHES = 72
REGULARIZATION_WEIGHT = 0.01

# LOGGING
MODEL_SAVE_AFTER_EPOCHS = 25


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
            denominator = (np.linalg.norm(embeddings[cloud_idx][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1)) + 1e-9)
#             if denominator.shape[0] != np.count_nonzero(denominator):
#                 print ('Numerator: ', numerator.shape, 'denominator: ', denominator.shape,
#                        'non zero denominator :', np.count_nonzero(denominator),
#                        'non zero querry: ', np.count_nonzero(np.linalg.norm(embeddings[cloud_idx][0])),
#                        'non zero allemb: ', np.count_nonzero(np.squeeze(np.linalg.norm(embeddings, axis=-1))))
#                 #exit()
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


def train_features_extraction(synthetic=False, name='test', batch_size=80, epochs=1,
                                        learning_rate=0.0001, gradient_clip=None,
                                        margin=0.2, margin_growth=False,
                                        device="/device:GPU:0"):
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
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE, clusterize=False, dev_fraction=0.2)

    ##################################################################################################
    ######################################### DEEPCLOUDS MODEL #######################################
    ##################################################################################################
    
    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            model = DeepCloudsModel(train=True,
                                    batch_size=data_gen.CLASSES_COUNT, pointcloud_size=int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP),
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
            
            class_weights = np.ones(data_gen.CLASSES_COUNT, dtype=np.float32) / data_gen.CLASSES_COUNT
            
#             ##################################################################################################
#             ####################################### TRAIN EMBEDDINGS #########################################
#             ##################################################################################################
#             
#             embeddings_train = {}
#             for i in range(40):
#                 embeddings_train[i] = []
#                 
# #             for batch_in_epoch_idx, (clouds, labels) in enumerate(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'train',
# #                 instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
# #                 shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
# #                 rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=720)):
# #                 
# #                 # calc embeddings
# #                 embeddings = []
# #                 embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
# #                 for embedding_input in embedding_inputs:
# #                     embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
# #                     embedding_input = np.expand_dims(embedding_input, axis=1)         
# #                     embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
# #                 embeddings = np.concatenate(embeddings)
# #             
# #                 # Add distances
# #                 for i in range(embeddings.shape[0]):
# #                     embeddings_train[labels[i]].append(embeddings[i])
#             
#             
#             for i in range(40):
#                 #embeddings_train[i] = np.concatenate(embeddings_train[i])
#                 #np.save('train_class_embeddings_' + str(i) + '.npy', embeddings_train[i])
#                 embeddings_train[i] = np.load('train_class_embeddings_' + str(i) + '.npy')
#                 print i, embeddings_train[i].shape
#             
#             ##################################################################################################
#             ####################################### CHECK CLASS SIMILARITY ########################################
#             ##################################################################################################         
#                       
#             hit = 0.
#             all = 0.   
#             for batch_in_epoch_idx, (clouds, labels) in enumerate(tqdm(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'devel',
#                         instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
#                         shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
#                         rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=240), total=24)):
#              
#                 # calc embeddings
#                 embeddings = []
#                 embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
#                 for embedding_input in embedding_inputs:
#                     embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
#                     embedding_input = np.expand_dims(embedding_input, axis=1)         
#                     embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
#                 embeddings = np.concatenate(embeddings)
#    
#                 # Add distances
#                 for i in range(embeddings.shape[0]):
#                     
#                     class_distances = []
#                     for j in range(40):
#                         # calc distances
#                         numerator = np.squeeze(np.sum(np.multiply(embeddings[i][0], embeddings_train[j]), axis=-1))
#                         denominator = (np.linalg.norm(embeddings[i][0]) * np.squeeze(np.linalg.norm(embeddings_train[j], axis=-1)) + 1e-9)
#                         distances = 1 - np.divide(numerator, denominator)
#                         class_distances.append(np.mean(distances))
#                         #print 'querry_class', labels[i], 'target_class', j, 'distance', np.mean(distances)
#                         
#                     
#                     if labels[i] == np.argmin(class_distances):
#                         print "HIT!"
#                         hit += 1
#                     else:
#                         for j in range(40):
#                             print 'querry_class', labels[i], 'target_class', j, 'distance', class_distances[j]
#                         print "MISS!", j
#                         exit()
#                     all += 1
#             
#             print "DEVEL ACC = ", hit/all
#             exit()
#             
#             ##################################################################################################
#             ####################################### FIND HARD CLASSES ########################################
#             ##################################################################################################         
#                           
#             class_distances = {}
#             for i in range(40):
#                 class_distances[i] = {}
#                 for j in range(40):
#                     class_distances[i][j] = 0.
#               
#             for batch_in_epoch_idx, (clouds, labels) in enumerate(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'devel',
#                         instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
#                         shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
#                         rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=240)):
#               
#                 # calc embeddings
#                 embeddings = []
#                 embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
#                 for embedding_input in embedding_inputs:
#                     embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
#                     embedding_input = np.expand_dims(embedding_input, axis=1)         
#                     embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
#                 embeddings = np.concatenate(embeddings)
#     
#                 # Add distances
#                 for i in range(embeddings.shape[0]):
#                     # calc distances
#                     numerator = np.squeeze(np.sum(np.multiply(embeddings[i][0], embeddings), axis=-1))
#                     denominator = (np.linalg.norm(embeddings[i][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1)) + 1e-9)
#                     distances = 1 - np.divide(numerator, denominator)
#                      
#                     # add distances
#                     for j in range(embeddings.shape[0]):
#                         class_distances[labels[i]][labels[j]] += distances[j]
#   
#             hard_classes = []
#             for i in range(40):
#                 hard_classes.append(np.argmin([class_distances[i][j] if i != j else np.inf for j in range(40)]))
#                 print 'for epoch', epoch, 'class', i, 'has a hard class of', hard_classes[-1]
 
#             for batch_in_epoch_idx, (clouds, labels) in enumerate(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'devel',
#                         instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
#                         shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
#                         rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=240)):
#              
#                 # calc embeddings
#                 embeddings = []
#                 embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
#                 for embedding_input in embedding_inputs:
#                     embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
#                     embedding_input = np.expand_dims(embedding_input, axis=1)         
#                     embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
#                 embeddings = np.concatenate(embeddings)
#    
#                 # Add distances
#                 for i in range(embeddings.shape[0]):
#                     
#                     #if labels[i] == 39:
#                     if i == 39:
#                     
#                         # calc distances
#                         numerator = np.squeeze(np.sum(np.multiply(embeddings[i][0], embeddings), axis=-1))
#                         denominator = (np.linalg.norm(embeddings[i][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1)) + 1e-9)
#                         distances = 1 - np.divide(numerator, denominator)
#                         
#                         # add distances
#                         for j in range(embeddings.shape[0]):
#                             
#                             if labels[j] == hard_classes[labels[i]]:
#                                 print 'for class', labels[i], 'and hard class', labels[j], 'distance', distances[j]
#                             if labels[j] == labels[i] and i != j:
#                                 print 'for class', labels[i], 'and same class', labels[j], 'distance', distances[j]
#                                 
#                         exit()
#                             #class_distances[labels[i]][labels[j]] += distances[j]
#             
#  
#             exit()


            ##################################################################################################
            ########################################### BATCHES LOOP #########################################
            ##################################################################################################
        
            for batch_in_epoch_idx, (clouds, labels) in enumerate(tqdm(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'train',
            #for batch_in_epoch_idx, (clouds, labels) in enumerate(tqdm(data_gen.generate_representative_batch_for_train(
                        instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
                        shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
                        rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=720), total=BATCHES)):
                        #rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD), total=BATCHES)):


#                 ##################################################################################################
#                 ############################### FIND HARD CLASSES THIS IS WORKING! ###############################
#                 ##################################################################################################         
#                                 
#                 class_distances = {}
#                 for i in range(40):
#                     class_distances[i] = {}
#                     for j in range(40):
#                         class_distances[i][j] = 0.
#                     
#                 for _, (clouds_dev, labels_dev) in enumerate(data_gen.generate_representative_batch_for_train_or_devel(train_or_devel = 'devel',
#                             instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
#                             shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
#                             rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD, augment=240)):
#                     
#                     # calc embeddings
#                     embeddings = []
#                     embedding_inputs = np.split(clouds_dev, INSTANCES_NUMBER)
#                     for embedding_input in embedding_inputs:
#                         embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
#                         embedding_input = np.expand_dims(embedding_input, axis=1)         
#                         embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
#                     embeddings = np.concatenate(embeddings)
#           
#                     # Add distances
#                     for i in range(embeddings.shape[0]):
#                         # calc distances
#                         numerator = np.squeeze(np.sum(np.multiply(embeddings[i][0], embeddings), axis=-1))
#                         denominator = (np.linalg.norm(embeddings[i][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1)) + 1e-9)
#                         distances = 1 - np.divide(numerator, denominator)
#                            
#                         # add distances
#                         for j in range(embeddings.shape[0]):
#                             class_distances[labels_dev[i]][labels_dev[j]] += distances[j]
#         
#                 hard_classes = []
#                 for i in range(40):
#                     hard_classes.append(np.argmin([class_distances[i][j] if i != j else np.inf for j in range(40)]))


                ##################################################################################################
                ################################# FIND SEMI HARD TRIPLETS TO LEARN ###############################
                ##################################################################################################

                # calc embeddings
                embeddings = []
                embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
                for embedding_input in embedding_inputs:
                    embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
                    embedding_input = np.expand_dims(embedding_input, axis=1)         
                    embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input}))
                embeddings = np.concatenate(embeddings)
                
                # Find hard examples to train on
                #pos_indices, neg_indices, non_zero = find_semi_hard_triples_to_train(embeddings, labels, margin)
                def find_nearest_idx(array, value):
                    return (np.abs(array - value)).argmin()
                
                # Find hard examples to train on
                anc_indices = []
                pos_indices = []
                neg_indices = []
                for i in range(embeddings.shape[0]):
                    
                    # calc distances
                    numerator = np.squeeze(np.sum(np.multiply(embeddings[i][0], embeddings), axis=-1))
                    denominator = (np.linalg.norm(embeddings[i][0]) * np.squeeze(np.linalg.norm(embeddings, axis=-1)) + 1e-9)
                    distances = 1 - np.divide(numerator, denominator)
                    # pos_and neg indices
                    class_idx = np.squeeze(np.argwhere(labels == labels[i]))
                    #other_idx = np.squeeze(np.argwhere(labels == hard_classes[labels[i]]))                    
                    other_idx = np.squeeze(np.argwhere(labels != labels[i]))
                    
                    # hardest
                    for positive_idx in class_idx:                                                      # For every other positive cloud
                        if i == positive_idx:
                            continue
                        positive_distance = distances[positive_idx]                                     # Take all negative cloud indices
                        negatives_distances = np.take(distances, other_idx)                             # Take all negative cloud distances
                        negatives_distances[negatives_distances <= positive_distance] = float('inf')    # Zero out all closer negatives
                        negative_distance = np.min(negatives_distances)                                 # Find closest negative
                        if negative_distance - positive_distance < margin:                              # If negative is closer than margin add indices to learn on
                            anc_indices.append(i)
                            pos_indices.append(positive_idx)
                            neg_indices.append(other_idx[np.argmin(negatives_distances)])
                    
#                     pos_hardest = class_idx[np.argmax(np.take(distances, class_idx))]
#                     pos_indices.append(pos_hardest)
#                     positive_distance = distances[pos_hardest]
# 
#                     # Help vars
#                     negatives_distances = np.take(distances, other_idx)  # Find distances to all other class distances
#                     negatives_distances[negatives_distances <= positive_distance] = float('inf')  # Ifinity distance when it's smaller than positive dist plus eps
#                     negative_idx = np.argmin(negatives_distances)  # Smallest
#                     neg_indices.append(other_idx[negative_idx])  # Find negative embedding index 

                # numpy
                anc_indices = np.array(anc_indices)
                pos_indices = np.array(pos_indices)
                neg_indices = np.array(neg_indices)
                
                # Divisionable by INSTANCES_NUMBER
                if len(anc_indices) % 40:
                    multp = np.floor(len(anc_indices) / 40)
                    count = int(multp*40)
                    indcs = np.random.permutation(np.arange(len(anc_indices)))[:count]
                    anc_indices = anc_indices[indcs, ...]
                    pos_indices = pos_indices[indcs, ...]
                    neg_indices = neg_indices[indcs, ...]
                  
                # Create triples to train
                anc_clouds = np.copy(clouds)
                anc_clouds = anc_clouds[anc_indices, ...]
                pos_clouds = np.copy(clouds)
                pos_clouds = pos_clouds[pos_indices, ...]
                neg_clouds = np.copy(clouds)
                neg_clouds = neg_clouds[neg_indices, ...]
                training_inputs = np.stack([anc_clouds, pos_clouds, neg_clouds], axis=1)
                  
                ##################################################################################################
                ############################################# TRAIN ##############################################
                ##################################################################################################
 
                #print len(training_inputs)
                splits = int(len(training_inputs) / 40)
                training_inputs = np.split(training_inputs, splits) # INSTANCES_NUMBER
                #training_labels = np.split(labels, splits)
                #for training_input, training_label in zip(training_inputs, training_labels):
                for training_input in training_inputs:
                    training_input = training_input[:, :, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
                    global_batch_idx, _, training_loss, training_pos, training_neg, summary_train, training_non_zero = sess.run(
                        [model.global_step, model.get_optimizer(), model.get_loss_function(), model.pos_dist, model.neg_dist,
                         model.get_summary(), model.non_zero_triplets], feed_dict={model.placeholder_train: training_input,
                                                                                   model.margin : [margin],
                                                                                   model.placeholder_is_tr : True,
                                                                                   #model.placeholder_label : training_label,
                                                                                   model.classes_learning_weights : class_weights})
                    if np.isnan(training_pos).any():
                        print "POS_DIST", training_pos
                    if np.isnan(training_neg).any():  
                        print "NEG_DIST", training_neg
            
            ##################################################################################################
            ############################################# LOG ################################################
            ##################################################################################################

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

if __name__ == "__main__":
    main(sys.argv[1:])
