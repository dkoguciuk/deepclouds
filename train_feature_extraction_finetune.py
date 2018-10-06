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
import scipy.spatial.distance as cos_dist
from timeit import default_timer as timer
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

WEIGHTS = [0.012022766, 0.035261128, 0.07107232, 0.23799036, 0.42551684, 0.07248676, 0.02016846, 0.03325137, 0.113675214, 0.05530229, 0.05136407, 0.08216307, 0.18014286, 0.27773413, 0.34478766, 0.47695652, 0.10179748, 0.013541839, 0.015065977, 0.06563898, 0.016787777, 0.14145115, 0.07306109, 0.22392283, 0.10848778, 0.103102416, 0.19784562, 0.20684634, 0.023361124, 0.046396278, 0.04400247, 0.07186616, 0.060257427, 0.17564489, 0.06396944, 0.041337732, 0.43234345, 0.14644158, 0.38690782, 0.18048736]
WEIGHTS_NORMED = np.array(WEIGHTS / np.sum(WEIGHTS), dtype=np.float32)

CLOUD_SIZE = 1024
DISTANCE = 'cosine'
SAMPLING_METHOD = 'fps'
LOAD_MODEL = True
CALC_DIST = True
SYNTHETIC = False
READ_BLOCK_UNITS = [256]
ROTATE_CLOUDS_UP = True
SHUFFLE_CLOUDS = True
checkpoint_skip_epochs = 25
READ_BLOCK_METHOD = 'pointnet'
#PROCESS_BLOCK_METHOD = 'attention-rnn'
PROCESS_BLOCK_METHOD = 'max-pool'
INSTANCES_NUMBER = 5	# train = 200, test = 60
REGULARIZATION_WEIGHT = 0.01

#CLASS_A = 11 # CURTAIN
#CLASS_B = 33 # TABLE

def plot_bar(e):
    bins = 64
    x = np.arange(0, e.shape[0] / bins)
    y = np.average(np.split(e, e.shape[0] / bins), axis=-1)
    plt.bar(x, y, 1, align='center')

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

def train_features_extraction(synthetic, name, batch_size, epochs,
                                        learning_rate=None, gradient_clip=None,
                                        margin=None, margin_growth=None,
                                        device=None):
    """
    Train deepclouds with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    if synthetic:
        data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE, permuted=SHUFFLE_CLOUDS,
                                          rotated_up=ROTATE_CLOUDS_UP, rotated_rand=False)
    else:
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            model = DeepCloudsModel(train=True,
                                    batch_size=40, pointcloud_size=CLOUD_SIZE,
                                    read_block_units=READ_BLOCK_UNITS, process_block_steps=[4],
                                    learning_rate=learning_rate, gradient_clip=gradient_clip,
                                    normalize_embedding=True, verbose=True,
                                    input_t_net=True, feature_t_net=True,
                                    distance=DISTANCE, read_block_method=READ_BLOCK_METHOD,
                                    process_block_method=PROCESS_BLOCK_METHOD,
                                    regularization_weight=REGULARIZATION_WEIGHT,
                                    classes_learning_weights=WEIGHTS_NORMED)
                                      
    # PRINT PARAM NO
#    total_parameters = 0
#    for variable in tf.trainable_variables():
#        # shape is an array of tf.Dimension
#        shape = variable.get_shape()
#        print(variable.name)
#        print(shape)
#        print(len(shape))
#        variable_parameters = 1
#        for dim in shape:
#            print(dim)
#            variable_parameters *= dim.value
#        print(variable_parameters)
#        total_parameters += variable_parameters
#    print(total_parameters)
#    exit()

    if LOAD_MODEL:
        features_model_saver = tf.train.Saver()

    if batch_size % model.CLASSES_COUNT != 0:
        print "Batch size should be multiple of CLASSES_COUNT = ", model.CLASSES_COUNT
        exit()

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
 
        # Run the initialization
        sess.run(tf.global_variables_initializer()) 
        
        if LOAD_MODEL:
            features_model_saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
            #features_model_saver.restore(sess, tf.train.latest_checkpoint('models_temp'))
        
        log_model_dir = os.path.join(df.LOGS_DIR, model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
        # writer.add_graph(sess.graph)
 
        histograms = []
        variables_names = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope="end-to-end")
        for var in variables_names:
            histograms.append(tf.summary.histogram(var.name, var))
        hist_summary = tf.summary.merge(histograms)

#        # Inner class distance
#        for class_idx, class_name in enumerate(data_gen.class_names):
#            for clouds, labels in data_gen.generate_representative_batch(train=False,
#                                                                         instances_number=INSTANCES_NUMBER,
#                                                                         shuffle_points=SHUFFLE_CLOUDS,
#                                                                         jitter_points=True,
#                                                                         rotate_pointclouds=False,
#                                                                         rotate_pointclouds_up=ROTATE_CLOUDS_UP,
#                                                                         sampling_method=SAMPLING_METHOD): 
#                # get class clouds
#                class_idcs = np.where(labels == class_idx)[0]
#                labels = labels[class_idcs]
#                clouds = clouds[class_idcs]
#
#               # count embeddings
#                embedding_input = np.expand_dims(clouds, axis=1)         
#                embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input,
#                                                                     model.margin : [margin],
#                                                                     model.placeholder_is_tr : True})
#            
#                embeddings = np.squeeze(embeddings)
#                distances = []
#                for querry_embedding in embeddings:
#                    pos_num  = np.squeeze(np.sum(np.multiply(querry_embedding, embeddings), axis=-1))
#                    neg_num  = np.linalg.norm(querry_embedding) * np.squeeze(np.linalg.norm(embeddings, axis=-1))
#                    distances.append(1 - np.divide(pos_num, neg_num))
#                distances = np.stack(distances)
#                distance = np.mean(distances[np.triu_indices(len(embeddings), 1)])
#                print (class_name, distance)
#        exit()

        # Do the training loop
        non_zeros = []
        summary_skip_batch = 25
        for epoch in range(epochs):
 
            time_0 = timer()
 
            # loop for all batches
            epoch_batch_idx = 0
            for clouds, labels in data_gen.generate_representative_batch(train=True,
                                                                         instances_number=INSTANCES_NUMBER,
                                                                         shuffle_points=SHUFFLE_CLOUDS,
                                                                         jitter_points=True,
                                                                         rotate_pointclouds=False,
                                                                         rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                                         sampling_method=SAMPLING_METHOD):

#                class_a_idcs = np.where(labels == CLASS_A)[0]
#                class_b_idcs = np.where(labels == CLASS_B)[0]
#                class_idcs = np.concatenate((class_a_idcs, class_b_idcs), axis=0)
#                labels = labels[class_idcs]
#                clouds = clouds[class_idcs]

                # time_1 = timer()

                ##################################################################################################
                ################################# FIND SEMI HARD TRIPLETS TO LEARN ###############################
                ##################################################################################################
                 
#                # count embeddings
#                embedding_input = np.expand_dims(clouds, axis=1)         
#                embeddings = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input,
#                                                                         model.margin : [margin],
#                                                                         model.placeholder_is_tr : True})

                # count embeddings
                embeddings = []
                embedding_inputs = np.split(clouds, INSTANCES_NUMBER)
                for embedding_input in embedding_inputs:
                    embedding_input = np.expand_dims(embedding_input, axis=1)         
                    embeddings.append(sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: embedding_input,
                                                                                  model.margin : [margin],
                                                                                  model.placeholder_is_tr : True}))
                embeddings = np.concatenate(embeddings)

                # time_2 = timer()
                 
                # Find hard examples to train on
                pos_indices, neg_indices, non_zero = find_semi_hard_triples_to_train(embeddings, labels, margin)
                  
                # Create triples to train
                pos_clouds = np.copy(clouds)
                pos_clouds = pos_clouds[pos_indices, ...]
                neg_clouds = np.copy(clouds)
                neg_clouds = neg_clouds[neg_indices, ...]
                training_inputs = np.stack([clouds, pos_clouds, neg_clouds], axis=1)

#                for instance_idx in range(INSTANCES_NUMBER*2):
#                    pos_num  = np.squeeze(np.sum(np.multiply(embeddings[instance_idx], embeddings[pos_indices[instance_idx]]), axis=-1))
#                    pos_den = np.linalg.norm(embeddings[instance_idx]) * np.linalg.norm(embeddings[pos_indices[instance_idx]])
#                    neg_num  = np.squeeze(np.sum(np.multiply(embeddings[instance_idx], embeddings[neg_indices[instance_idx]]), axis=-1))
#                    neg_den = np.linalg.norm(embeddings[instance_idx]) * np.linalg.norm(embeddings[neg_indices[instance_idx]])
#                    print ("POS DIST = ", 1 - np.divide(pos_num, pos_den), "NEG DIST = ", 1 - np.divide(neg_num, neg_den))
                
                # time_3 = timer()

                ##################################################################################################
                ############################################ VISUALIZATION #######################################
                ##################################################################################################

#                 # non_zero
#                 if epoch_batch_idx == 1:
#                     print "ACTUAL   = ", labels[0]
#                     print "POSITIVE = ", pos_indices[0], labels[pos_indices[0]], np.linalg.norm(embeddings[0]-embeddings[pos_indices[0]], axis=0) 
#                     print "NEGATIVE = ", neg_indices[0], labels[neg_indices[0]], np.linalg.norm(embeddings[0]-embeddings[neg_indices[0]], axis=0)
#                     plt.figure()
#                     plot_bar(embeddings[0])
#                     plt.figure()
#                     plot_bar(embeddings[pos_indices[0]])
#                     plt.figure()
#                     plot_bar(embeddings[neg_indices[0]])
#                     plt.figure()
#                     plot_bar(embeddings[0] - embeddings[pos_indices[0]])
#                     plt.figure()
#                     plot_bar(embeddings[0] - embeddings[neg_indices[0]])
#                     plt.show()
#                 exit()
                  
                ##################################################################################################
                ############################################# TRAIN ##############################################
                ##################################################################################################

 
                # run optimizer
#                global_batch_idx, _, training_loss, summary_train, training_non_zero = sess.run(
#                    [model.global_step, model.get_optimizer(), model.get_loss_function(),
#                     model.get_summary(), model.non_zero_triplets], feed_dict={model.placeholder_train: training_inputs,
#                                                                               model.margin : [margin],
#                                                                               model.placeholder_label : labels,
#                                                                               model.placeholder_is_tr : True})

                # run optimizer
                training_losses = []
                training_non_zeros = []
                training_inputs = np.split(training_inputs, INSTANCES_NUMBER)
                training_labels = np.split(labels, INSTANCES_NUMBER)
                for training_input, training_label in zip(training_inputs, training_labels):
                    global_batch_idx, _, training_loss, summary_train, training_non_zero = sess.run(
                        [model.global_step, model.get_optimizer(), model.get_loss_function(),
                         model.get_summary(), model.non_zero_triplets], feed_dict={model.placeholder_train: training_input,
                                                                                   model.margin : [margin],
                                                                                   model.placeholder_label : training_label,
                                                                                   model.placeholder_is_tr : True})
                    training_losses.append(training_loss)
                    training_non_zeros.append(training_non_zero)

                ##################################################################################################
                ############################## GET POS-NEG DIST DIFF ON TEST SET #################################
                ##################################################################################################
 
                if global_batch_idx % (summary_skip_batch*INSTANCES_NUMBER) == 1:
#                if True:

                    # pos/neg dist
                    if CALC_DIST:
                        pos_man, neg_man = test_features_extraction(data_gen, model, sess, partial_score=synthetic)
                        
                    # time_5 = timer()
                      
                    ##################################################################################################
                    ############################################# SUMMARIES ##########################################
                    ##################################################################################################
                      
                    # Variables histogram
                    summary_histograms = sess.run(hist_summary)
                    writer.add_summary(summary_histograms, global_batch_idx)             
   
                    # Add summary
                    summary_test = tf.Summary()
                    if CALC_DIST:
                        summary_test.value.add(tag="%spos_neg_test_dist" % "", simple_value=neg_man - pos_man)
                    summary_test.value.add(tag="%smargin" % "", simple_value=margin)
                    summary_test.value.add(tag="%snon_zero" % "", simple_value=training_non_zero)
                    writer.add_summary(summary_test, global_batch_idx)
                    writer.add_summary(summary_train, global_batch_idx)
                    if CALC_DIST:
                        print "Epoch: %06d batch: %03d loss: %09f dist_diff: %09f non_zero: %03d margin: %09f learning_rate: %06f" % (epoch + 1, epoch_batch_idx, np.mean(training_losses), neg_man - pos_man, np.mean(training_non_zeros), margin, learning_rate)
                    else:
                        print "Epoch: %06d batch: %03d loss: %09f non_zero: %03d margin: %09f learning_rate: %06f" % (epoch + 1, epoch_batch_idx, np.mean(training_losses), np.mean(training_non_zeros), margin, learning_rate)
                    
                    # time_6 = timer()
                    # print "generator    : ", (time_1 - time_0)
                    # print "embedding    : ", (time_2 - time_1)
                    # print "triplets     : ", (time_3 - time_2)
                    # print "optimier     : ", (time_4 - time_3)
                    # print "pos_neg_dist : ", (time_5 - time_4)
                    # print "summary      : ", (time_6 - time_5)
                    # exit()

                # inc
                epoch_batch_idx += 1
        
            if epoch % checkpoint_skip_epochs == checkpoint_skip_epochs - 1:

                # Save model
                save_path = model.save_model(sess, name)
                print "Model saved in file: %s" % save_path

def test_features_extraction(data_gen, model, sess, partial_score=True):
    """
    Train deepclouds with synthetic data.
    """

#     start = datetime.datetime.now()

    # Get test embeddings
    batches = 0
    test_embeddings = { k : [] for k in range(40)}
    for clouds, labels in data_gen.generate_random_batch(False, 40, sampling_method=SAMPLING_METHOD):  # 400 test examples / 80 clouds = 5 batches

        # count embeddings
        test_embedding_input = np.stack([clouds], axis=1)
        test_embedding = sess.run(model.get_embeddings(), feed_dict={model.placeholder_embdg: test_embedding_input,
                                                                     model.placeholder_is_tr : False})
    
        # add embeddings
        for cloud_idx in range(labels.shape[0]):
            test_embeddings[labels[cloud_idx]].append(test_embedding[cloud_idx].tolist())
            
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
    
#     end = datetime.datetime.now()
#     print (end-start).seconds, (end-start).microseconds
    
    
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
