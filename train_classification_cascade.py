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
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import train_with_hparam
import matplotlib.pyplot as plt
import deepclouds.defines as df
import sklearn.metrics as metrics
from sklearn.metrics import f1_score

import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

from sklearn.utils.extmath import softmax



CLOUD_SIZE = 1024
INPUT_CLOUD_DROPOUT_KEEP = 1.00

FEATURES_MODEL_1 = 'features_1/pointnet_with_euclidian_e25:1002018-09-09_11:54:31.ckpt'
CLASSIFIER_MODEL_1 = 'classifier_1/tset2018-09-12_09:24:38.ckpt'

FEATURES_MODEL_2 = 'features_2/pointnet_with_euclidian_finetune_f1_e125:2002018-09-11_06:58:02.ckpt'
CLASSIFIER_MODEL_2 = 'classifier_2/general_22018-09-12_10:02:12.ckpt'

FEATURES_MODEL_3 = 'features_3/pointnet_with_euclidian_finetune_f1_e200:2502018-09-12_14:28:43.ckpt'


CLASSIFIER_MODEL_LOAD = False
CLASSIFIER_MODEL_TRAIN = True
SYNTHETIC = False
READ_BLOCK_UNITS = [256]
ROTATE_CLOUDS_UP = True
SHUFFLE_CLOUDS = True
READ_BLOCK_METHOD = 'pointnet'
PROCESS_BLOCK_METHOD = 'max-pool'
#PROCESS_BLOCK_METHOD = 'attention-rnn'
SAMPLING_METHOD = 'fps'
#DISTANCE = 'cosine'
DISTANCE = 'euclidian'
CLASSIFIER_REG_WEIGHT = 0.001
CLASSIFIER_DROP_KEEPPROB = 0.70

INSTANCES_NUMBER = 10   # 889 / 10 = 88 batches      # train = 889 for each cloud / 127 = 7 batches
BATCHES = 88

# GOOD AFTER 100
ALREADY_GOOD_1 = ['airplane', 'bed', 'bottle', 'car', 'chair', 'laptop', 'sofa', 'toilet']  #0, 2, 5, 7, 8, 20, 30, 35
ALREADY_GOOD_2 = ['airplane', 'bathtub', 'bed', 'bottle', 'bowl', 'car', 'chair',
                  'cone', 'guitar', 'keyboard', 'lamp', 'laptop', 'sofa', 'toilet',
                  'mantel', 'monitor', 'person', 'piano', 'sink', 'table']



#    REG    DROP    TRAIN   TEST
#    0.0    0.25    89.4    75.8
#    0.0    0.35    95.4    81.7
#    0.0    0.42    96.0    82.1
#    0.0    0.50    96.7    82.6
#    0.0    0.62    96.8    82.6
#    0.0    0.75    97.0    82.3

#    0.0001 0.60    96.9    82.0
#    0.0003 0.60    96.8    82.5
#    0.001- 0.60    96.4    81.8

def train_classification(name, batch_size, epochs, learning_rate, device,
                         read_block_units, process_block_steps,
                         classifier_layers = [512, 256, 128, 64, 40]):
    """
    Train deepclouds classificator with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    if SYNTHETIC:
        data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE, permuted=SHUFFLE_CLOUDS,
                                          rotated_up=ROTATE_CLOUDS_UP, rotated_rand=False)
    else:
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE, clusterize=False)

    print data_gen.class_names
    exit()

    ##################################################################################################
    ################################## FEATURES EXTRACTION MODEL #####################################
    ##################################################################################################

    with tf.variable_scope("end-to-end-1"):
        with tf.device(device):
            features_model_1 = DeepCloudsModel(train=False,
                                               batch_size = batch_size,
                                               pointcloud_size = int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP),
                                               read_block_units = READ_BLOCK_UNITS,
                                               process_block_steps=process_block_steps,
                                               normalize_embedding=True, verbose=True,
                                               input_t_net=True, feature_t_net=True,
                                               read_block_method=READ_BLOCK_METHOD,
                                               process_block_method=PROCESS_BLOCK_METHOD,
                                               distance=DISTANCE)
            features_vars_1 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="end-to-end")
            features_model_saver_1 = tf.train.Saver(features_vars_1)
            
    with tf.variable_scope("end-to-end-2"):
        with tf.device(device):
            features_model_2 = DeepCloudsModel(train=False,
                                               batch_size = batch_size,
                                               pointcloud_size = int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP),
                                               read_block_units = READ_BLOCK_UNITS,
                                               process_block_steps=process_block_steps,
                                               normalize_embedding=True, verbose=True,
                                               input_t_net=True, feature_t_net=True,
                                               read_block_method=READ_BLOCK_METHOD,
                                               process_block_method=PROCESS_BLOCK_METHOD,
                                               distance=DISTANCE)
            features_vars_2 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="end-to-end-2")
            features_model_saver_2 = tf.train.Saver(features_vars_2)
            
    with tf.variable_scope("end-to-end-3"):
        with tf.device(device):
            features_model_3 = DeepCloudsModel(train=False,
                                               batch_size = batch_size,
                                               pointcloud_size = int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP),
                                               read_block_units = READ_BLOCK_UNITS,
                                               process_block_steps=process_block_steps,
                                               normalize_embedding=True, verbose=True,
                                               input_t_net=True, feature_t_net=True,
                                               read_block_method=READ_BLOCK_METHOD,
                                               process_block_method=PROCESS_BLOCK_METHOD,
                                               distance=DISTANCE)
            features_vars_3 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="end-to-end-3")
            features_model_saver_3 = tf.train.Saver(features_vars_3)

    ##################################################################################################
    ##################################### CLASSIFICATION MODEL #######################################
    ##################################################################################################

    with tf.device(device):
        classifier_model_1 = MLPClassifier('classifier-1', classifier_layers, batch_size, learning_rate,
                                           CLASSIFIER_REG_WEIGHT, CLASSIFIER_DROP_KEEPPROB)
        classifier_vars_1 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='classifier-1')
        classifier_model_saver_1 = tf.train.Saver(classifier_vars_1)
            
    with tf.variable_scope("classifier-2"):
        with tf.device(device):
            classifier_model_2 = MLPClassifier('classifier-2', classifier_layers, batch_size, learning_rate,
                                             CLASSIFIER_REG_WEIGHT, CLASSIFIER_DROP_KEEPPROB)
            classifier_vars_2 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='classifier-2')
            classifier_model_saver_2 = tf.train.Saver(classifier_vars_2)
            
    with tf.variable_scope("classifier-3"):
        with tf.device(device):
            classifier_model_3 = MLPClassifier('classifier-3', classifier_layers, batch_size, learning_rate,
                                             CLASSIFIER_REG_WEIGHT, CLASSIFIER_DROP_KEEPPROB)
            classifier_vars_3 = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='classifier-3')
            classifier_model_saver_3 = tf.train.Saver(classifier_vars_3)

    ##################################################################################################
    ############################################ SESSION #############################################
    ##################################################################################################

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        ##############################################################################################
        ######################################## INIT VARIABLES ######################################
        ##############################################################################################

        sess.run(tf.global_variables_initializer())
        
        # Load features and classification model
        features_model_saver_1.restore(sess, FEATURES_MODEL_1)
        classifier_model_saver_1.restore(sess, CLASSIFIER_MODEL_1)
        features_model_saver_2.restore(sess, FEATURES_MODEL_2)
        classifier_model_saver_2.restore(sess, CLASSIFIER_MODEL_2)
        features_model_saver_3.restore(sess, FEATURES_MODEL_3)

        ##############################################################################################
        ######################################## SUMMARY WRITER ######################################
        ##############################################################################################

        log_model_dir = os.path.join(df.LOGS_DIR, classifier_model_3.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
        #writer.add_graph(sess.graph)

        ##############################################################################################
        ##################################### EPOCH TRAINING LOOP ####################################
        ##############################################################################################

        accuracies = []
        global_batch_idx = 1
        checkpoint_skip_epochs = 25
        for epoch in range(epochs):

            ##########################################################################################
            ##################################### BATCH TRAINING LOOP ################################
            ##########################################################################################

            for batch_in_epoch_idx, (clouds, labels) in enumerate(tqdm(data_gen.generate_representative_batch_for_train(
                        instances_number=INSTANCES_NUMBER, shuffle_points=SHUFFLE_CLOUDS,
                        shuffle_clouds=True, jitter_points=True, rotate_pointclouds=False,
                        rotate_pointclouds_up=ROTATE_CLOUDS_UP, sampling_method=SAMPLING_METHOD), total=BATCHES)):

#             for clouds, labels in data_gen.generate_random_batch(train = True,
#                                                                  batch_size = batch_size,
#                                                                  shuffle_points=SHUFFLE_CLOUDS,
#                                                                  jitter_points=True,
#                                                                  rotate_pointclouds=False,
#                                                                  rotate_pointclouds_up=ROTATE_CLOUDS_UP,
#                                                                  sampling_method=SAMPLING_METHOD):

                # NEED FINETUNED!
                need_finetuned_idx = [idx for idx in range(len(labels)) if data_gen.class_names[labels[idx]] not in ALREADY_GOOD_2]                
                clouds = clouds[need_finetuned_idx]
                labels = labels[need_finetuned_idx]

                ##################################################################################################
                ######################################### GET EMBEDDING ##########################################
                ##################################################################################################

                # calc embeddings
                embeddings = []
                embedding_inputs = np.split(clouds, 5) # (40 - 20) * 10 = 200      200/5 = 40
                embedding_labels = np.split(labels, 5) # (40 - 8) * 10 = 320        320/8 = 40
                for embedding_input, embedding_label in zip(embedding_inputs, embedding_labels):
                    embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
                    embedding_input = np.expand_dims(embedding_input, axis=1)         
                    embeddings = sess.run(features_model_3.get_embeddings(), feed_dict={features_model_3.placeholder_embdg: embedding_input,
                                                                                  features_model_3.placeholder_is_tr : False})
                    embeddings = np.squeeze(embeddings)

                    # run optimizer
                    new_labels = sess.run(tf.one_hot(embedding_label, 40))
                    _, loss, summary_train = sess.run([classifier_model_3.get_optimizer(),
                                                       classifier_model_3.get_loss_function(),
                                                       classifier_model_3.get_summary()],
                                                       feed_dict={classifier_model_3.placeholder_embed: embeddings,
                                                                  classifier_model_3.placeholder_label: new_labels})


                global_batch_idx += 1
                writer.add_summary(summary_train, global_batch_idx)
        save_path = classifier_model_3.save_model(sess, name)
        print "Model saved in file: %s" % save_path

        ##################################################################################################
        ################################### CASCADE CLASSIFICATION #######################################
        ##################################################################################################
        
        print ("Calculating accuracy...")
        pred_labels_1 = []
        pred_labels_2 = []
        pred_labels_3 = []
        true_labels_0 = []
        pred_prblts_1 = []
        pred_prblts_2 = []
        pred_prblts_3 = []
        querry_idx = 32
        batch_idx = 0
        for clouds, labels in tqdm(data_gen.generate_random_batch(train=False, #16):# 400 test examples / 16 clouds = 25 batches
                                                             batch_size = 16,
                                                             shuffle_points=SHUFFLE_CLOUDS,
                                                             jitter_points=True,
                                                             rotate_pointclouds=False,
                                                             rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                             sampling_method=SAMPLING_METHOD), total=len(data_gen.pointclouds_train)/16):
    
#            if batch_idx + 16 <= querry_idx:
#                batch_idx += 16
#                continue
            
#           cloud_idx = querry_idx - batch_idx
#           print labels[cloud_idx]
#            np.save('cloud.npy', clouds[cloud_idx])
            
            temp_pred = []
            temp_prob = []
            for i in range(3):
            
                # REGENERATE
                clouds = np.expand_dims(clouds, axis=1)  #[B, X, N, 3]
                clouds = modelnet.ModelnetData._shuffle_points_in_batch(clouds)
                clouds = modelnet.ModelnetData._jitter_batch(clouds)
                batch_clouds = modelnet.ModelnetData._rotate_batch(clouds, random_axis=False)
                clouds = np.squeeze(clouds, axis=1)
            
                # padding
                clouds_padding = np.zeros((batch_size - 16, CLOUD_SIZE, 3), dtype=np.float)
                clouds_padded = np.concatenate((clouds, clouds_padding), axis=0)
                labels_padding = np.zeros((batch_size - 16), dtype=np.int)
                labels_padded = np.concatenate((labels, labels_padding), axis=0)
    
                # count embeddings
                clouds_padded = clouds_padded[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
                embedding_input = np.stack([clouds_padded], axis=1)
                
                # First extractor
                embeddings = sess.run(features_model_1.get_embeddings(),
                                      feed_dict={features_model_1.placeholder_embdg: embedding_input,
                                                 features_model_1.placeholder_is_tr : False})
                embeddings = np.squeeze(embeddings)
        
                # One hot
                labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
                pred = sess.run(classifier_model_1.get_classification_prediction(),
                                feed_dict={classifier_model_1.placeholder_embed: embeddings,
                                           classifier_model_1.placeholder_label: labels_padded_one_hot})

                # min is zero
                #pred_min = np.expand_dims(np.min(pred, axis=1), axis=1)
                #pred = pred - pred_min
                #pred_norm = np.expand_dims(np.sum(pred, axis=1), axis=1)
                #pred = pred / pred_norm
                pred = softmax(pred)
                
                predictions_1 = np.argmax(pred, axis=1)
                probabilities_1 = np.max(pred, axis=1)
                predictions_1 = predictions_1[:len(labels)]
                probabilities_1 = probabilities_1[:len(labels)]
                labels_1 = labels
                
                temp_pred.append(predictions_1)
                temp_prob.append(probabilities_1)
            
            temp_pred = np.stack(temp_pred)
            temp_prob = np.stack(temp_prob)
            cols = np.arange(16)
            rows = np.argmax(temp_prob, axis=0)
            pred_labels_1.append(temp_pred[rows, cols])
            pred_prblts_1.append(temp_prob[rows, cols])
                
            #pred_labels_1.append(predictions_1)
            true_labels_0.append(labels_1)
            #pred_prblts_1.append(probabilities_1)
    
            
            #hit_1 += float(np.sum(predictions == labels))
            #print "FIRST HIT = ", float(np.sum(predictions == labels))
            already_good_idcs_1 = [idx for idx in range(len(labels)) if data_gen.class_names[predictions_1[idx]] in ALREADY_GOOD_1]
            already_notg_idcs_1 = [idx for idx in range(len(labels)) if data_gen.class_names[predictions_1[idx]] not in ALREADY_GOOD_1]
            #pred_labels.append(predictions_1[already_good_idcs_1])
            #true_labels.append(labels_1[already_good_idcs_1])
            #print "FIRST HIT = ", float(np.sum(predictions_1 == labels_1)), len(labels_1), len(labels)
            
            ###################################################################################################
#             clouds_2 = clouds[already_notg_idcs_1]
#             labels_2 = labels[already_notg_idcs_1]
            clouds_2 = clouds
            labels_2 = labels
            
            # padding
            clouds_padding = np.zeros((batch_size - len(clouds_2), CLOUD_SIZE, 3), dtype=np.float)
            clouds_padded = np.concatenate((clouds_2, clouds_padding), axis=0)
            labels_padding = np.zeros((batch_size - len(clouds_2)), dtype=np.int)
            labels_padded = np.concatenate((labels_2, labels_padding), axis=0)
            
            # count embeddings
            clouds_padded = clouds_padded[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
            embedding_input = np.stack([clouds_padded], axis=1)
            
            # Second extractor
            embeddings = sess.run(features_model_2.get_embeddings(),
                                  feed_dict={features_model_2.placeholder_embdg: embedding_input,
                                             features_model_2.placeholder_is_tr : False})
            embeddings = np.squeeze(embeddings)
            
            # One hot
            labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
            pred = sess.run(classifier_model_2.get_classification_prediction(),
                            feed_dict={classifier_model_2.placeholder_embed: embeddings,
                                       classifier_model_2.placeholder_label: labels_padded_one_hot})
            
            # min is zero
#             pred_min = np.expand_dims(np.min(pred, axis=1), axis=1)
#             pred = pred - pred_min
#             pred_norm = np.expand_dims(np.sum(pred, axis=1), axis=1)
#             pred = pred / pred_norm
            pred = softmax(pred)
            
            predictions_2 = np.argmax(pred, axis=1)
            probabilities_2 = np.max(pred, axis=1)
            predictions_2 = predictions_2[:len(labels_2)]
            probabilities_2 = probabilities_2[:len(labels_2)]
            already_good_idcs_2 = [idx for idx in range(len(labels_2)) if data_gen.class_names[predictions_2[idx]] in ALREADY_GOOD_2]
            already_notg_idcs_2 = [idx for idx in range(len(labels_2)) if data_gen.class_names[predictions_2[idx]] not in ALREADY_GOOD_2]
            #pred_labels.append(predictions_2[already_good_idcs_2])
            #true_labels.append(labels_2[already_good_idcs_2])
            pred_labels_2.append(predictions_2)
            pred_prblts_2.append(probabilities_2)
            #print "SECOND HIT = ", float(np.sum(predictions_2 == labels_2)), len(labels_2), len(labels)

            ###################################################################################################
#             clouds_3 = clouds_2[already_notg_idcs_2]
#             labels_3 = labels_2[already_notg_idcs_2]
            clouds_3 = clouds_2
            labels_3 = labels_2
            
            # padding
            clouds_padding = np.zeros((batch_size - len(clouds_3), CLOUD_SIZE, 3), dtype=np.float)
            clouds_padded = np.concatenate((clouds_3, clouds_padding), axis=0)
            labels_padding = np.zeros((batch_size - len(clouds_3)), dtype=np.int)
            labels_padded = np.concatenate((labels_3, labels_padding), axis=0)
            
            # count embeddings
            clouds_padded = clouds_padded[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
            embedding_input = np.stack([clouds_padded], axis=1)
            
            # Second extractor
            embeddings = sess.run(features_model_3.get_embeddings(),
                                  feed_dict={features_model_3.placeholder_embdg: embedding_input,
                                             features_model_3.placeholder_is_tr : False})
            embeddings = np.squeeze(embeddings)
            
            # One hot
            labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
            pred = sess.run(classifier_model_3.get_classification_prediction(),
                            feed_dict={classifier_model_3.placeholder_embed: embeddings,
                                       classifier_model_3.placeholder_label: labels_padded_one_hot})
            
#             # min is zero
#             pred_min = np.expand_dims(np.min(pred, axis=1), axis=1)
#             pred = pred - pred_min
#             pred_norm = np.expand_dims(np.sum(pred, axis=1), axis=1)
#             pred = pred / pred_norm
            pred = softmax(pred)
            
            
            predictions_3 = np.argmax(pred, axis=1)
            probabilities_3 = np.max(pred, axis=1)
            predictions_3 = predictions_3[:len(labels_3)]
            probabilities_3 = probabilities_3[:len(labels_3)]
            #pred_labels.append(predictions_3)
            #true_labels.append(labels_3)
            pred_labels_3.append(predictions_3)
            pred_prblts_3.append(probabilities_3)
            #print "SECOND HIT = ", float(np.sum(predictions_2 == labels_2)), len(labels_2), len(labels)       
            
#             print predictions_1, probabilities_1, labels
#             print predictions_2, probabilities_2, labels_2
#             print predictions_3, probabilities_3, labels_3
#              
#             l = np.concatenate((predictions[already_good_idcs_1], predictions_2[already_good_idcs_2], predictions_3))
#             t = np.concatenate((labels[already_good_idcs_1], labels_2[already_good_idcs_2], labels_3))
#             hit = float(sum(l == t))
#             all = len(t)
#             print "ACC = ", hit/all
#             if hit/all < 0.8:
#                 exit()
        true_labels_0 = np.concatenate(true_labels_0)
        pred_labels_1 = np.concatenate(pred_labels_1)
        pred_labels_2 = np.concatenate(pred_labels_2)
        pred_labels_3 = np.concatenate(pred_labels_3)
        pred_prblts_1 = np.concatenate(pred_prblts_1)
        pred_prblts_2 = np.concatenate(pred_prblts_2)
        pred_prblts_3 = np.concatenate(pred_prblts_3)
        
        np.save('true_0.npy', true_labels_0)
        np.save('pred_1.npy', pred_labels_1)
        np.save('prob_1.npy', pred_prblts_1)
        np.save('pred_2.npy', pred_labels_2)
        np.save('prob_2.npy', pred_prblts_2)
        np.save('pred_3.npy', pred_labels_3)
        np.save('prob_3.npy', pred_prblts_3)
        
        sum = 0.
        for idx, true_label in enumerate(true_labels_0):
            l1 = true_labels_0[idx] ==  pred_labels_1[idx]
            l2 = true_labels_0[idx] ==  pred_labels_2[idx]
            l3 = true_labels_0[idx] ==  pred_labels_3[idx]
            if (l1 or l2 or l3):
                sum = sum + 1
        
        print "MAX ACC:", sum / len(true_labels_0)
        exit()
            

#     
#         # accuracy and confusion matrix
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        hit = float(sum(pred_labels == true_labels))
        all = len(true_labels)
        confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
        

        class_acc = []
        print ("==========================TRAIN ACCURACIES==========================")
        for class_idx, class_name in enumerate(data_gen.class_names):
            
            true_pos = float(np.sum(confusion_matrix[class_idx]))                                            # sum in rows
            flse_pos = float(np.sum(confusion_matrix[:, class_idx]) - confusion_matrix[class_idx, class_idx])  # sum in cols without querry
            flse_neg = float(np.sum(confusion_matrix[class_idx]) - confusion_matrix[class_idx, class_idx])     # sum in rows without querry
            
            precsn = true_pos / (true_pos + flse_pos)
            recall = true_pos / (true_pos + flse_neg)
            f1_scr = 2 * precsn * recall / (precsn + recall)
            
            acc = np.sum(float(confusion_matrix[class_idx, class_idx]) / np.sum(confusion_matrix[class_idx]))
            class_acc.append(acc)
            print (class_name, acc, f1_scr)
            
        print ("TRAN_HIT = ", hit, "TRAN_ALL = ", all, "TRAIN_ACC" , hit/all)
        print ("TRAIN_CLASS_ACC = ", np.mean(class_acc))
        exit()

        ##################################################################################################
        ################################### CLASSIFICATION ACCURACY ######################################
        ##################################################################################################

        train_hit, train_all, train_conf_mat = calc_acc(train=True, data_gen=data_gen, batch_size=batch_size,
                                                        sess=sess, features_model=features_model, classifier_model=classifier_model)

        train_acc = train_hit / train_all

        test_hit, test_all, test_conf_mat = calc_acc(train=False, data_gen=data_gen, batch_size=batch_size,
                                                     sess=sess, features_model=features_model, classifier_model=classifier_model)

        test_acc = test_hit / test_all
        
#             print ("==========================TRAIN ACCURACIES UNNORM==========================")
#             for class_idx, class_name in enumerate(data_gen.class_names):
#                 acc = np.sum(float(train_conf_mat[class_idx, class_idx]) / np.sum(train_conf_mat[class_idx]))
#                 valmax1 = 0
#                 valmax2 = 0
#                 if class_idx > 0:
#                     argmax1 = np.argmax(train_conf_mat[class_idx,:class_idx])
#                     valmax1 = np.max(train_conf_mat[class_idx,:class_idx])
#                 if class_idx < class_idx-2:
#                     argmax2 = np.argmax(train_conf_mat[class_idx,class_idx+1:])
#                     valmax2 = np.max(train_conf_mat[class_idx,class_idx+1:])
#                 if valmax1 == valmax2 and valmax1 == 0:
#                     print (class_idx, class_name, acc)
#                 elif valmax1 >= valmax2:
#                     print (class_idx, class_name, 'accuracy:', acc, 'predicted_as:', argmax1, 'how_many:', valmax1)
#                 elif valmax1 < valmax2:
#                     print (class_idx, class_name, 'accuracy:', acc, 'predicted_as:', argmax2, 'how_many:', valmax2)

        class_acc = []
        print ("==========================TRAIN ACCURACIES==========================")
        for class_idx, class_name in enumerate(data_gen.class_names):
            
            true_pos = float(np.sum(train_conf_mat[class_idx]))                                            # sum in rows
            flse_pos = float(np.sum(train_conf_mat[:, class_idx]) - train_conf_mat[class_idx, class_idx])  # sum in cols without querry
            flse_neg = float(np.sum(train_conf_mat[class_idx]) - train_conf_mat[class_idx, class_idx])     # sum in rows without querry
            
            precsn = true_pos / (true_pos + flse_pos)
            recall = true_pos / (true_pos + flse_neg)
            f1_scr = 2 * precsn * recall / (precsn + recall)
            
            acc = np.sum(float(train_conf_mat[class_idx, class_idx]) / np.sum(train_conf_mat[class_idx]))
            class_acc.append(acc)
            print (class_name, acc, f1_scr)

        class_acc_test = []
        print ("==========================TEST ACCURACIES==========================")
        for class_idx, class_name in enumerate(data_gen.class_names):
            
            true_pos = float(np.sum(test_conf_mat[class_idx]))                                            # sum in rows
            flse_pos = float(np.sum(test_conf_mat[:, class_idx]) - test_conf_mat[class_idx, class_idx])  # sum in cols without querry
            flse_neg = float(np.sum(test_conf_mat[class_idx]) - test_conf_mat[class_idx, class_idx])     # sum in rows without querry
            
            precsn = true_pos / (true_pos + flse_pos)
            recall = true_pos / (true_pos + flse_neg)
            f1_scr = 2 * precsn * recall / (precsn + recall)
            
            acc = np.sum(float(test_conf_mat[class_idx, class_idx]) / np.sum(test_conf_mat[class_idx]))
            class_acc_test.append(acc)
            print (class_name, acc, f1_scr)

        print ("TRAN_HIT = ", train_hit, "TRAN_ALL = ", train_all, "TRAIN_ACC" , train_hit/train_all)
        print ("TRAIN_CLASS_ACC = ", np.mean(class_acc))
        print ("TEST_HIT = ", test_hit, "TEST_ALL = ", test_all, "TEST_ACC" , test_hit/test_all)
        print ("TEST_CLASS_ACC = ", np.mean(class_acc_test))


        plot_confusion_matrix(train_conf_mat, classes=data_gen.class_names, normalize=False, title='Train confusion matrix unnorm')
        plot_confusion_matrix(train_conf_mat, classes=data_gen.class_names, normalize=True, title='Train confusion matrix norm')
        plot_confusion_matrix(test_conf_mat, classes=data_gen.class_names, normalize=False, title='Test confusion matrix unnorm')
        plot_confusion_matrix(test_conf_mat, classes=data_gen.class_names, normalize=True, title='Test confusion matrix norm')

#   #              if epoch == epochs - 1: 
#                 if epoch == 0: 
#                     plot_confusion_matrix(train_conf_mat, classes=data_gen.class_names, normalize=False, title='Train confusion matrix unnorm')
#                     plot_confusion_matrix(train_conf_mat, classes=data_gen.class_names, normalize=True, title='Train confusion matrix norm')
#                     plot_confusion_matrix(test_conf_mat, classes=data_gen.class_names, normalize=False, title='Test confusion matrix unnorm')
#                     plot_confusion_matrix(test_conf_mat, classes=data_gen.class_names, normalize=True, title='Test confusion matrix norm')
# 
#                 ##################################################################################################
#                 ############################################# SUMMARIES ##########################################
#                 ##################################################################################################
# 
#                 accuracies.append(test_acc)
#                 summary_test = tf.Summary()
#                 summary_test.value.add(tag="%stest_classification_accuracy" % "", simple_value=test_acc)
#                 writer.add_summary(summary_test, global_batch_idx)
#                 writer.add_summary(summary_train, global_batch_idx)
#                 print "Epoch: %06d batch: %03d loss: %06f train_accuracy: %06f test_accuracy: %06f" % (epoch + 1, global_batch_idx, loss, train_acc, test_acc)
# 
#                 ##################################################################################################
#                 ################################## SAVE CLASSIFICATION MODEL #####################################
#                 ##################################################################################################
# 
          #if (epoch+1) % checkpoint_skip_epochs == 0:

        # Save model
        save_path = classifier_model.save_model(sess, name)
        print "Model saved in file: %s" % save_path
        #print "MAX ACC = ", np.max(accuracies)

#         else:
#             accuracies = []
#             global_votes = []
#             global_labels = []
#             for _ in range(1):
#                 # Get test embeddings
#                 hit = {i : 0. for i in range(40)}
#                 all = {i : 0. for i in range(40)}
#                 test_batch_size = 8
#                 batch_votes = []
#                 batch_labels = []
#                 sys.stdout.write("Classify modelnet")
#                 sys.stdout.flush()
#                 for clouds, labels in data_gen.generate_random_batch(False, batch_size=test_batch_size,
#                                                                      shuffle_points=False,
#                                                                      rotate_pointclouds_up=False):# 400 test examples / 16 clouds = 25 batches
#                     
#                     sys.stdout.write(".")
#                     sys.stdout.flush()
# 
#                     # padding
#                     clouds_padding = np.zeros((batch_size - test_batch_size, CLOUD_SIZE, 3), dtype=np.float)
#                     clouds_padded = np.concatenate((clouds, clouds_padding), axis=0)
#                     labels_padding = np.zeros((batch_size - test_batch_size), dtype=np.int)
#                     labels_padded = np.concatenate((labels, labels_padding), axis=0)
# 
#                     # count embeddings
#                     embedding_input = embedding_input[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
#                     embedding_input = np.stack([clouds_padded], axis=1)
#                     embeddings = sess.run(features_model.get_embeddings(),
#                                                feed_dict={features_model.placeholder_embdg: embedding_input,
#                                                           features_model.placeholder_is_tr : False})
# 
#                     # One hot
#                     embeddings = np.squeeze(embeddings)
#                     labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
#                     predictions = sess.run(classifier_model.get_classification_prediction(),
#                                            feed_dict={classifier_model.placeholder_embed: embeddings,
#                                                       classifier_model.placeholder_label: labels_padded_one_hot})
# 
#                     # accuracy 
#                     predictions_args = np.argmax(predictions, axis=1)
#                     predictions_args = predictions_args[:len(labels)]
#                     for cloud_idx in range(len(clouds)):
#                         hit[labels[cloud_idx]] = hit[labels[cloud_idx]] + (predictions_args[cloud_idx] == labels[cloud_idx])
#                         all[labels[cloud_idx]] = all[labels[cloud_idx]] + 1
#                     #hit = hit + sum(predictions_args == labels)
#                     #all = all + len(labels)
#                     for idx in range(len(labels)):
#                         batch_votes.append(predictions[idx])
#                         batch_labels.append(labels[idx])
#     
#                 asd_best = []
#                 asd_all = []
#                 print "\n"
#                 for cloud_idx in range(40):
#                     print "Accuracy: ", cloud_idx, "  :  ", hit[cloud_idx]/all[cloud_idx]
#                     asd_all.append(hit[cloud_idx]/all[cloud_idx])
#                     if cloud_idx not in [3, 15, 32, 33, 38, 39]:
#                         asd_best.append(hit[cloud_idx]/all[cloud_idx])
# 
#                 print "ALL      :", np.mean(asd_all)
#                 print "BEST ONLY:", np.mean(asd_best)
#                 print "ACCURACY :", np.sum(hit.values())/np.sum(all.values())
#                 print "AVG CLASS ACCURACY :", np.mean(np.array(hit.values(),np.float32)/np.array(all.values(), np.float32))
# 
#                 exit()        
#                 accuracies.append(hit / all)
#                 global_votes.append(batch_votes)
#                 print "GLOBALVOTES", len(global_votes), len(global_votes[0])
#                 global_labels.append(batch_labels)
#                 if len(global_votes) == 2:
#                     break
#             np.save("votes.npy", np.transpose(np.squeeze(np.array(global_votes))))
#             np.save("labels.npy", np.transpose(np.squeeze(np.array(global_labels))))
#             print "AVG ACCURACY = ", np.mean(accuracies)

def calc_acc(train, data_gen, batch_size, sess, features_model, classifier_model):

    all_predcs = []
    all_labels =[]
    for clouds, labels in data_gen.generate_random_batch(train, #16):# 400 test examples / 16 clouds = 25 batches
                                                         batch_size = 16,
                                                         shuffle_points=SHUFFLE_CLOUDS,
                                                         jitter_points=True,
                                                         rotate_pointclouds=False,
                                                         rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                         sampling_method=SAMPLING_METHOD):

        # padding
        clouds_padding = np.zeros((batch_size - 16, CLOUD_SIZE, 3), dtype=np.float)
        clouds_padded = np.concatenate((clouds, clouds_padding), axis=0)
        labels_padding = np.zeros((batch_size - 16), dtype=np.int)
        labels_padded = np.concatenate((labels, labels_padding), axis=0)

        # count embeddings
        clouds_padded = clouds_padded[:, :int(CLOUD_SIZE*INPUT_CLOUD_DROPOUT_KEEP), :]  # input dropout
        embedding_input = np.stack([clouds_padded], axis=1)
        embeddings = sess.run(features_model.get_embeddings(),
                              feed_dict={features_model.placeholder_embdg: embedding_input,
                                         features_model.placeholder_is_tr : False})
        embeddings = np.squeeze(embeddings)


        # One hot
        labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
        pred = sess.run(classifier_model.get_classification_prediction(),
                        feed_dict={classifier_model.placeholder_embed: embeddings,
                                   classifier_model.placeholder_label: labels_padded_one_hot})


        # predictions
        predictions = np.argmax(pred, axis=1)
        predictions = predictions[:len(labels)]
        all_predcs.append(predictions)
        all_labels.append(labels)

    # accuracy and confusion matrix
    all_predcs = np.concatenate(all_predcs)
    all_labels = np.concatenate(all_labels)
    hit = float(sum(all_predcs == all_labels))
    all = len(all_predcs)
    confusion_matrix = metrics.confusion_matrix(all_labels, all_predcs)

    # ret acc
    return hit, all, confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Args:
        confusion_matrix (np.ndarray of size CxC): Confusion matrix with unnormalized values.
        classes (list of str): List of classes names.
        normalize (boolean): Should I normalize confusion matrix?
        cmap (plt.cm): color map.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest', aspect='auto', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        if confusion_matrix[i, j] != 0:
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=40)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=25)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-r", "--read_block_units", help="Read block units", type=int, required=False, default=512)
    parser.add_argument("-p", "--process_block_steps", help="Process block steps", type=int, required=False, default=4)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    args = vars(parser.parse_args())

    # train
    train_classification(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
                         learning_rate=args["learning_rate"], device=args["device"],
                         read_block_units=[args["read_block_units"]], process_block_steps=[args["process_block_steps"]])

    # Print all settings at the end of learning
    print "Classification model:"
    print "name                 = ", args["name"]
    print "batch_size           = ", args["batch_size"]
    print "epochs               = ", args["epochs"]
    print "learning rate        = ", args["learning_rate"]
    print "read_block_units     = ", args["read_block_units"]
    print "process_block_steps  = ", args["process_block_steps"]

    # Send end of training e-mail 
#    email_sender = train_with_hparam.EmailSender()
#    email_sender.send(["daniel.koguciuk@gmail.com"], "Work Done!", "")


if __name__ == "__main__":
    main(sys.argv[1:])
