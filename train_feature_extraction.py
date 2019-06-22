#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
"""

import os
import sys
import argparse
import importlib
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import deepclouds.defines as df
import deepclouds.data_provider as data_provider
from deepclouds.model import SiamesePointClouds


sys.path.append('settings')
sys.path.append('deepclouds/backbones')


def train_features_extraction(name, setting=None):

    # Reset
    tf.reset_default_graph()

    ##################################################################################################
    ########################################## DATA GENERATOR ########################################
    ##################################################################################################

    data_gen = data_provider.ModelNet40(pointcloud_size=setting.points_num, clusterize=False)

    ##################################################################################################
    ######################################### DEEPCLOUDS MODEL #######################################
    ##################################################################################################
    
    with tf.variable_scope("end-to-end"):
        with tf.device(setting.device):
            model = SiamesePointClouds(setting)
        
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
            
        ##################################################################################################
        ########################################### LOG OPTIONS ##########################################
        ##################################################################################################
        
        log_model_dir = os.path.join(df.LOGS_DIR, model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name), sess.graph)
 
        histograms = []
        variables_names = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="end-to-end")
        for var in variables_names:
            histograms.append(tf.summary.histogram(var.name, var))
        hist_summary = tf.summary.merge(histograms)
        save_model_index = 0
        save_summary_index = 0
        
        ##################################################################################################
        ########################################### EPOCHS LOOP ##########################################
        ##################################################################################################

        pbar = tqdm(total=setting.training_iterations)
        while True:

            ##################################################################################################
            ########################################### BATCHES LOOP #########################################
            ##################################################################################################

            for clouds, labels in data_gen.generate_batch_c_i(classes_no=setting.classes_no_in_batch,
                                                              instances_no=setting.instances_no_in_batch,
                                                              shuffle_points=setting.augment_shuffle_points,
                                                              shuffle_clouds=setting.augment_shuffle_clouds,
                                                              jitter_points=setting.augment_jitter_points,
                                                              rotate_pointclouds=False,
                                                              rotate_pointclouds_up=setting.augment_rotate_clouds,
                                                              sampling_method=setting.dataset_sampling_method):

                clouds = np.reshape(clouds, (setting.classes_no_in_batch*setting.instances_no_in_batch,
                                             setting.points_num, 3))
                global_batch_idx, _, training_loss, training_pos, training_neg, summary_train = sess.run([
                    model.global_step, model.get_optimizer(), model.get_loss_function(), model.pos_dist, model.neg_dist,
                    model.get_summary()], feed_dict={model.input_point_cloud: clouds, model.is_training: True})
            pbar.n = global_batch_idx
            pbar.refresh()
             
            if global_batch_idx // setting.save_summary_after_iterations > save_summary_index:

                # Update index
                save_summary_index += 1

                # Log summary
                summary_log = tf.Summary()
                summary_log.value.add(tag="%sdist_pos" % "", simple_value=np.sum(training_pos))
                summary_log.value.add(tag="%sdist_neg" % "", simple_value=np.sum(training_neg))

                ##################################################################################################
                ############################################# LOG ################################################
                ##################################################################################################

                # # pos/neg dist
                # if CALC_DIST:  # and (epoch % MODEL_SAVE_AFTER_EPOCHS == MODEL_SAVE_AFTER_EPOCHS - 1):
                #     pos_man, neg_man = test_features_extraction(data_gen, model, sess)
                #     summary_log.value.add(tag="%spos_neg_test_dist" % "", simple_value=neg_man - pos_man)

                # Variables histogram
                summary_histograms = sess.run(hist_summary)
                writer.add_summary(summary_histograms, global_batch_idx)

                # Write summary
                writer.add_summary(summary_log, global_batch_idx)
                writer.add_summary(summary_train, global_batch_idx)

                ##################################################################################################
                ########################################## SAVE MODEL ############################################
                ##################################################################################################

                if global_batch_idx // setting.save_model_after_iterations > save_model_index:
                    save_model_index += 1
                    save_path = model.save_model(sess, name)
                    print("Model saved in file: %s" % save_path)
        pbar.close()

# def test_features_extraction(data_gen, model, sess, partial_score=True):
#     """
#     Train deepclouds with synthetic data.
#     """
#
#     # Get test embeddings
#     batches = 0
#     test_embeddings = { k : [] for k in range(40)}
#     for clouds, labels in data_gen.generate_random_batch(False, batch_size=classes_no*instances_no, sampling_method=SAMPLING_METHOD):  # 400 test examples / 80 clouds = 5 batches
#
#         # count embeddings
#         #test_embedding_input = np.stack([clouds], axis=1)
#         test_embedding = sess.run(model.data_after_step_5, feed_dict={model.input_point_clouds: clouds,
#                                                                      model.placeholder_is_tr : False})
#         #test_embedding = np.squeeze(test_embedding, axis=1)
#
#         # add embeddings
#         for cloud_idx in range(labels.shape[0]):
#             test_embeddings[labels[cloud_idx]].append(test_embedding[cloud_idx])
#
#         # not the whole dataset
#         if partial_score:
#             batches += 1
#             if batches == 5:
#                 break
#
#     # Convert to numpy
#     class_embeddings = []
#     for k in range(40):
#         class_embeddings.append(test_embeddings[k])
#
# #    import pickle
# #    with open('class_embeddings.pkl', 'wb') as f:
# #        pickle.dump(class_embeddings, f, pickle.HIGHEST_PROTOCOL)
# #    exit()
#
#     # Calc distances between every embedding in one class
#     pos_man = []
#     for class_idx in range(len(class_embeddings)):
#         positive_dist_class = []
#         for instance_idx_1 in range(len(class_embeddings[class_idx])):
#             for instance_idx_2 in range(len(class_embeddings[class_idx])):
#                 if instance_idx_1 != instance_idx_2:
#                     if DISTANCE == 'euclidian':
#                         positive_dist_class.append(np.linalg.norm(class_embeddings[class_idx][instance_idx_1] -
#                                                                   class_embeddings[class_idx][instance_idx_2]))
#                     elif DISTANCE == 'cosine':
#                         numerator = np.squeeze(np.sum(np.multiply(class_embeddings[class_idx][instance_idx_1], class_embeddings[class_idx][instance_idx_2]), axis=-1))
#                         denominator = np.linalg.norm(class_embeddings[class_idx][instance_idx_1]) * np.linalg.norm(class_embeddings[class_idx][instance_idx_2]) + 1e-9
#                         positive_dist_class.append(1 - np.divide(numerator, denominator))
#
#     #                        positive_dist_class.append(cos_dist.cosine(class_embeddings[class_idx][instance_idx_1],
#     #                                                                   class_embeddings[class_idx][instance_idx_2]))
#         pos_man.append(positive_dist_class)
#     pos_man_flat = [item for sublist in pos_man for item in sublist]
#
#     # Calc distances between every embedding in one class and every other class
#     neg_man = []
#     for class_idx_1 in range(len(class_embeddings)):
#         negative_dist_class = []
#         for class_idx_2 in range(len(class_embeddings)):
#             if class_idx_1 != class_idx_2:
#                 for instance_idx_1 in range(len(class_embeddings[class_idx_1])):
#                     for instance_idx_2 in range(len(class_embeddings[class_idx_2])):
#                         if instance_idx_1 != instance_idx_2:
#                             if DISTANCE == 'euclidian':
#                                 negative_dist_class.append(np.linalg.norm(class_embeddings[class_idx_1][instance_idx_1] -
#                                                                           class_embeddings[class_idx_2][instance_idx_2]))
#                             elif DISTANCE == 'cosine':
#                                 numerator = np.squeeze(np.sum(np.multiply(class_embeddings[class_idx_1][instance_idx_1], class_embeddings[class_idx_2][instance_idx_2]), axis=-1))
#                                 denominator = np.linalg.norm(class_embeddings[class_idx_1][instance_idx_1]) * np.linalg.norm(class_embeddings[class_idx_2][instance_idx_2]) + 1e-9
#                                 negative_dist_class.append(1 - np.divide(numerator, denominator))
#
#                                 # negative_dist_class.append(cos_dist.cosine(class_embeddings[class_idx_1][instance_idx_1],
#                                 #                                           class_embeddings[class_idx_2][instance_idx_2]))
#         neg_man.append(negative_dist_class)
#     neg_man_flat = [item for sublist in neg_man for item in sublist]
#
#     return np.mean(pos_man_flat), np.mean(neg_man_flat)

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
    parser.add_argument("-s", "--setting", help="Setting file name", type=str, required=False,
                        default="pointnet_setting")
    args = vars(parser.parse_args())

    # Import setting module
    setting_module = importlib.import_module(args['setting'])
    setting = setting_module.Setting

    # train
    train_features_extraction(args["name"], setting)

    # Print all settings at the end of learning
    print("Training params:")
    print("name          = ", args["name"])
    print("batch_size    = ", args["batch_size"])
    print("epochs        = ", args["epochs"])
    print("learning rate = ", args["learning_rate"])
    print("margin        = ", args["margin"])

if __name__ == "__main__":
    main(sys.argv[1:])
