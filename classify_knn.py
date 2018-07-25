#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import pypcd
import shutil
import argparse
import numpy as np
import tensorflow as tf
import train_with_hparam
import deepclouds.defines as df
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

CLOUD_SIZE = 1024
CLASSIFIER_MODEL_LOAD = False
CLASSIFIER_MODEL_TRAIN = True
SYNTHETIC = False
READ_BLOCK_UNITS = [256]
ROTATE_CLOUDS_UP = True
SHUFFLE_CLOUDS = True
READ_BLOCK_METHOD = 'pointnet'
PROCESS_BLOCK_METHOD = 'max-pool'
#PROCESS_BLOCK_METHOD = 'attention-rnn'
PROCESS_BLOCK_STEPS = [4]
SAMPLING_METHOD = 'random'
TRAIN_EMBDGS_DIR = 'train_embgds'

def calc_train_embedding(dirpath, epochs=1, batch_size=64, device="/device:GPU:0"):
    '''
    Calc embedding of train point clouds and save it in @dirpath arg.

    Args:
        dirpath (str): Where to save train embeddings.
        epochs (int): How many times should I save train set (with rotating and permuting).
    '''

    ##################################################################################################
    ######################################### INIT THINGS ############################################
    ##################################################################################################

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    if SYNTHETIC:
        data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE, permuted=SHUFFLE_CLOUDS,
                                          rotated_up=ROTATE_CLOUDS_UP, rotated_rand=False)
    else:
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    ##################################################################################################
    ################################## FEATURES EXTRACTION MODEL #####################################
    ##################################################################################################

    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            features_model = DeepCloudsModel(train=False,
                                             batch_size = batch_size,
                                             pointcloud_size = CLOUD_SIZE,
                                             read_block_units = READ_BLOCK_UNITS,
                                             process_block_steps=PROCESS_BLOCK_STEPS,
                                             normalize_embedding=True, verbose=True,
                                             input_t_net=True, feature_t_net=True,
                                             read_block_method=READ_BLOCK_METHOD,
                                             process_block_method=PROCESS_BLOCK_METHOD,
                                             distance='cosine')
            
            features_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="end-to-end")
            features_model_saver = tf.train.Saver(features_vars)

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
        features_model_saver.restore(sess, tf.train.latest_checkpoint('models_temp'))

        train_embdgs = []
        train_labels = []
        train_clouds = []

        ##############################################################################################
        ######################################### GET EMBEDDING ######################################
        ##############################################################################################

        for epoch in range(epochs):

            sys.stdout.write("Calculating train embedding for epoch " + str(epoch) + " ")
            sys.stdout.flush()
            for clouds, labels in data_gen.generate_random_batch(train = True,
                                                                 batch_size = batch_size,
                                                                 shuffle_points=SHUFFLE_CLOUDS,
                                                                 jitter_points=True,
                                                                 rotate_pointclouds=False,
                                                                 rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                                 sampling_method=SAMPLING_METHOD):

                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                                  features_model.placeholder_is_tr : False})
                train_embdgs.append(np.squeeze(embeddings))
                train_labels.append(labels)
                train_clouds.append(clouds)
                sys.stdout.write('.')
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
         
        ##############################################################################################
        ######################################## SAVE EMBEDDINGS #####################################
        ##############################################################################################

        train_embdgs = np.concatenate(train_embdgs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_clouds = np.concatenate(train_clouds, axis=0)
        filepath_embdgs = os.path.join(dirpath, 'train_embdgs.npy')
        filepath_labels = os.path.join(dirpath, 'train_labels.npy')
        filepath_clouds = os.path.join(dirpath, 'train_clouds.npy')
        np.save(filepath_embdgs, train_embdgs)
        np.save(filepath_labels, train_labels)
        np.save(filepath_clouds, train_clouds)



def classify_knn(name, train_embdgs_dirpath, batch_size, epochs, learning_rate, device,
                 read_block_units, process_block_steps):
    """
    Train deepclouds classificator with synthetic data.
    """

    ##################################################################################################
    ######################################### INIT THINGS ############################################
    ##################################################################################################

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    if SYNTHETIC:
        data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE, permuted=SHUFFLE_CLOUDS,
                                          rotated_up=ROTATE_CLOUDS_UP, rotated_rand=False)
    else:
        data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

    # train embeddings
    train_embdgs = np.load(os.path.join(train_embdgs_dirpath, 'train_embdgs.npy'))
    train_labels = np.load(os.path.join(train_embdgs_dirpath, 'train_labels.npy'))
    train_clouds = np.load(os.path.join(train_embdgs_dirpath, 'train_clouds.npy'))

    ##################################################################################################
    ################################## FEATURES EXTRACTION MODEL #####################################
    ##################################################################################################

    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            features_model = DeepCloudsModel(train=False,
                                             batch_size = batch_size,
                                             pointcloud_size = CLOUD_SIZE,
                                             read_block_units = READ_BLOCK_UNITS,
                                             process_block_steps=process_block_steps,
                                             normalize_embedding=True, verbose=True,
                                             input_t_net=True, feature_t_net=True,
                                             read_block_method=READ_BLOCK_METHOD,
                                             process_block_method=PROCESS_BLOCK_METHOD,
                                             distance='cosine')
            
            features_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="end-to-end")
            features_model_saver = tf.train.Saver(features_vars)

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
        features_model_saver.restore(sess, tf.train.latest_checkpoint('models_temp'))
    
        ##########################################################################################
        ##################################### TEST CLOUD LOOP ################################
        ##########################################################################################

        hit = 0.
        all = 0.
        sys.stdout.write("Calculating test embedding")
        sys.stdout.flush()
        for clouds, labels in data_gen.generate_random_batch(train = False,
                                                             batch_size = batch_size,
                                                             shuffle_points=SHUFFLE_CLOUDS,
                                                             jitter_points=True,
                                                             rotate_pointclouds=False,
                                                             rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                             sampling_method=SAMPLING_METHOD):
    
            ##################################################################################################
            ######################################### GET EMBEDDING ##########################################
            ##################################################################################################
    
            # count embeddings
            embedding_input = np.stack([clouds], axis=1)
            embeddings = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                              features_model.placeholder_is_tr : False})
            embeddings = np.squeeze(embeddings)
            sys.stdout.write('.')
            sys.stdout.flush()                    

            ##################################################################################################
            ################################### CLASSIFICATION ACCURACY ######################################
            ##################################################################################################
            
            for cloud_idx in range(len(embeddings)):
                querry_embdg = embeddings[cloud_idx]
                
                numerator = np.squeeze(np.sum(np.multiply(querry_embdg, train_embdgs), axis=-1))
                denominator = np.linalg.norm(querry_embdg) * np.squeeze(np.linalg.norm(train_embdgs, axis=-1))
                distances = 1 - np.divide(numerator, denominator)
                
                argmins = distances.argsort()[:5]
                if labels[cloud_idx] == train_labels[argmins[0]]:# or labels[cloud_idx] == train_labels[argmins[1]] or labels[cloud_idx] == train_labels[argmins[2]]:
                    hit = hit + 1
#                else:
#                    if cloud_idx in [35, 36, 38, 58]:
#                       continue
#                    print (cloud_idx, labels[cloud_idx], train_labels[argmins[0]], distances[argmins[0]], train_labels[argmins[1]], distances[argmins[1]], train_labels[argmins[2]], distances[argmins[2]])
#                    np.save('querry_cloud.npy', clouds[cloud_idx])
#                    np.save('neighb_cloud.npy', train_clouds[argmins[0]])
#                    querry_cloud = pypcd.make_xyz_point_cloud(clouds[cloud_idx])
#                    neighb_cloud = pypcd.make_xyz_point_cloud(train_clouds[argmins[0]])
#                    querry_cloud.save_pcd("querry_cloud.pcd")
#                    neighb_cloud.save_pcd("neighb_cloud.pcd")
#                    concat_cloud = np.concatenate([clouds[cloud_idx], train_clouds[argmins[0]]], axis=0)
#                    concat_cloud = pypcd.make_xyz_point_cloud(concat_cloud)
#                    concat_cloud.save_pcd("concat_cloud.pcd")
#                    exit()
                all = all + 1
#            exit()

        print('')
        print ("Test accuracy = ", hit/all)
            
            

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=64)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=25)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-r", "--read_block_units", help="Read block units", type=int, required=False, default=512)
    parser.add_argument("-p", "--process_block_steps", help="Process block steps", type=int, required=False, default=4)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    args = vars(parser.parse_args())

    # train


    if not os.path.exists(TRAIN_EMBDGS_DIR):
        calc_train_embedding(TRAIN_EMBDGS_DIR, epochs=10)

    classify_knn(args["name"], TRAIN_EMBDGS_DIR, batch_size=args["batch_size"], epochs=args["epochs"],
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
