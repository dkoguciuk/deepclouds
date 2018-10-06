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
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import train_with_hparam
import deepclouds.defines as df
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition
from sklearn.svm import SVC

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
SAMPLING_METHOD = 'fps'
TRAIN_EMBDGS_DIR = 'train_embgds'
REGULARIZATION_WEIGHT = 0.01

EMBEDDINGS_DIR = 'embeddings'

def calc_inner_class_distance(class_embeddings):
    """
    Calc inner class distance of input embeddings.

    Args:
        class_embeddings (B, E): For those clouds inner class distance will be computed.

    Returns:
        (float): Mean class distance between every pair of input pointclouds.
    """
    
    # expand dims
    class_embeddings = np.expand_dims(class_embeddings, axis=1) 
    
    # Calc distance matrix
    distances = []
    for querry_embedding in class_embeddings:
        pos_num  = np.squeeze(np.sum(np.multiply(querry_embedding, class_embeddings), axis=-1))
        neg_num  = np.linalg.norm(querry_embedding) * np.squeeze(np.linalg.norm(class_embeddings, axis=-1))
        distances.append(1 - np.divide(pos_num, neg_num))
    distances = np.stack(distances)
    
    # Get mean of upper triangle
    if len(distances) > 1:
        distance = np.mean(distances[np.triu_indices(len(class_embeddings), 1)])
    else:
        distance = 0
    return distance

def classify_knn(name, embeddings_dir, batch_size, epochs, learning_rate, device,
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
    data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

#     # train embeddings
#     train_embdgs = np.load(os.path.join(train_embdgs_dirpath, 'train_embdgs.npy'))
#     train_labels = np.load(os.path.join(train_embdgs_dirpath, 'train_labels.npy'))
#     train_clouds = np.load(os.path.join(train_embdgs_dirpath, 'train_clouds.npy'))
    
    # Means
    train_embdgs = []
    train_labels = []
    cluster_files = [f for f in os.listdir(embeddings_dir) if 'npy' in f]
    for cluster_file in cluster_files:
        cluster_path = os.path.join(embeddings_dir, cluster_file)
        cluster_clss = int(cluster_file.split('_')[1])        
        cluster_mean = np.load(cluster_path)
        train_embdgs.append(cluster_mean)
        train_labels.append(cluster_clss)    
    train_embdgs = np.stack(train_embdgs, axis=0)
    train_labels = np.array(train_labels)
    
    train_embdgs_a = train_embdgs[train_labels == 4]
    train_embdgs_b = train_embdgs[train_labels == 38]
    train_labels_a = train_labels[train_labels == 4]
    train_labels_b = train_labels[train_labels == 38]
    train_embdgs = np.concatenate((train_embdgs_a, train_embdgs_b))
    train_labels = np.concatenate((train_labels_a, train_labels_b))
    
    
    #clf = SVC(C=0.1, gamma=0.01)
    clf = SVC()
    clf.fit(train_embdgs, train_labels)
    train_label_hat = clf.predict(train_embdgs)
    print ("TRAIN ACC ", float(np.sum(train_label_hat == train_labels)) / len(train_labels))

#     cluster_classes = np.unique([f.split('_')[1] for f in subclasses_files])
#     for class_idx in subclasses_classes:
#         # Get nonclass clouds from trainset
#         nonclass_idx = np.where(self.labels_train != int(class_idx))[0]
#         nonclass_lbls = self.labels_train[nonclass_idx]
#         nonclass_clds = self.pointclouds_train[nonclass_idx]
#         # Get class clouds from trainset
#         class_idcs = np.where(self.labels_train == int(class_idx))[0]
#         class_lbls = self.labels_train[class_idcs]
#         class_clds = self.pointclouds_train[class_idcs]
#         # Get subclass clusters
#         new_clds = []
#         new_lbls = []
#         subclasses_clusters = [f.split('_')[3][0] for f in subclasses_files if f.split('_')[1] == class_idx]
#         for cluster in np.sort(subclasses_clusters):
#             # Load cluster indices
#             cluster_indices = np.load(os.path.join(df.DATA_MODELNET_DIR, 'cloud_' + class_idx + '_cluster_' + cluster + '.npy'))
#             # Split
#             new_cld = class_clds[cluster_indices]
#             new_lbl = np.ones((cluster_indices.shape[0],1), dtype=np.int32) * (int(class_idx)*100 + int(cluster))
#             new_clds.append(new_cld)
#             new_lbls.append(new_lbl)
# 
#         # Concatenate
#         new_clds = np.concatenate(new_clds)
#         new_lbls = np.concatenate(new_lbls)
#         self.pointclouds_train = np.concatenate((nonclass_clds, new_clds))
#         self.labels_train = np.concatenate((nonclass_lbls, new_lbls))
#             # Align label numering
#             labels_actual = np.unique(self.labels_train)
#             labels_aligned = np.arange(len(labels_actual))
#             labels_missing = np.setdiff1d(labels_aligned, labels_actual, assume_unique=True)
#             labels_additional = np.setdiff1d(labels_actual, labels_aligned, assume_unique=True)
#             for label_missing, label_additional in zip(labels_missing, labels_additional):
#                 self.labels_train[self.labels_train == label_additional] = label_missing
#             self.CLASSES_COUNT = len(np.unique(self.labels_train))

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
                                             regularization_weight=REGULARIZATION_WEIGHT,
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
        features_model_saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
    
        ##########################################################################################
        ##################################### TEST CLOUD LOOP ################################
        ##########################################################################################

        hit_a = 0.
        hit_b = 0.
        all_a = 0.
        all_b = 0.
        sys.stdout.write("Calculating test embedding")
        sys.stdout.flush()
        for clouds, labels in data_gen.generate_random_batch(train = False,
                                                             batch_size = batch_size,
                                                             shuffle_points=SHUFFLE_CLOUDS,
                                                             jitter_points=True,
                                                             rotate_pointclouds=False,
                                                             rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                             sampling_method=SAMPLING_METHOD):
            sys.stdout.write('.')
            sys.stdout.flush()
    
            ##################################################################################################
            ######################################### GET EMBEDDING ##########################################
            ##################################################################################################
            
            embedding_input = np.stack([clouds], axis=1)
            embeddings = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                              features_model.placeholder_is_tr : False})
            embeddings = np.squeeze(embeddings)
            
            ##################################################################################################
            ############################################### SVM ##############################################
            ##################################################################################################
            
            y_hat = clf.predict(embeddings)
            
            where_a = np.where(labels == 4)
            hit_a += np.sum(y_hat[where_a] == 4)
            all_a += len(where_a[0])
            where_b = np.where(labels == 38)
            hit_b += np.sum(y_hat[where_b] == 38)
            all_b += len(where_b[0])
            continue
            
            y_a = [labels == 4]
            y_b = [labels == 38]
            
            hit_a = np.sum(y_hat[y_a] == 4)
            hit_b = np.sum(y_hat[y_a] == 4)
            all_a = len(y_a)
            all_b = len(y_b)
            
            print ("A = ", hit_a, all_a, float(hit_a)/all_a)
            print ("B = ", hit_b, all_b, float(hit_b)/all_a)
            hit += (hit_a + hit_b)
            all += (all_a + all_b)
            
            
#             y_hat = clf.predict(embeddings)
#             hit += np.sum(y_hat == labels)
#             all += len(labels)
            continue
            
            N = 4
            clouds_ext = []
            for i in range(N):
                batch_ext = np.expand_dims(clouds, 1).copy()
                batch_ext = modelnet.GenericData._shuffle_points_in_batch(batch_ext)
                batch_ext = modelnet.GenericData._jitter_batch(batch_ext)
                batch_ext = modelnet.GenericData._rotate_batch(batch_ext, random_axis=False)
                batch_ext = np.squeeze(batch_ext, axis=1)
                clouds_ext.append(batch_ext)
            clouds_ext = np.stack(clouds_ext)
            
#             print clouds_ext[0][0][0]
#             print clouds_ext[1][0][0]
#             print clouds_ext[2][0][0]
#             print clouds_ext[3][0][0]

            # count embeddings
            embeddings_ext = []
            for i in range(N):
                embedding_input = np.stack([clouds_ext[i]], axis=1)
                embeddings = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                                  features_model.placeholder_is_tr : False})
                embeddings = np.squeeze(embeddings)
                embeddings_ext.append(embeddings)
            embeddings_ext = np.stack(embeddings_ext)

            
#             print embeddings_ext.shape
#             print embeddings_ext[0][0][0]
#             print embeddings_ext[1][0][0]
#             print embeddings_ext[2][0][0]
#             print embeddings_ext[3][0][0]                  

            ##################################################################################################
            ################################### CLASSIFICATION ACCURACY ######################################
            ##################################################################################################
            
            for cloud_idx in range(len(clouds)):
                
#                 # MULTiPLE
#                 argmins = []
#                 for i in range(N):
#                     
#                     querry_embdg = embeddings_ext[i][cloud_idx]
#                     
#                     numerator = np.squeeze(np.sum(np.multiply(querry_embdg, train_embdgs), axis=-1))
#                     denominator = np.linalg.norm(querry_embdg) * np.squeeze(np.linalg.norm(train_embdgs, axis=-1))
#                     distances = 1 - np.divide(numerator, denominator)
#                 
#                     argmin = distances.argsort()[:5]
#                     argmins.append(argmin)
#                 argmins = np.concatenate(argmins)

                querry_embdg = embeddings_ext[0][cloud_idx]
                numerator = np.squeeze(np.sum(np.multiply(querry_embdg, train_embdgs), axis=-1))
                denominator = np.linalg.norm(querry_embdg) * np.squeeze(np.linalg.norm(train_embdgs, axis=-1))
                distances = 1 - np.divide(numerator, denominator)
                argmins = distances.argsort()[:30]
                
                #argmin = stats.mode(argmins)[0]
                #print argmins
                
                
#                 print labels[cloud_idx], train_labels[argmins[0]]
#                 print labels[cloud_idx], train_labels[argmins[0]]
                #if (labels[cloud_idx] == train_labels[argmins[0]]):# or labels[cloud_idx] == train_labels[argmins[1]] or labels[cloud_idx] == train_labels[argmins[2]] or
                    #labels[cloud_idx] == train_labels[argmins[3]] or labels[cloud_idx] == train_labels[argmins[4]]):
                if labels[cloud_idx] == stats.mode(train_labels[argmins])[0][0]:
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
        print ("Test accuracy a= ", float(hit_a)/all_a, 'missed: ', all_a - hit_a)
        print ("Test accuracy b= ", float(hit_b)/all_b, 'missed: ', all_b - hit_b)
       
def calc_and_save_train_embeddings(embeddings_dir, abstraction_level):
    """
    Args:
        abstracion_level (str): One of the following: class, cluster, instance, where:
            class - save mean embedding for each class
            cluster - save mean embedding of each cluster of each class
            instance - save embedding of each instance of each class
    """

    ##################################################################################################
    ############################################ RM/MK DIR ###########################################
    ##################################################################################################

    if os.path.exists(embeddings_dir):
        shutil.rmtree(embeddings_dir)
    os.mkdir(embeddings_dir)

    ##################################################################################################
    ############################################ DATA GEN ############################################
    ##################################################################################################

    data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE, clusterize=False)
    
    ##################################################################################################
    ########################################### FEXT MODEL ###########################################
    ##################################################################################################

    with tf.variable_scope("end-to-end"):
        with tf.device("/device:GPU:0"):
            features_model = DeepCloudsModel(train=True,
                                             batch_size = 1,
                                             pointcloud_size = CLOUD_SIZE,
                                             read_block_units = READ_BLOCK_UNITS,
                                             process_block_steps=PROCESS_BLOCK_STEPS,
                                             normalize_embedding=True, verbose=True,
                                             input_t_net=True, feature_t_net=True,
                                             read_block_method=READ_BLOCK_METHOD,
                                             process_block_method=PROCESS_BLOCK_METHOD,
                                             regularization_weight=REGULARIZATION_WEIGHT,
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
        features_model_saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
    
        ##########################################################################################
        ######################################### CLASSES LOOP ###################################
        ##########################################################################################
    
        # Get training clouds
        for class_idx, class_name in enumerate(tqdm(data_gen.class_names)):
        
            ##########################################################################################
            ######################################## CALC EMBEDDINGS #################################
            ##########################################################################################
            
            clouds, labels = data_gen.generate_class_clouds(train=True, class_idx=class_idx, sampled=True,
                                                            shuffle_points=True,
                                                            jitter_points=True,
                                                            rotate_pointclouds=False,
                                                            rotate_pointclouds_up=ROTATE_CLOUDS_UP,
                                                            sampling_method=SAMPLING_METHOD)
                 
            embeddings = []
            embedding_inputs = np.split(clouds, len(clouds))
            for embedding_input in embedding_inputs:
                embedding_input = np.expand_dims(embedding_input, axis=1)         
                embeddings.append(sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input,
                                                                                       features_model.margin : [0.],
                                                                                       features_model.placeholder_is_tr : True}))        
            embeddings = np.squeeze(np.concatenate(embeddings))
            
#             ##########################################################################################
#             ########################################### PCA ##########################################
#             ##########################################################################################
#              
#             # PCA decomposition 
#             start_time = time.time()
#             pca = decomposition.PCA(n_components=2)
#             pca.fit(embeddings)
#             features_pca = pca.transform(embeddings)
#             pca_time = time.time()
#             print ("DISTANCE = ", calc_inner_class_distance(embeddings))
#             print("PCA features calculated in ", pca_time - start_time, " seconds with variance ", pca.explained_variance_ratio_)
#             plt.scatter(features_pca[:, 0], features_pca[:, 1], c='red')
#             plt.ylim(-1, 1)
#             plt.xlim(-1, 1)
#             plt.show()
#             exit()

            if abstraction_level == 'class':
                mean_embdd = np.squeeze(np.mean(embeddings, axis=0))
                mean_embdd = mean_embdd / np.linalg.norm(mean_embdd)
                np.save(os.path.join(embeddings_dir, 'cloud_' + str(class_idx) + '_mean_0.npy'), mean_embdd)
            elif abstraction_level == 'cluster':
                
                class_inner_dist = calc_inner_class_distance(embeddings)
                
                # Dist bigger
                #if class_inner_dist > 0.1:
 
                # KMeans
                N = 2
                #kmeans_model = KMeans(n_clusters=N, n_init=100)
                kmeans_model = SphericalKMeans(n_clusters=N, n_init=100)
                kmeans_model.fit(embeddings)
                
                subclass_labels = kmeans_model.predict(embeddings)
                for i in range(N):
                    cluster = embeddings[subclass_labels == i]
                    print cluster.shape, calc_inner_class_distance(cluster)
                     
                    #if calc_inner_class_distance(cluster) < 0.1:
                    mean_embdd = np.squeeze(np.mean(cluster, axis=0))
                    mean_embdd = mean_embdd / np.linalg.norm(mean_embdd)
                    np.save(os.path.join(embeddings_dir, 'cloud_' + str(class_idx) + '_mean_' + str(i) + '.npy'), mean_embdd)
                    #np.save(os.path.join(embeddings_dir, 'cloud_' + str(class_idx) + '_mean_' + str(i) + '.npy'), kmeans_model.cluster_centers_[0])
            elif abstraction_level == 'instance':
                np.random.shuffle(embeddings)
                for i in range(50):
                    np.save(os.path.join(embeddings_dir, 'cloud_' + str(class_idx) + '_mean_' + str(i) + '.npy'), embeddings[i])
            else:
                print ("THIS abstraction_level IS NOT IMPLEMENTED YET")
                exit()
            

def main(argv):

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the run", type=str, required=True)
    parser.add_argument("-b", "--batch_size", help="The size of a batch", type=int, required=False, default=2)
    parser.add_argument("-e", "--epochs", help="Number of epochs of training", type=int, required=False, default=25)
    parser.add_argument("-l", "--learning_rate", help="Learning rate value", type=float, required=True)
    parser.add_argument("-r", "--read_block_units", help="Read block units", type=int, required=False, default=512)
    parser.add_argument("-p", "--process_block_steps", help="Process block steps", type=int, required=False, default=4)
    parser.add_argument("-d", "--device", help="Which device to use (i.e. /device:GPU:0)", type=str, required=False, default="/device:GPU:0")
    
    parser.add_argument("-s", "--save_embeddings", help="Save train embeddings on disc", action='store_true', required=False, default=False)
    args = vars(parser.parse_args())

    # save embeddings
    if args['save_embeddings']:
        calc_and_save_train_embeddings(EMBEDDINGS_DIR, abstraction_level='instance')

    # classify
    classify_knn(args["name"], EMBEDDINGS_DIR, batch_size=args["batch_size"], epochs=args["epochs"],
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
