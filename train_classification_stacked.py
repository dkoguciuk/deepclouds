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
import train_with_hparam
import deepclouds.defines as df
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import OrderMattersModel

SAMPLES = 5
CLOUD_SIZE = 128
SAMPLING_METHOD = 'via_graphs'
CLASSIFIER_MODEL_LOAD = False
CLASSIFIER_MODEL_TRAIN = True

def train_classification_stacked(name, batch_size, epochs, learning_rate, device,
                                 read_block_units, process_block_steps,
                                 classifier_layers = [2048, 1024, 512, 256, 128, 40]):
                                 #classifier_layers = [10240, 2560, 640, 160, 40]):
    """
    Train deepclouds classificator with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    #data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)
    data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)

    ##################################################################################################
    ################################## FEATURES EXTRACTION MODEL #####################################
    ##################################################################################################

    with tf.variable_scope("end-to-end"):
        with tf.device(device):
            features_model = OrderMattersModel(train=False,
                                               batch_size = batch_size,
                                               pointcloud_size = CLOUD_SIZE,
                                               read_block_units = read_block_units,
                                               process_block_steps=process_block_steps)
            features_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope="end-to-end")
            features_model_saver = tf.train.Saver(features_vars)

    ##################################################################################################
    ##################################### CLASSIFICATION MODEL #######################################
    ##################################################################################################

    with tf.device(device):
        classifier_model = MLPClassifier(classifier_layers, batch_size, learning_rate)
        if CLASSIFIER_MODEL_LOAD:
            classifier_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope=classifier_model.MODEL_NAME)
            classifier_model_saver = tf.train.Saver(classifier_vars)

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
        features_model_saver.restore(sess, tf.train.latest_checkpoint('100_classification_128_points_up_modelnet'))
        if CLASSIFIER_MODEL_LOAD:
            classifier_model_saver.restore(sess, tf.train.latest_checkpoint('models_classifier'))

        ##############################################################################################
        ######################################## SUMMARY WRITER ######################################
        ##############################################################################################

        log_model_dir = os.path.join(df.LOGS_DIR, classifier_model.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
#         writer.add_graph(sess.graph)
        
        ##############################################################################################
        ##################################### EPOCH TRAINING LOOP ####################################
        ##############################################################################################
        
        if CLASSIFIER_MODEL_TRAIN:
        
            global_batch_idx = 1
            #summary_skip_batch = 10
            checkpoint_skip_epochs = 25
            accuracies = []
            for epoch in range(epochs):
    
                ##########################################################################################
                ##################################### BATCH TRAINING LOOP ################################
                ##########################################################################################
    
                #epoch_batch_idx = 1
                sys.stdout.write('Training epoch ' + str(epoch) + " ")
                for clouds, labels in data_gen.generate_random_batch_multiple_clouds(train = True,
                                                                                     batch_size = batch_size,
                                                                                     shuffle_points=True,
                                                                                     jitter_points=True,
                                                                                     rotate_pointclouds=False,
                                                                                     rotate_pointclouds_up=True,
                                                                                     samples = SAMPLES,
                                                                                     sampling_method = SAMPLING_METHOD):
    
                    sys.stdout.write('.')
                    sys.stdout.flush()
    
                    ##################################################################################################
                    ######################################### GET EMBEDDING ##########################################
                    ##################################################################################################

                    # count embeddings
                    embeddings = []
                    for sample_idx in range(SAMPLES):
                         embedding_input = np.stack([clouds[:,sample_idx]], axis=1)
                         embeddings_sample = sess.run(features_model.get_embeddings(), feed_dict={features_model.placeholder_embdg: embedding_input})
                         embeddings.append(embeddings_sample)
                    embeddings = np.stack(embeddings, axis=1)
                    #embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
                    embeddings = np.mean(embeddings, axis=1)
    
                    ##################################################################################################
                    ############################################# TRAIN ##############################################
                    ##################################################################################################
    
                    # run optimizer
                    new_labels = sess.run(tf.one_hot(labels, 40))
                    _, loss, summary_train = sess.run([classifier_model.get_optimizer(),
                                                       classifier_model.get_loss_function(),
                                                       classifier_model.get_summary()],
                                                       feed_dict={classifier_model.placeholder_embed: embeddings,
                                                                  classifier_model.placeholder_label: new_labels})
     
                    global_batch_idx += 1
                    #epoch_batch_idx += 1
                    
                    #print global_batch_idx, loss
                    
            ##################################################################################################
            ################################### CLASSIFICATION ACCURACY ######################################
            ##################################################################################################
            
            #if global_batch_idx % summary_skip_batch == 0:
                
                # Get test embeddings
                hit = 0.
                all = 0.
                print("")
                sys.stdout.write('Calculating test classification accuracy after epoch ' + str(epoch) + " ")
                for clouds, labels in data_gen.generate_random_batch_multiple_clouds(False, 2,
                                                                                     samples = SAMPLES,
                                                                                     sampling_method = SAMPLING_METHOD):# 400 test examples / 16 clouds = 25 batches
                    
                    # count embeddings
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    
                    embeddings = []
                    for sample_idx in range(SAMPLES):
                        
                        # padding
                        clouds_padding = np.zeros((batch_size - 2, CLOUD_SIZE, 3), dtype=np.float)
                        clouds_padded = np.concatenate((clouds[:,sample_idx], clouds_padding), axis=0)
                        labels_padding = np.zeros((batch_size - 2), dtype=np.int)
                        labels_padded = np.concatenate((labels, labels_padding), axis=0)
                            
                        # count embeddings
                        embedding_input = np.stack([clouds_padded], axis=1)
                        embeddings_sample = sess.run(features_model.get_embeddings(),
                                                   feed_dict={features_model.placeholder_embdg: embedding_input})
                        embeddings.append(embeddings_sample)
    
                    embeddings = np.stack(embeddings, axis=1)
                    embeddings = np.mean(embeddings, axis=1)
                    #embeddings = np.reshape(embeddings, (embeddings.shape[0], -1))
                        
                    # One hot
                    labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
                    pred = sess.run(classifier_model.get_classification_prediction(),
                                    feed_dict={classifier_model.placeholder_embed: embeddings,
                                               classifier_model.placeholder_label: labels_padded_one_hot})
                        
                    # accuracy 
                    predictions = np.argmax(pred, axis=1)
                    predictions = predictions[:len(labels)]
                    hit = hit + sum(predictions == labels)
                    all = all + len(labels)
                    
                ##################################################################################################
                ############################################# SUMMARIES ##########################################
                ##################################################################################################
                
                accuracies.append(hit / all)
                summary_test = tf.Summary()
                summary_test.value.add(tag="%stest_classification_accuracy" % "", simple_value= (hit / all))
                writer.add_summary(summary_test, global_batch_idx)
                writer.add_summary(summary_train, global_batch_idx)
                print("")
                print "Epoch: %06d batch: %03d loss: %06f test_accuracy: %06f" % (epoch + 1, global_batch_idx, loss, (hit / all))
    
                ##################################################################################################
                ################################## SAVE CLASSIFICATION MODEL #####################################
                ##################################################################################################
    
                if (epoch+1) % checkpoint_skip_epochs == 0:
     
                    # Save model
                    save_path = classifier_model.save_model(sess, name)
                    print "Model saved in file: %s" % save_path
                    print "MAX ACC = ", np.max(accuracies) 

        else:
            accuracies = []
            global_votes = []
            global_labels = []
            for _ in range(10):
                # Get test embeddings
                hit = 0.
                all = 0.
                test_batch_size = 2
                batch_votes = []
                batch_labels = []
                for clouds, labels in data_gen.generate_random_batch(False, batch_size=test_batch_size,
                                                                     shuffle_points=False,
                                                                     rotate_pointclouds_up=False):# 400 test examples / 16 clouds = 25 batches
                    
                    # count embeddings
                    embeddings = []
                    for sample_idx in range(SAMPLES):
                    
                        # padding
                        clouds_padding = np.zeros((batch_size - test_batch_size, CLOUD_SIZE, 3), dtype=np.float)
                        clouds_padded = np.concatenate((clouds[:,sample_idx], clouds_padding), axis=0)
                        labels_padding = np.zeros((batch_size - test_batch_size), dtype=np.int)
                        labels_padded = np.concatenate((labels, labels_padding), axis=0)
                            
                        # count embeddings
                        embedding_input = np.stack([clouds_padded], axis=1)
                        embeddings_sample = sess.run(features_model.get_embeddings(),
                                                   feed_dict={features_model.placeholder_embdg: embedding_input})
                        embeddings.append(embeddings_sample)
                    embeddings = np.stack(embeddings, axis=1)
                    embeddings = np.sum(embeddings, axis=1)
                        
                    # One hot
                    labels_padded_one_hot = sess.run(tf.one_hot(labels_padded, 40))
                    predictions = sess.run(classifier_model.get_classification_prediction(),
                                           feed_dict={classifier_model.placeholder_embed: embeddings,
                                                      classifier_model.placeholder_label: labels_padded_one_hot})
                        
                    # accuracy 
                    predictions_args = np.argmax(predictions, axis=1)
                    predictions_args = predictions_args[:len(labels)]
                    hit = hit + sum(predictions_args == labels)
                    all = all + len(labels)
                    for idx in range(len(labels)):
                        batch_votes.append(predictions[idx])
                        batch_labels.append(labels[idx])
            
                print "Accuracy: ", hit/all        
                accuracies.append(hit / all)
                global_votes.append(batch_votes)
                print "GLOBALVOTES", len(global_votes), len(global_votes[0])
                global_labels.append(batch_labels)
                if len(global_votes) == 2:
                    break
            np.save("votes.npy", np.transpose(np.squeeze(np.array(global_votes))))
            np.save("labels.npy", np.transpose(np.squeeze(np.array(global_labels))))
            print "AVG ACCURACY = ", np.mean(accuracies)
            
            

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
    train_classification_stacked(args["name"], batch_size=args["batch_size"], epochs=args["epochs"],
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
