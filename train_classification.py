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
from siamese_pointnet.model import RNNBidirectionalModel, MLPModel

CLOUD_SIZE = 32

def train_classification(name, batch_size, epochs, learning_rate, device):
    """
    Train siamese pointnet classificator with synthetic data.
    """

    # Reset
    tf.reset_default_graph()

    # Generate data if needed
    data_gen = modelnet.SyntheticData(pointcloud_size=CLOUD_SIZE)

    # Define model
    with tf.device(device):
        model_features = OrderMattersModel(batch_size = 1, pointcloud_size = CLOUD_SIZE,
                                           read_block_units = [2**(np.floor(np.log2(3*CLOUD_SIZE)) + 1)],
                                           process_block_steps=32)

    # Saver
    saver = tf.train.Saver()

    # Define model
    with tf.device(device):
        model_classifier = MLPClassifier([512, 128, 40], batch_size, learning_rate)

    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    with tf.Session(config=config) as sess:

        # Run the initialization
        sess.run(tf.global_variables_initializer())
         
        # saver
        saver.restore(sess, tf.train.latest_checkpoint('models_feature_extractor'))
        
        # Logs
        log_model_dir = os.path.join(df.LOGS_DIR, model_classifier.get_model_name())
        writer = tf.summary.FileWriter(os.path.join(log_model_dir, name))
#         writer.add_graph(sess.graph)
        
        # Do the training loop
        global_batch_idx = 1
        summary_skip_batch = 1
        for epoch in range(epochs):

            # loop for all batches
            index = 1
            for clouds, labels in data_gen.generate_random_batch(train = True,
                                                                 batch_size = batch_size,
                                                                 shuffle_points=False,
                                                                 jitter_points=True,
                                                                 rotate_pointclouds=True):
            
                # count embeddings
                embedding_input = np.stack([clouds], axis=1)
                embeddings = sess.run(model_features.get_embeddings(), feed_dict={model_features.placeholder_embdg: embedding_input})

                # run optimizer
                new_labels = sess.run(tf.one_hot(labels, 40))
                _, loss, pred, summary = sess.run([model_classifier.get_optimizer(), model_classifier.get_loss_function(),
                                                   model_classifier.get_classification_prediction(), model_classifier.get_summary()],
                                                   feed_dict={model_classifier.placeholder_embed: embeddings,
                                                              model_classifier.placeholder_label: new_labels})
 
                
 
                print pred[0], labels[0]
                print "PREDICTED: %02d REAL: %02d" % (np.argmax(pred, axis=1)[0], labels[0])
                exit()

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
