#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''

import os
import sys
import time
import shutil
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition

READ_BLOCK_SIZE = 32

def main(argv):

    # Load
    embeddings = { k : [] for k in range(40)}
    for filename in os.listdir("embeddings"):
        if 'label' in filename:
            continue
        file_idx = int(filename[filename.find("_")+1:filename.find(".")])
        class_idx = (file_idx % 40)
        data = np.squeeze(np.load(os.path.join("embeddings", filename)))
        embeddings[class_idx].append(data)

    # Convert to numpy
    class_embeddings = []
    for k in range(40):
        class_embeddings.append(embeddings[k])
    class_embeddings = np.stack(class_embeddings, axis=0)    

    # Calc distances between every embedding in one class 
    pos_man = []
    for class_idx in range(class_embeddings.shape[0]):
        positive_dist_class = []
        for instance_idx_1 in range(class_embeddings.shape[1]):
            for instance_idx_2 in range(class_embeddings.shape[1]):
                if instance_idx_1 != instance_idx_2:
                    positive_dist_class.append(np.linalg.norm(class_embeddings[class_idx][instance_idx_1] -
                                                              class_embeddings[class_idx][instance_idx_2]))
        pos_man.append(positive_dist_class)
        print("POSITIVE DIST FOR CLASS {0:02d} calculated".format(class_idx + 1))
        
    # Calc distances between every embedding in one class and every other class
    neg_man = []
    for class_idx_1 in range(class_embeddings.shape[0]):
        negative_dist_class = []
        for class_idx_2 in range(class_embeddings.shape[0]):
            if class_idx_1 != class_idx_2:
                for instance_idx_1 in range(class_embeddings.shape[1]):
                    for instance_idx_2 in range(class_embeddings.shape[1]):
                        if instance_idx_1 != instance_idx_2:
                            negative_dist_class.append(np.linalg.norm(class_embeddings[class_idx_1][instance_idx_1] -
                                                                      class_embeddings[class_idx_2][instance_idx_2]))
        neg_man.append(negative_dist_class)
        print("NEGATIVE DIST FOR CLASS {0:02d} calculated".format(class_idx_1 + 1))
        
    # t-SNE decomposition
    skip = class_embeddings.shape[1]
    dec_features = class_embeddings.reshape(-1, READ_BLOCK_SIZE*4)
    dec_labels = []
    for class_idx in range(40):
        for instance_idx in range(10):
            dec_labels.append(class_idx)
    
    # PCA decomposition 
    start_time = time.time()
    pca = decomposition.PCA(n_components=2)
    pca.fit(dec_features)
    features_pca = pca.transform(dec_features)
    pca_time = time.time()
    print("PCA features calculated in ", pca_time - start_time, " seconds with variance ", pca.explained_variance_ratio_)
    
    features = []
    for cls in range(40):
        features.append(features_pca[cls*skip:(cls+1)*skip, :])
    
#     elems = 5000
#     tsne = TSNE(n_components=2, verbose=True, perplexity=40, n_iter=10000)
#     features_tsne = tsne.fit_transform(dec_features, dec_labels)
#     tsne_time = time.time()
#     print "t-SNE features calculated in ", tsne_time - pca_time, " seconds "
    # plots
    
#    plt.figure(figsize=(15, 15))
#    print(features_pca.shape)
#    plt.scatter(features[0][:, 0], features[0][:, 1], c='red')
    #plt.scatter(feature_2[:, 0], feature_2[:, 1], c='blue')
    #plt.scatter(feature_3[:, 0], feature_3[:, 1], c='orange')
    #plt.scatter(feature_4[:, 0], feature_4[:, 1], c='magenta')
    #plt.scatter(feature_4[:, 0], feature_4[:, 1], c='gray')
    #plt.scatter(feature_5[:, 0], feature_5[:, 1], c='tan')
    #plt.scatter(feature_6[:, 0], feature_6[:, 1], c='gold')
#    plt.scatter(features[7][:, 0], features[7][:, 1], c='navy')
    #plt.scatter(feature_8[:, 0], feature_8[:, 1], c='g')
    #plt.scatter(feature_9[:, 0], feature_9[:, 1], c='cyan')
    #plt.scatter(feature_10[:, 0], feature_10[:, 1], c='indigo')
    #plt.scatter(feature_11[:, 0], feature_11[:, 1], c='brown')
    #plt.scatter(feature_12[:, 0], feature_12[:, 1], c='darkgray')
    #plt.scatter(features[13][:, 0], features[13][:, 1], c='olive')
    #plt.scatter(features[14][:, 0], features[14][:, 1], c='purple')
    #plt.scatter(features[15][:, 0], features[15][:, 1], c='sienna')
    #plt.scatter(features[24][:, 0], features[24][:, 1], c='green')
    #plt.scatter(feature_8[:, 0], feature_8[:, 1], c='violet')
    #plt.scatter(features_pca[:, 0], features_pca[:, 1], c=dec_labels)
#     plt.figure(figsize=(15, 15))
#     plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=dec_labels)
    #plt.show()
    
    print("AVERAGE POS DIST = ", np.mean(pos_man))
    print("AVERAGE NEG DIST = ", np.mean(neg_man))
    print("AVERAGE MARGIN   = ", np.mean(neg_man) - np.mean(pos_man))

if __name__ == "__main__":
    main(sys.argv[1:])
