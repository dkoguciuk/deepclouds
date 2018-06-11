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
from sklearn.svm import SVC
from sklearn import linear_model
from scipy import spatial

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

    # model
    #model_embeddings = class_embeddings[:,0]
    model_embeddings = np.mean(class_embeddings, axis=1)
    #print model_embeddings[0], class_embeddings[0][0]
    
    for instance_idx in range(1,10):
        dist = []
        for idx in range(0,40):
            #dist.append(spatial.distance.cosine(class_embeddings[0][instance_idx],model_embeddings[idx]))
            dist.append(np.linalg.norm(class_embeddings[0][instance_idx] - model_embeddings[idx]))
        print (np.argmin(dist),
               np.linalg.norm(class_embeddings[0][instance_idx] - model_embeddings[0]),
               np.linalg.norm(class_embeddings[0][instance_idx] - model_embeddings[np.argmin(dist)]))
    exit()
                  
    
    class_dist = np.linalg.norm(model_embeddings[0] - class_embeddings, axis=-1)
    for class_idx in range(1, 40):
        class_dist.append()
    
    print len(class_dist)






    exit()



    # SVC
    class_embeddings = class_embeddings.reshape(-1, 1024)
    labels = np.zeros((40*10))
    for idx in range(40*10):
        labels[idx] = idx/10
    
    print class_embeddings.shape
    print labels.shape 
    clf = SVC()
    clf.fit(class_embeddings, labels)
    
    pred = clf.predict(class_embeddings)
    print float(sum(pred == labels)) / labels.shape[0]
    exit()
    
    print clf.predict(class_embeddings[0].reshape(1, -1)), labels[0]
    print clf.predict(class_embeddings[1].reshape(1, -1)), labels[1]
    print clf.predict(class_embeddings[2].reshape(1, -1)), labels[2]
    print clf.predict(class_embeddings[3].reshape(1, -1)), labels[3]
    print clf.predict(class_embeddings[4].reshape(1, -1)), labels[4]
    print clf.predict(class_embeddings[5].reshape(1, -1)), labels[5]
    exit()
    
    print clf.predict(class_embeddings[:,0])
    print clf.predict(class_embeddings[:,1])
    clf.fit(class_embeddings[:,1], np.arange(40))
    print clf.predict(class_embeddings[:,0])
    print clf.predict(class_embeddings[:,1])
     
    exit()
    

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
        print "POSITIVE DIST FOR CLASS {0:02d} calculated".format(class_idx + 1)
        
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
        print "NEGATIVE DIST FOR CLASS {0:02d} calculated".format(class_idx_1 + 1)
    
    print "AVERAGE POS DIST = ", np.mean(pos_man)
    print "AVERAGE NEG DIST = ", np.mean(neg_man)
    print "AVERAGE MARGIN   = ", np.mean(neg_man) - np.mean(pos_man)

if __name__ == "__main__":
    main(sys.argv[1:])
