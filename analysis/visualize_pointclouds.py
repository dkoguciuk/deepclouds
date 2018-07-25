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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
import deepclouds.defines as df
import deepclouds.modelnet_data as modelnet
from deepclouds.classifiers import MLPClassifier
from deepclouds.model import DeepCloudsModel

CLOUD_SIZE = 1024
SAMPLING_METHOD = 'random'

def visualize_pointclouds():

    # Generate data if needed
    data_gen = modelnet.ModelnetData(pointcloud_size=CLOUD_SIZE)
    
    # Modelnet generator
    index = 0
    for clouds, labels in data_gen.generate_representative_batch(train=False,
                                                                 instances_number=1,
                                                                 shuffle_points=True,
                                                                 jitter_points=True,
                                                            rotate_pointclouds=False,
                                                                 rotate_pointclouds_up=True,
                                                                 sampling_method=SAMPLING_METHOD):

        # Save as pcd
        cloud_out = pypcd.make_xyz_point_cloud(clouds[15])
        if index < 10:
            cloud_out.save_pcd("flowerpot_0" + str(index) + ".pcd")
        else:
            cloud_out.save_pcd("flowerpot_" + str(index) + ".pcd")
        index = index + 1
        
            

def main(argv):

    # save embeddings
    visualize_pointclouds()

    # Print all settings at the end of learning
    print "DONE!"

if __name__ == "__main__":
    main(sys.argv[1:])
