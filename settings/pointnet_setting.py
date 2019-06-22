#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 24.12.2017
'''


class Setting():

    # data - related
    points_num = 1024
    augment_rotate_clouds = True             # Along up axis
    augment_jitter_points = True
    augment_shuffle_points = True
    augment_shuffle_clouds = True
    dataset_sampling_method = 'fps'

    # Backbone model - related
    backbone_model = 'pointnet_basic'

    # Siamese learning - related
    distance_metric = 'euclidian'

    # learning - related
    classes_no_in_batch = 8
    instances_no_in_batch = 4

    gradient_clip = -1
    learning_rate = 0.001
    training_iterations = 100000
    save_model_after_iterations = 25000
    save_summary_after_iterations = 1000
    device = '/device:GPU:0'

