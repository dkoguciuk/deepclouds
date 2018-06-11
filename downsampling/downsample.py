#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 01.06.2018
'''
import os
import sys
import pypcd
import numpy as np
import pointcloud_downsample

# Import deepclouds
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import deepclouds.modelnet_data as modelnet_data

idx = 2

# filepaths
pcd_filepath_in = '../data/pcd/modelnet_orig.pcd'
pcd_filepath_out_unif = '../data/pcd/modelnet_unif.pcd'
pcd_filepath_out_rand = '../data/pcd/modelnet_rand.pcd'
pcd_filepath_out_grph = '../data/pcd/modelnet_grph.pcd'

data_gen = modelnet_data.ModelnetData(2048)
for X, Y in data_gen.generate_random_batch(True):

  # Downsample 
  output_pointcloud_raw_uni = pointcloud_downsample.uniform(X[idx])
  output_pointcloud_raw_grp = pointcloud_downsample.via_graphs(X[idx])
  print "OLD SHAPE          : ", X[idx].shape
  print "NEW SHAPE [UNIFORM]: ", output_pointcloud_raw_uni.shape
  print "NEW SHAPE [VIA_GRP]: ", output_pointcloud_raw_grp.shape

  # Save as pcd (original)
  cloud_in = pypcd.make_xyz_point_cloud(X[idx])
  cloud_in.save_pcd(pcd_filepath_in)

  # Save as pcd (uniform)
  cloud_out = pypcd.make_xyz_point_cloud(output_pointcloud_raw_uni)
  cloud_out.save_pcd(pcd_filepath_out_unif)

  # Save as pcd (via_graph)
  cloud_out = pypcd.make_xyz_point_cloud(output_pointcloud_raw_grp)
  cloud_out.save_pcd(pcd_filepath_out_grph)

  # Save as pcd (random)
  cloud_point_idxs = np.arange(len(X[idx]))
  cloud_randm_idxs = np.random.choice(cloud_point_idxs, 128, replace=False)
  cloud_out_rand = X[idx][cloud_randm_idxs]
  cloud_out_rand = pypcd.make_xyz_point_cloud(cloud_out_rand)
  cloud_out_rand.save_pcd(pcd_filepath_out_rand)  

  break


