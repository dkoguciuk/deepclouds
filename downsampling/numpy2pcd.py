#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Daniel Koguciuk <daniel.koguciuk@gmail.com>
@note: Created on 01.06.2018
'''

import pypcd
import numpy as np

# filepaths
npy_filepath = '../data/synthetic/train/0001_01.npy'
pcd_filepath = '../data/pcd/0000_01.pcd'
augment = False

# load
a = np.load(npy_filepath)

# augment
if augment:
  dist = a[1:] - a[:-1]
  new_points = []
  for idx in range(dist.shape[0]):
    act_dist = dist[idx]
    for ins in range(1, 10):
      incr = act_dist * ins / 10
      new_points.append(a[idx] + incr)

if augment:
  b = np.array(new_points)
  c = np.concatenate((a, b), axis=0)
else:
  c= a

# Save as pcd
cloud_out = pypcd.make_xyz_point_cloud(c)
cloud_out.save_pcd(pcd_filepath)


