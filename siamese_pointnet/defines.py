#!/usr/bin/env python
# # -*- coding: utf-8 -*-


"""
This is the module of siamese_pointnet introducing all const defines
used in the package.
"""


__author__ = "Daniel Koguciuk and Rafał Koguciuk"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Daniel Koguciuk"
__email__ = "daniel.koguciuk@gmail.com"
__status__ = "Development"


import os


ROOT_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
"""
The root directory of the siamiese_pointnet package.
"""

DATA_DIR = os.path.join(ROOT_DIR, "data")
"""
Data directory for the modelnet data to be downloaded and stored.
"""

DATA_URL = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
"""
URL to the modelnet data.
"""

DATA_MODELNET_DIR = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')
"""
Extracted modelnet data directory.
"""