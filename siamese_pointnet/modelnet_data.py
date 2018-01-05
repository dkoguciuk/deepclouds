#!/usr/bin/env python
# # -*- coding: utf-8 -*-


"""
This is the module of siamese_pointnet implementing all functionality
connected with the modelnet data manipulation. 
"""


__author__ = "Daniel Koguciuk and Rafał Koguciuk"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Daniel Koguciuk"
__email__ = "daniel.koguciuk@gmail.com"
__status__ = "Development"


import os
import sys
import math
import h5py
import random
import exceptions
import numpy as np
import siamese_pointnet.defines as df


class ModelnetData(object):
    """
    Class implementing all needed functionality with modelnet data manipulation.
    """

    def __init__(self):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        # check platform
        if sys.platform != "linux" and sys.platform != "linux2":
            raise exceptions.OSError("Your OS is not supported, please switch to linux")
        # download data if it's needed
        if not os.path.exists(df.DATA_MODELNET_DIR):
            if not os.path.exists(df.DATA_DIR):
                os.mkdir(df.DATA_DIR)
            zipfile = os.path.basename(df.DATA_URL)
            os.system('wget %s; unzip %s' % (df.DATA_URL, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], df.DATA_DIR))
            os.system('rm %s' % (zipfile))
            print "Modelnet database downloading completed..."
        # internal vars
        self.train_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "train_files.txt"))]
        self.test_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "test_files.txt"))]

    def generate_train_tripples(self, batch_size, shuffle_files=False, shuffle_pointclouds=False):
        """ 
        Generator returns 3 point clouds A (anchor), P (positive), N (negative).
    
        Args:
            batch_size (int): Batch size we would split the dataset into (numer of triples
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle train files?
            shuffle_pointclouds (boolean): Should we shuffle pointclouds for each file?
        Returns:
            tuple(A, P, N): where:
                A - random permutation of next cloud
                P - another random permutation of the same cloud
                N - random permutation of cloud from another class
        """

        # shuffle train files
        train_file_idxs = np.arange(0, len(self.train_files))
        if shuffle_files:
            np.random.shuffle(train_file_idxs)

        # iterate over all files
        for idx in range(len(self.train_files)):
            # load pointclouds
            pointclouds, labels = ModelnetData._load_h5_file(self.train_files[train_file_idxs[idx]])
            # shuffle pointclouds
            if shuffle_pointclouds:
                pointclouds, labels, _ = ModelnetData._shuffle_data_with_labels(pointclouds, labels)
            # iterate over all batches
            for batch_idx in range(int(math.floor(pointclouds.shape[0] / batch_size))):
                A = np.empty([batch_size, pointclouds.shape[1], pointclouds.shape[2]])
                P = np.empty([batch_size, pointclouds.shape[1], pointclouds.shape[2]])
                N = np.empty([batch_size, pointclouds.shape[1], pointclouds.shape[2]])
                for cloud_idx in range(batch_size):
                    global_idx = batch_idx * batch_size + cloud_idx
                    other_cloud_idx = ModelnetData._find_index_of_another_class(pointclouds, labels, global_idx)
                    A[cloud_idx] = ModelnetData._shuffle_data(pointclouds[global_idx])
                    P[cloud_idx] = ModelnetData._shuffle_data(pointclouds[global_idx])
                    N[cloud_idx] = ModelnetData._shuffle_data(pointclouds[other_cloud_idx]) 
                yield A, P, N
    
    @staticmethod
    def _get_filenames(filepath):
        """
        Get listed filenames listed in the file.

        Args:
            filepath (str): Path to the file containing other files as the following lines.
        """
        with open(filepath) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    @staticmethod
    def _load_h5_file(filepath):
        """
        Load modelnet data from h5 file.

        Args:
            filepath (str): Path to the modelnet h5 file.
        """
        f = h5py.File(filepath)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    @staticmethod
    def _shuffle_data_with_labels(data, labels):
        """
        Shuffle pointclouds data and labels.

        Args:
            data (numpy.ndarray): point clouds data of size [number_of_pointclouds,
                number_of_points_in_cloud, numer_of_coord_of_one_point]
            labels (numpy.ndarray): labels of each pointcloud of size
                [number_of_pointclouds, 0]
        Returns:
            (numpy.ndarray), (numpy.ndarray), (numpy.array): shuffled point cloud,
                shuffled labels and shuffle indices
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx], idx

    @staticmethod
    def _shuffle_data(data):
        """
        Shuffle the data and returned its random permutation
        (WARNING: The original data would be copied!).

        Args:
            data (numpy.ndarray): point clouds data of size [number_of_pointclouds,
                number_of_points_in_cloud, numer_of_coord_of_one_point] to be shuffled
                against first dimension.
        Returns:
            (numpy.ndarray): shuffled point cloud
        """
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        return data[idx, ...]

    @staticmethod
    def _find_index_of_another_class(data, labels, index):
        """
        Find random index of a cloud with other label than specified with index.

        Args:
            data (numpy.ndarray): point clouds data of size [number_of_pointclouds,
                number_of_points_in_cloud, numer_of_coord_of_one_point]
            labels (numpy.ndarray): labels of each pointcloud of size
                [number_of_pointclouds, 0]
            index (int): Index of a query cloud.
        Returns:
            (int): Index of a cloud with other label than specified with index arg.
        """
        while True:
            j = random.randint(0, data.shape[0] - 1)
            if labels[j][0] != labels[index][0]:
                return j
