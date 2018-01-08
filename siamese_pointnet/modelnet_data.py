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

    def generate_train_tripples(self, batch_size, shuffle_files=False, shuffle_pointclouds=False,
                                jitter_pointclouds=False, rotate_pointclouds=False):
        """ 
        Generator returns 3 point clouds A (anchor), P (positive), N (negative).
    
        Args:
            batch_size (int): Batch size we would split the dataset into (numer of triples
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle train files?
            shuffle_pointclouds (boolean): Should we shuffle pointclouds for each file?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around axis,
                but this axis has to contain (0,0) point.
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
                batch_data = (A, P, N)
                if jitter_pointclouds:
                    batch_data = ModelnetData._jitter_pointclouds((A, P, N))
                if rotate_pointclouds:
                    batch_data = ModelnetData._rotate_pointclouds((A, P, N))
                return batch_data
    
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

    @staticmethod
    def _jitter_pointclouds(batch_tuples, sigma=0.01, clip=0.05):
        """
        Randomly jitter points. jittering is per point, but for all tuples in batch data. 
    
        Args:
            batch_tuples (3x np.ndarray of size BxNx3): Batch data with tripple of point clouds.
            sigma (float): Sigma value of gaussian noise to be applied pointwise.
            clip (float): Clipping value of gaussian noise.
        Returns:
              (3x np.ndarray of size BxNx3): Jittered pointclouds data. 
        """
        # Get size
        if len(batch_tuples) != 3:
            raise exceptions.AssertionError("Batch should consist of tripples of pointclouds")
        if batch_tuples[0].shape != batch_tuples[1].shape or batch_tuples[0].shape != batch_tuples[2].shape: 
            raise exceptions.AssertionError("Clouds in batch should be same size")
        B, N, C = batch_tuples[0].shape
        
        # Generate noise
        if clip <= 0:
            raise exceptions.ValueError("Clip should be a positive number")
        jittered_data = np.clip(sigma * np.random.randn(3, B, N, C), -1 * clip, clip)
        
        # Add to pointcloud
        jittered_data += batch_tuples
        return jittered_data

    @staticmethod
    def _rotate_pointclouds(batch_tuples):
        """
        Randomly rotate the point clouds to augument the dataset -- the rotation is performed
        with random angle around random axis, but this axis has to contain (0,0) point.
    
        Args:
            batch_tuples (3x np.ndarray of size BxNx3): Batch data with tripple of point clouds.
        Returns:
              (3x np.ndarray of size BxNx3): Rotated pointclouds data. 
        """
        # Get size
        if len(batch_tuples) != 3:
            raise exceptions.AssertionError("Batch should consist of tripples of pointclouds")
        if batch_tuples[0].shape != batch_tuples[1].shape or batch_tuples[0].shape != batch_tuples[2].shape: 
            raise exceptions.AssertionError("Clouds in batch should be same size")
        B, N, C = batch_tuples[0].shape
        
        raise exceptions.NotImplementedError("Rotating pointclouds not implemented yet!")
        
#         rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
#         for k in range(batch_data.shape[0]):
#             rotation_angle = np.random.uniform() * 2 * np.pi
#             cosval = np.cos(rotation_angle)
#             sinval = np.sin(rotation_angle)
#             rotation_matrix = np.array([[cosval, 0, sinval],
#                                         [0, 1, 0],
#                                         [-sinval, 0, cosval]])
#             shape_pc = batch_data[k, ...]
#             rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
#         return rotated_data