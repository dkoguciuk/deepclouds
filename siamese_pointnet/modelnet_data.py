#!/usr/bin/env python
# # -*- coding: utf-8 -*-


"""
This is the module of siamese_pointnet implementing all functionality
connected with the data manipulation (modelnet/synthetic). 
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


class GenericData(object):
    """
    Generic (base) class for different input data manipulation. 
    """
    def __init__(self):
        """
        Default constructor, where os is checked.
        """
        # check platform
        if sys.platform != "linux" and sys.platform != "linux2":
            raise exceptions.OSError("Your OS is not supported, please switch to linux") 

    @staticmethod
    def _shuffle_data(data):
        """
        Shuffle the data and return its random permutation.

        Args:
            data (numpy.ndarray of size [N,3]): point clouds data to be shuffled along first axis
        Returns:
            (numpy.ndarray of size [N,3]): shuffled point cloud
        """
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        return data[idx, ...]

    @staticmethod
    def _shuffle_data_with_labels(data, labels):
        """
        Shuffle pointclouds data and labels.

        Args:
            data (numpy.ndarray of size [B,N,3]): point clouds data to be shuffled along first axis
            labels (numpy.ndarray of size [B,1]): labels of each pointcloud
        Returns:
            (numpy.ndarray of size [B,N,3], numpy.ndarray [B:1]): shuffled point cloud and shuffled labels
        """
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx], idx

    @staticmethod
    def _find_index_of_another_class(data, labels, index):
        """
        Find random index of a cloud with other label than specified with index.

        Args:
            data (numpy.ndarray of size [B,N,3]): point clouds data
            labels (numpy.ndarray of size [B,1]): labels of each pointcloud
            index (int): Index of a query cloud
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
        Randomly jitter points. Jittering is per point, but for all tuples in batch data. 
    
        Args:
            batch_tuples (3x np.ndarray of size [B,N,3]): Batch data with tripple of point clouds.
            sigma (float): Sigma value of gaussian noise to be applied pointwise.
            clip (float): Clipping value of gaussian noise.
        Returns:
              (3x np.ndarray of size [B,N,3]): Jittered pointclouds data. 
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
        batch_tuples += jittered_data
        return batch_tuples

    @staticmethod
    def _rotate_pointclouds_up(batch_tuples):
        """
        Randomly rotate the point clouds to augument the dataset -- the rotation is performed
        with random angle around random axis, but this axis has to contain (0,0) point.

        Args:
            batch_tuples (3x np.ndarray of size [B,N,3]): Batch data with tripple of point clouds.
        Returns:
              (3x np.ndarray of size [B,N,3]): Rotated pointclouds data.
        """
        # Get size
        if len(batch_tuples) != 3:
            raise exceptions.AssertionError("Batch should consist of tripples of pointclouds")
        if batch_tuples[0].shape != batch_tuples[1].shape or batch_tuples[0].shape != batch_tuples[2].shape:
            raise exceptions.AssertionError("Clouds in batch should be same size")

        # New batch data
        batch_rotated = (np.zeros(batch_tuples[0].shape, dtype=np.float32),
                         np.zeros(batch_tuples[1].shape, dtype=np.float32),
                         np.zeros(batch_tuples[1].shape, dtype=np.float32))
        for batch_idx, batch in enumerate(batch_tuples):
            for cloud_idx, cloud in enumerate(batch):
                rotation_angle = np.random.uniform() * 2 * np.pi
                cosval = np.cos(rotation_angle)
                sinval = np.sin(rotation_angle)
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])
                batch_rotated[batch_idx][cloud_idx] = np.dot(cloud.reshape((-1, 3)), rotation_matrix)
        return batch_rotated

    @staticmethod
    def _rotate_pointclouds_rand(batch_tuples):
        """
        Randomly rotate the point clouds to augument the dataset -- the rotation is performed
        with random angle around random axis, but this axis has to contain (0,0) point.
    
        Args:
            batch_tuples (3x np.ndarray of size [B,N,3]): Batch data with tripple of point clouds.
        Returns:
              (3x np.ndarray of size [B,N,3]): Rotated pointclouds data. 
        """
        # Get size
        if len(batch_tuples) != 3:
            raise exceptions.AssertionError("Batch should consist of tripples of pointclouds")
        if batch_tuples[0].shape != batch_tuples[1].shape or batch_tuples[0].shape != batch_tuples[2].shape: 
            raise exceptions.AssertionError("Clouds in batch should be same size")
        B, N, C = batch_tuples[0].shape
        
        raise exceptions.NotImplementedError("Rotating pointclouds not implemented yet!")

class ModelnetData(GenericData) : 
    """
    Class implementing all needed functionality with modelnet data manipulation.
    """

    def __init__(self):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        # super
        super(ModelnetData,self).__init__()
        # download data if it's needed
        if not os.path.exists(df.DATA_MODELNET_DIR):
            zipfile = os.path.basename(df.DATA_URL)
            os.system('wget %s; unzip %s' % (df.DATA_URL, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], df.DATA_DIR))
            os.system('rm %s' % (zipfile))
            print "Modelnet database downloading completed..."
        # internal vars
        self.train_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "train_files.txt"))]
        self.test_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "test_files.txt"))]

    def generate_train_tripples(self, batch_size,
                                shuffle_files=False, shuffle_pointclouds=False,
                                jitter_pointclouds=False, rotate_pointclouds_up=False,
                                rotate_pointclouds_rand=False, reshape_flags=[]):
        """ 
        Generator returns 3 point clouds A (anchor), P (positive), N (negative).
    
        Args:
            batch_size (int): Batch size we would split the dataset into (numer of triples
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle train files?
            shuffle_pointclouds (boolean): Should we shuffle pointclouds for each file?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds_up (boolean): Rotate pointclouds with random angle around up axis,
                but this axis has to contain (0,0) point.
            rotate_pointclouds_rand (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [batch_size, pointcloud_size, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [batch_size, N*3]
                transpose_pointclouds --  transpose flatten pointclouds to be shape of
                    [N*3, batch_size], this flag could be specified only with flatten_pointclouds.
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
                # get A, P, N
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

                # jitter
                if jitter_pointclouds:
                    batch_data = ModelnetData._jitter_pointclouds(batch_data)
                # rotate
                if rotate_pointclouds_up:
                    batch_data = ModelnetData._rotate_pointclouds_up(batch_data)
                # reshape
                if "flatten_pointclouds" in reshape_flags:
                    batch_data = (np.reshape(A, [batch_size, -1]),
                                  np.reshape(P, [batch_size, -1]),
                                  np.reshape(N, [batch_size, -1]))
                    if "transpose_pointclouds" in reshape_flags:
                        batch_data = (batch_data[0].T, batch_data[1].T, batch_data[2].T)

                # yield
                yield batch_data
    
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

class SyntheticData(GenericData):
    """
    Class implementing all needed functionality with synthetic data manipulation.
    """

    def __init__(self):
        """
        Default constructor.
        """
        # super
        super(SyntheticData,self).__init__()
        if not os.path.exists(df.DATA_SYNTHETIC_DIR):
            os.mkdir(df.DATA_SYNTHETIC_DIR)

    def check_generated_size(self):
        """
        Returns size of generated data [num_of_pointclouds:num_of_points] or None if synthetic
        data is not generated yet.

        Returns:
            ([num_of_pointclouds:num_of_points]): Size of generated data or None.
        """
        pointclouds_filenames = [filename for filename in os.listdir(df.DATA_SYNTHETIC_DIR) if filename.endswith(".npy")]
        if not pointclouds_filenames:
            return None
        pointcloud = np.load(os.path.join(df.DATA_SYNTHETIC_DIR, pointclouds_filenames[0]))
        return [len(pointclouds_filenames), pointcloud.shape[0]]

    def regenerate_files(self, pointcloud_size=32, pointclouds_amount=3200):
        """
        Delete all synthetic data if present and generate new with specified params.

        Args:
            pointcloud_size (int): Number of 3D points in the generated data.
            pointclouds_amount (int): How many pointclouds to generate.
        """
        # remove old
        pointclouds_filenames = [filename for filename in os.listdir(df.DATA_SYNTHETIC_DIR) if filename.endswith(".npy")]
        for pointcloud_filename in pointclouds_filenames:
            pointcloud_filepath = os.path.join(df.DATA_SYNTHETIC_DIR, pointcloud_filename)
            os.remove(pointcloud_filepath)
        
        # generate
        print "Generating %d synthetic pointclouds, each with %d 3D points..", pointclouds_amount, pointcloud_size
        for cloud_idx in range(0, pointclouds_amount):
            cloud_path = os.path.join(df.DATA_SYNTHETIC_DIR, format(cloud_idx, '04d') + ".npy")
            np.save(cloud_path, np.random.rand(pointcloud_size, 3))

    def generate_train_tripples(self, batch_size, shuffle_pointclouds=False,
                                jitter_pointclouds=False, rotate_pointclouds_up=False,
                                rotate_pointclouds_rand=False, reshape_flags=[]):
        """ 
        Generator returns 3 point clouds A (anchor), P (positive), N (negative).
    
        Args:
            batch_size (int): Batch size we would split the dataset into (numer of triples
                to be returned by the generator).
            shuffle_pointclouds (boolean): Should we shuffle pointclouds for each file?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds_up (boolean): Rotate pointclouds with random angle around up axis,
                but this axis has to contain (0,0) point.
            rotate_pointclouds_rand (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [batch_size, pointcloud_size, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [batch_size, N*3]
                transpose_pointclouds --  transpose flatten pointclouds to be shape of
                    [N*3, batch_size], this flag could be specified only with flatten_pointclouds.
        Returns:
            tuple(A, P, N): where:
                A - random permutation of next cloud
                P - another random permutation of the same cloud
                N - random permutation of cloud from another class
        """
        # trainfiles
        pointclouds_filepaths = [os.path.join(df.DATA_SYNTHETIC_DIR, filename) for filename
                                 in os.listdir(df.DATA_SYNTHETIC_DIR) if filename.endswith(".npy")]
        # shuffle train files
        if shuffle_pointclouds:
            np.random.shuffle(pointclouds_filepaths)

        # size
        pointclouds_size = self.check_generated_size()

        # iterate over all files
        for batch_idx in range(int(len(pointclouds_filepaths) / batch_size)):
            # load pointclouds
            A = np.empty([batch_size, pointclouds_size[1], 3])
            P = np.empty([batch_size, pointclouds_size[1], 3])
            N = np.empty([batch_size, pointclouds_size[1], 3])
            for cloud_idx in range(batch_size):
                # anchor & positive
                global_idx = batch_idx * batch_size + cloud_idx
                A[cloud_idx] = np.load(pointclouds_filepaths[global_idx])
                P[cloud_idx] = SyntheticData._shuffle_data(A[cloud_idx])
                # negative
                other_idx = random.randint(0, len(pointclouds_filepaths) - 1)
                while other_idx == global_idx:
                    other_idx = random.randint(0, len(pointclouds_filepaths) - 1)
                N[cloud_idx] = SyntheticData._shuffle_data(np.load(pointclouds_filepaths[other_idx]))
            batch_data = (A, P, N)

            # jitter
            if jitter_pointclouds:
                batch_data = SyntheticData._jitter_pointclouds(batch_data)
            # rotate
            if rotate_pointclouds_up:
                batch_data = SyntheticData._rotate_pointclouds_up(batch_data)
            # reshape
            if "flatten_pointclouds" in reshape_flags:
                batch_data = (np.reshape(A, [batch_size, -1]),
                              np.reshape(P, [batch_size, -1]),
                              np.reshape(N, [batch_size, -1]))
                if "transpose_pointclouds" in reshape_flags:
                    batch_data = (batch_data[0].T, batch_data[1].T, batch_data[2].T)

            # yield
            yield batch_data
