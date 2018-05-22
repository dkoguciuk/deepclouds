#!/usr/bin/env python
# # -*- coding: utf-8 -*-


"""
This is the module of deepclouds implementing all functionality
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
import downsample_uniform
import numpy.core.umath_tests as nm
import deepclouds.defines as df


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
    def _shuffle_points_in_batch(batch):
        """
        Shuffle points in the pointclouds and return its random permutation.

        Args:
            batch (np.ndarray of size [B, X, N, 3]): Point clouds stacked in single batch
                would be shuffled along third axis.
        Returns:
            (numpy.ndarray of size [B, X, N,3]): shuffled point clouds.
        """
        shape = batch.shape
        pointclouds = np.reshape(batch, [-1, shape[2], shape[3]])
        for cloud_idx in range(shape[0]*shape[1]):
            idx = np.arange(pointclouds[cloud_idx].shape[0])
            np.random.shuffle(idx)
            pointclouds[cloud_idx] = pointclouds[cloud_idx][idx, :]
        return np.reshape(pointclouds, shape)

    @staticmethod
    def _jitter_batch(batch, sigma=0.01, clip=0.05):
        """
        Randomly jitter points. Jittering is per point, but for pointclouds in the batch. 
    
        Args:
            batch (np.ndarray of size [B, X, N, 3]): Point clouds stacked in single batch.
            sigma (float): Sigma value of gaussian noise to be applied pointwise.
            clip (float): Clipping value of gaussian noise.
        Returns:
              (np.ndarray of size [B, 3, N, 3]): Jittered pointclouds data. 
        """
        # Get size
        B, X, N, C = batch.shape
        
        # Generate noise
        if clip <= 0:
            raise exceptions.ValueError("Clip should be a positive number")
        jittered_data = np.clip(sigma * np.random.randn(B, X, N, C), -1 * clip, clip)
        
        # Add to pointcloud
        batch += jittered_data
        return batch
    
    @staticmethod
    def _rotate_pointcloud(pointcloud):
        """
        Randomly rotate the point cloud to augument the dataset -- the rotation is performed
        with random angle around random axis, but this axis has to contain (0,0) point.
    
        Args:
            pointcloud (np.ndarray of size [N, 3]): Pointcloud to be rotated.
        Returns:
            (np.ndarray of size [N, 3]): Rotated pointcloud data. 
        """
        # Rotation params
        theta = np.random.uniform() * 2 * np.pi         # Get random rotation
        axis = np.random.uniform(size=3)                # Get random axis
        axis /= np.linalg.norm(axis)                    # Normalize it
        # Rodrigues' rotation formula (see wiki for more)
        pointcloud = (pointcloud * np.cos(theta) +
                      np.cross(axis, pointcloud) * np.sin(theta) +
                      axis * nm.inner1d(axis, pointcloud).reshape(-1, 1) * (1 - np.cos(theta)))
        return pointcloud

    @staticmethod
    def _rotate_pointcloud_up(pointcloud):
        """
        Randomly rotate the point cloud to augument the dataset -- the rotation is performed
        with random angle around up axis.
    
        Args:
            pointcloud (np.ndarray of size [N, 3]): Pointcloud to be rotated.
        Returns:
            (np.ndarray of size [N, 3]): Rotated pointcloud data. 
        """
        # Rotation params
        theta = np.random.uniform() * 2 * np.pi         # Get random rotation
        cosval = np.cos(theta)                          # Cos of rot angle
        sinval = np.sin(theta)                          # Sin of rot angle
        rotation_matrix = np.array([[cosval, 0, sinval],    # rot matrix
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        pointcloud = np.dot(pointcloud.reshape((-1, 3)), rotation_matrix)
        return pointcloud

    @staticmethod
    def _rotate_batch(batch, random_axis=True):
        """
        Randomly rotate the point clouds to augument the dataset -- the rotation is performed
        with random angle around random axis, but this axis has to contain (0,0) point.
    
        Args:
            batch (np.ndarray of size [B, X, N, 3]): Batch data with point clouds.
            random_axis (bool): If true rotation will be perform around random axis, if false
                will be perfomed around up axis. 
        Returns:
            (np.ndarray of size [B, X, N, 3]): Rotated pointclouds data. 
        """
        shape = batch.shape
        batch = np.reshape(batch, [-1, shape[2], shape[3]])
        for cloud_idx in range(shape[0]*shape[1]):
            if random_axis:
                batch[cloud_idx] = GenericData._rotate_pointcloud(batch[cloud_idx])
            else:
                batch[cloud_idx] = GenericData._rotate_pointcloud_up(batch[cloud_idx])
        return np.reshape(batch, shape)

class ModelnetData(GenericData) : 
    """
    Class implementing all needed functionality with modelnet data manipulation. The generator
    returns 3 pointclouds: anchor, permuted anchor and 
    """
    
    CLASSES_COUNT = 40
    """
    How many classes do we have in the modelnet dataset.
    """

    def __init__(self, pointcloud_size):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        
        # super
        super(ModelnetData, self).__init__()
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
        self.pointcloud_size = pointcloud_size

    def generate_random_tripples(self, batch_size,
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
        raise NotImplemented("For a long time this class was not tested!")

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
                    batch_data = ModelnetData._jitter_pointclouds_tuples(batch_data)
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

    def generate_representative_batch(self,
                                      train,
                                      instances_number=2,
                                      shuffle_files=True,
                                      shuffle_points=False,
                                      jitter_points=False,
                                      rotate_pointclouds=False,
                                      rotate_pointclouds_up=False,
                                      reshape_flags=[]):
        """
        Take pointclouds with at least instances_number of each class.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            instances_number (int): Size of a batch expressed in instances_number of each CLASSES_COUNT,
                resulting in batch_size equals instances_number * CLASSES_COUNT. Please assume every
                object should appear at least two times in the batch, so recommended batches_number is 2.
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_points (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, 3 N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, 3, N*3]
        Returns:
            (np.ndarray of size [B, N, 3], np.ndarray of size [B, N]): Representative batch and it's labels.
        """
        # Train or test dir?
        if train:
            pointcloud_files = self.train_files
        else:
            pointcloud_files = self.test_files
        
        # shuffle train files
        pointcloud_files_idxs = np.arange(0, len(pointcloud_files))
        if shuffle_files:
            np.random.shuffle(pointcloud_files_idxs)
 
         # iterate over all files
        for idx in range(len(pointcloud_files)):
            # load pointclouds from file
            file_pointclouds, file_labels = ModelnetData._load_h5_file(pointcloud_files[pointcloud_files_idxs[idx]])
            # shuffle pointclouds from file
            if shuffle_files:
                file_pointclouds, file_labels, _ = ModelnetData._shuffle_data_with_labels(file_pointclouds, file_labels)
            file_labels = np.squeeze(file_labels)
            # get sorted pointclouds along labels
            pointclouds = { k : [] for k in range(self.CLASSES_COUNT)}
            for cloud_idx in range(file_labels.shape[0]):
                pointclouds[file_labels[cloud_idx]].append(file_pointclouds[cloud_idx])
                
            # iterate over all batches in this file
            for _ in range(int(math.floor(file_pointclouds.shape[0] / self.CLASSES_COUNT / instances_number))):
                
                batch_clouds = []
                batch_labels = []
                for instance_idx in range(instances_number):
                    for class_idx in range(self.CLASSES_COUNT):
                        cloud_class_idx = np.random.choice(np.arange(len(pointclouds[class_idx])), 1)[0]
                        batch_clouds.append(pointclouds[class_idx][cloud_class_idx])
                        batch_labels.append(class_idx)

                # stack
                batch_clouds = np.stack(batch_clouds, axis=0)
                batch_clouds = np.stack([batch_clouds], axis=1)
                batch_labels = np.stack(batch_labels, axis=0)

                # RESIZE
                if self.pointcloud_size < 2048:
                    batch_clouds_resized = []
                    for cloud_idx in range(len(batch_clouds)):
    #                     cloud = downsample_uniform.downsample_uniform(batch_clouds[cloud_idx][0])
    #                     sum_x = float(np.sum(cloud[:,0], axis=1)) / cloud.shape[0]
    #                     sum_y = float(np.sum(cloud[:,1], axis=1)) / cloud.shape[0]
    #                     sum_z = float(np.sum(cloud[:,2], axis=1)) / cloud.shape[0]
    #                     print sum_x, sum_y, sum_z
    #                     exit()
                        
                        batch_clouds_resized.append(downsample_uniform.downsample_uniform(batch_clouds[cloud_idx][0]))
                        #cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        #cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        #batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    batch_clouds = np.stack(batch_clouds_resized, axis=0)
                    batch_clouds = np.stack([batch_clouds_resized], axis=1)

                # shuffle points
                if shuffle_points:
                    batch_clouds = SyntheticData._shuffle_points_in_batch(batch_clouds)
                # jitter
                if jitter_points:
                    batch_clouds = SyntheticData._jitter_batch(batch_clouds)
                # rotate
                if rotate_pointclouds:
                    batch_clouds = SyntheticData._rotate_batch(batch_clouds)
                elif rotate_pointclouds_up:
                    batch_clouds = SyntheticData._rotate_batch(batch_clouds, random_axis=False)
                # reshape
                if "flatten_pointclouds" in reshape_flags:
                    shape = batch_clouds.shape
                    batch_clouds = np.reshape(batch_clouds, [shape[0], shape[1], -1])
    
                # yield
                yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_random_batch(self, train, batch_size=64, shuffle_files=False,
                              shuffle_points=False, jitter_points=False,
                              rotate_pointclouds=False, rotate_pointclouds_up=False,
                              reshape_flags=[]):
        """ 
        Take random pointcloud, apply optional operations on each pointclouds and return 
        batch of such pointclouds with labels.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            batch_size (int): Batch size we would split the dataset into (numer of triplets
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle files with pointclouds?
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, N*3]
        Returns:
            (np.ndarray of shape [B, N, 3]), where B: batch size, N: number of points in each
                pointcloud, 3: number of dimensions of each pointcloud (x,y,z).
        """
        # Train or test dir?
        if train:
            pointcloud_files = self.train_files
        else:
            pointcloud_files = self.test_files
        
        # shuffle train files
        pointcloud_files_idxs = np.arange(0, len(pointcloud_files))
        if shuffle_files:
            np.random.shuffle(pointcloud_files_idxs)

        # iterate over all files
        for idx in range(len(pointcloud_files)):
            # load pointclouds from file
            file_pointclouds, file_labels = ModelnetData._load_h5_file(pointcloud_files[pointcloud_files_idxs[idx]])
            # shuffle pointclouds from file
            if shuffle_files:
                file_pointclouds, file_labels, _ = ModelnetData._shuffle_data_with_labels(file_pointclouds, file_labels)
            # iterate over all batches in this file
            for batch_idx in range(int(math.floor(file_pointclouds.shape[0] / batch_size))):
                
                # Get pointclouds and labels
                batch_clouds = file_pointclouds[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_clouds = np.stack([batch_clouds], axis=1)
                batch_labels = file_labels[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_labels = np.squeeze(batch_labels)

                # RESIZE
                if self.pointcloud_size < 2048:
                    batch_clouds_resized = []
                    for cloud_idx in range(len(batch_clouds)):
                        batch_clouds_resized.append(downsample_uniform.downsample_uniform(batch_clouds[cloud_idx][0]))
                        #cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        #cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        #batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    batch_clouds = np.stack(batch_clouds_resized, axis=0)
                    batch_clouds = np.stack([batch_clouds], axis=1)

                # shuffle points
                if shuffle_points:
                    batch_clouds = SyntheticData._shuffle_points_in_batch(batch_clouds)
                # jitter
                if jitter_points:
                    batch_clouds = SyntheticData._jitter_batch(batch_clouds)
                # rotate
                if rotate_pointclouds:
                    batch_clouds = SyntheticData._rotate_batch(batch_clouds)
                elif rotate_pointclouds_up:
                    batch_clouds = SyntheticData._rotate_batch(batch_clouds, random_axis=False)
                # reshape
                if "flatten_pointclouds" in reshape_flags:
                    shape = batch_clouds.shape
                    batch_clouds = np.reshape(batch_clouds, [shape[0], shape[1], -1])
    
                # yield
                yield np.squeeze(batch_clouds, axis=1), batch_labels

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

    CLASSES_COUNT = 40
    """
    How many classes do we have in the modelnet dataset.
    """

    def __init__(self, pointcloud_size):
        """
        Default constructor.

        Args:
            pointcloud_size (int): Number of 3D points in the generated data.
        """
        # super
        super(SyntheticData, self).__init__()

        if not os.path.exists(df.DATA_SYNTHETIC_DIR):
            os.mkdir(df.DATA_SYNTHETIC_DIR)

        self.train_dir_path = os.path.join(df.DATA_SYNTHETIC_DIR, "train")
        if not os.path.exists(self.train_dir_path):
            os.mkdir(self.train_dir_path)

        self.test_dir_path = os.path.join(df.DATA_SYNTHETIC_DIR, "test")
        if not os.path.exists(self.test_dir_path):
            os.mkdir(self.test_dir_path)

        self.pointcloud_size = pointcloud_size
        actual_pc_size = self.check_generated_size()
        if not actual_pc_size or actual_pc_size[1] != self.pointcloud_size:
            self.regenerate_files()

    def check_generated_size(self, train=True):
        """
        Returns size of generated data [num_of_pointclouds:num_of_points] or None if synthetic
        data is not generated yet.

        Args:
            train (bool): Do you want to know train or test dataset size?

        Returns:
            ([num_of_pointclouds:num_of_points]): Size of generated data or None.
        """
        if train:
            directory = self.train_dir_path
        else: 
            directory = self.test_dir_path

        pointclouds_filenames = [filename for filename in os.listdir(directory) if filename.endswith(".npy")]
        if not pointclouds_filenames:
            return None
        pointcloud = np.load(os.path.join(directory, pointclouds_filenames[0]))
        return [len(pointclouds_filenames), pointcloud.shape[0]]

    def regenerate_files(self, instances_per_class_train=100, instances_per_class_test=10):
        """
        Delete all synthetic data if present and generate new with specified params.

        Args:
            pointcloud_size (int): Number of 3D points in the generated data.
            pointclouds_amount (int): How many pointclouds to generate.
        """
        # remove old
        for directory in [self.train_dir_path, self.test_dir_path]: 
            pointclouds_filenames = [filename for filename in os.listdir(directory) if filename.endswith(".npy")]
            for pointcloud_filename in pointclouds_filenames:
                pointcloud_filepath = os.path.join(directory, pointcloud_filename)
                os.remove(pointcloud_filepath)
        
        # Generate model classes
        clouds = []
        for cloud_idx in range(0, self.CLASSES_COUNT):
            cloud = np.random.rand(self.pointcloud_size, 3)     # generate random
            cloud -= np.mean(cloud, axis=0)                     # zero the mean
            cloud /= np.max(np.linalg.norm(cloud, axis=1))      # normalize to unit sphere
            clouds.append(cloud)
        clouds = np.stack(clouds)                               # [B, N, 3]
        clouds = np.stack([clouds], axis=1)                     # [B, 1, N, 3]
        
        # Augment dataset and save (train)
        print "Generating %d synthetic pointclouds, each with %d 3D points.." % (instances_per_class_train*self.CLASSES_COUNT, self.pointcloud_size)
        for instance_idx in range(0, instances_per_class_train):
            clouds_new = np.copy(clouds)
            clouds_new = self._rotate_batch(clouds_new, random_axis=False)                 # rotate along random axis and random angle
            #clouds_new = self._rotate_batch(clouds_new, random_axis=True)                 # rotate along random axis and random angle
            clouds_new = self._jitter_batch(clouds_new)                 # jitter points
            clouds_new = self._shuffle_points_in_batch(clouds_new)      # shuffle point in the pointcloud
            for cloud_idx in range(self.CLASSES_COUNT):                 # save pointclouds
                global_idx = instance_idx * self.CLASSES_COUNT + cloud_idx
                cloud_path = os.path.join(self.train_dir_path, format(global_idx, '04d') + '_' + format(cloud_idx, '02d') + '.npy')
                np.save(cloud_path, clouds_new[cloud_idx][0])

        # Augment dataset and save (test)
        for instance_idx in range(0, instances_per_class_test):
            clouds_new = np.copy(clouds)
            clouds_new = self._rotate_batch(clouds_new, random_axis=False)                 # rotate along random axis and random angle
            #clouds_new = self._rotate_batch(clouds_new, random_axis=True)                 # rotate along random axis and random angle
            clouds_new = self._jitter_batch(clouds_new)                 # jitter points
            clouds_new = self._shuffle_points_in_batch(clouds_new)      # shuffle point in the pointcloud
            for cloud_idx in range(self.CLASSES_COUNT):                 # save pointclouds
                global_idx = instance_idx * self.CLASSES_COUNT + cloud_idx
                cloud_path = os.path.join(self.test_dir_path, format(global_idx, '04d') + '_' + format(cloud_idx, '02d') + '.npy')
                np.save(cloud_path, clouds_new[cloud_idx][0])

    def generate_random_triplets(self, train, batch_size, shuffle_files=False, 
                                 jitter_pointclouds=False, rotate_pointclouds=False,
                                 reshape_flags=[]):
        """ 
        Take random pointcloud (anchor), shuffle it (positive) and take random pointcloud
        from other class (negative) and return  batch of such triplets.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            batch_size (int): Batch size we would split the dataset into (numer of triplets
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle files with pointclouds?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, 3 N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, 3, N*3]
        Returns:
            (np.ndarray of shape [B, 3, N, 3]), where B: batch size, 3: triplets of pointclouds
                anchor, positive, negative, N: number of points in each pointcloud, 3: number of
                dimensions of each pointcloud (x,y,z).
        """
        # Train or test dir?
        if train:
            pointclouds_dir = self.train_dir_path
        else:
            pointclouds_dir = self.test_dir_path

        # Get filepaths
        pointclouds_filepaths = [os.path.join(pointclouds_dir, filename) for filename
                                 in os.listdir(pointclouds_dir) if filename.endswith(".npy")]

        # shuffle files?
        if shuffle_files:
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
                global_class = pointclouds_filepaths[pointclouds_filepaths[global_idx].find("_"):]
                other_idx = random.randint(0, len(pointclouds_filepaths) - 1)
                other_class = pointclouds_filepaths[pointclouds_filepaths[other_idx].find("_"):]
                while other_class == global_class:
                    other_idx = random.randint(0, len(pointclouds_filepaths) - 1)
                    other_class = pointclouds_filepaths[pointclouds_filepaths[other_idx].find("_"):]
                N[cloud_idx] = SyntheticData._shuffle_data(np.load(pointclouds_filepaths[other_idx]))
            batch_data = np.stack([A,P,N], axis=1)

            # jitter
            if jitter_pointclouds:
                batch_data = SyntheticData._jitter_batch(batch_data)
            # rotate
            if rotate_pointclouds:
                batch_data = SyntheticData._rotate_batch(batch_data)
            # reshape
            if "flatten_pointclouds" in reshape_flags:
                shape = batch_data.shape
                batch_data = np.reshape(batch_data, [shape[0], shape[1], -1])

            # yield
            yield batch_data

    def generate_representative_batch(self,
                                      train,
                                      instances_number=2,
                                      shuffle_points=False,
                                      jitter_points=False,
                                      rotate_pointclouds=False,
                                      rotate_pointclouds_up=False,
                                      reshape_flags=[]):
        """
        Take pointclouds with at least instances_number of each class.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            instances_number (int): Size of a batch expressed in instances_number of each CLASSES_COUNT,
                resulting in batch_size equals instances_number * CLASSES_COUNT. Please assume every
                object should appear at least two times in the batch, so recommended batches_number is 2.
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_points (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, 3 N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, 3, N*3]
        Returns:
            (np.ndarray of size [B, N, 3], np.ndarray of size [B, N]): Representative batch and it's labels.
        """
        # Train or test dir?
        if train:
            pointclouds_dir = self.train_dir_path
        else:
            pointclouds_dir = self.test_dir_path

        # Get filepaths
        pointclouds_filepaths = [os.path.join(pointclouds_dir, filename) for filename
                                 in os.listdir(pointclouds_dir) if filename.endswith(".npy")]

        # sort
        pointclouds_indices = []
        for filepath in pointclouds_filepaths:
            start_idx = filepath.rfind("/") + 1
            stop_idx = filepath.rfind("_")
            pointclouds_indices.append(int(filepath[start_idx:stop_idx]))
        filpaths_sorted = [path for _,path in sorted(zip(pointclouds_indices, pointclouds_filepaths))]
        
        # filepaths
        instances_count = len(filpaths_sorted)/self.CLASSES_COUNT
        filepaths = []
        for class_idx in range(self.CLASSES_COUNT):
            class_filepaths = []
            for instance_idx in range(instances_count):
                class_filepaths.append(filpaths_sorted[self.CLASSES_COUNT*instance_idx + class_idx])
            filepaths.append(class_filepaths)

        # iterate over batches
        for _ in range(int(len(pointclouds_filepaths) / (instances_number * self.CLASSES_COUNT))):
            batch_clouds = []
            batch_labels = []
            for instance_idx in range(instances_number):
                for class_idx in range(self.CLASSES_COUNT):
                    filepath = random.choice(filepaths[class_idx])
                    filepaths[class_idx].remove(filepath)
                    cloud_class = int(filepath[filepath.rfind("_") + 1 : filepath.rfind(".")])
                    batch_clouds.append(np.load(filepath))
                    batch_labels.append(cloud_class)
            # stack
            batch_clouds = np.stack(batch_clouds, axis=0)
            batch_clouds = np.stack([batch_clouds], axis=1)
            batch_labels = np.stack(batch_labels, axis=0)

            # shuffle points
            if shuffle_points:
                batch_clouds = SyntheticData._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = SyntheticData._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = SyntheticData._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = SyntheticData._rotate_batch(batch_clouds, random_axis=False)
            # reshape
            if "flatten_pointclouds" in reshape_flags:
                shape = batch_clouds.shape
                batch_clouds = np.reshape(batch_clouds, [shape[0], shape[1], -1])

            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_random_batch(self, train, batch_size=64, shuffle_files=False,
                              shuffle_points=False, jitter_points=False,
                              rotate_pointclouds=False, rotate_pointclouds_up=False,
                              reshape_flags=[]):
        """ 
        Take random pointcloud, apply optional operations on each pointclouds and return 
        batch of such pointclouds with labels.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            batch_size (int): Batch size we would split the dataset into (numer of triplets
                to be returned by the generator).
            shuffle_files (boolean): Should we shuffle files with pointclouds?
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, N*3]
        Returns:
            (np.ndarray of shape [B, N, 3]), where B: batch size, N: number of points in each
                pointcloud, 3: number of dimensions of each pointcloud (x,y,z).
        """
        # Train or test dir?
        if train:
            pointclouds_dir = self.train_dir_path
        else:
            pointclouds_dir = self.test_dir_path

        # Get filepaths
        pointclouds_filepaths = [os.path.join(pointclouds_dir, filename) for filename
                                 in os.listdir(pointclouds_dir) if filename.endswith(".npy")]

        # shuffle files?
        if shuffle_files:
            np.random.shuffle(pointclouds_filepaths)

        # iterate over all files
        for batch_idx in range(int(len(pointclouds_filepaths) / batch_size)):
            # batch data
            batch_clouds = []
            batch_labels = []
            
            # load pointclouds
            for cloud_idx in range(batch_size):
                # anchor & positive
                global_idx = batch_idx * batch_size + cloud_idx
                batch_labels.append(int(pointclouds_filepaths[global_idx][pointclouds_filepaths[global_idx].rfind("_") + 1 :
                                                                          pointclouds_filepaths[global_idx].rfind(".")]))
                batch_clouds.append(np.load(pointclouds_filepaths[global_idx]))
            # stack
            batch_clouds = np.stack(batch_clouds, axis=0)
            batch_clouds = np.stack([batch_clouds], axis=1)
            batch_labels = np.stack(batch_labels, axis=0)

            # shuffle points
            if shuffle_points:
                batch_clouds = SyntheticData._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = SyntheticData._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = SyntheticData._rotate_batch(batch_clouds)
            if rotate_pointclouds_up:
                batch_clouds = SyntheticData._rotate_batch(batch_clouds, random_axis=False)
            # reshape
            if "flatten_pointclouds" in reshape_flags:
                shape = batch_clouds.shape
                batch_clouds = np.reshape(batch_clouds, [shape[0], shape[1], -1])

            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels
