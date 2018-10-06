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
import pointcloud_downsample
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

    def __init__(self, pointcloud_size, clusterize=False):
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
        # merge training data
        train_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "train_files.txt"))]
        train_data = [ModelnetData._load_h5_file(train_file) for train_file in train_files]
        pointclouds_train, labels_train = zip(*train_data)
        self.pointclouds_train = np.concatenate(pointclouds_train, axis=0)
        self.labels_train = np.concatenate(labels_train, axis=0)
        # merge test data
        test_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelnetData._get_filenames(os.path.join(df.DATA_MODELNET_DIR, "test_files.txt"))]
        test_data = [ModelnetData._load_h5_file(test_file) for test_file in test_files]
        pointclouds_test, labels_test = zip(*test_data)
        self.pointclouds_test = np.concatenate(pointclouds_test, axis=0)
        self.labels_test = np.concatenate(labels_test, axis=0)
        # subclasses
        if clusterize:
            subclasses_files = [f for f in os.listdir(df.DATA_MODELNET_DIR) if 'npy' in f]
            subclasses_classes = np.unique([f.split('_')[1] for f in subclasses_files])
            for class_idx in subclasses_classes:
                # Get nonclass clouds from trainset
                nonclass_idx = np.where(self.labels_train != int(class_idx))[0]
                nonclass_lbls = self.labels_train[nonclass_idx]
                nonclass_clds = self.pointclouds_train[nonclass_idx]
                # Get class clouds from trainset
                class_idcs = np.where(self.labels_train == int(class_idx))[0]
                class_lbls = self.labels_train[class_idcs]
                class_clds = self.pointclouds_train[class_idcs]
                # Get subclass clusters
                new_clds = []
                new_lbls = []
                subclasses_clusters = [f.split('_')[3][0] for f in subclasses_files if f.split('_')[1] == class_idx]
                for cluster in np.sort(subclasses_clusters):
                    # Load cluster indices
                    cluster_indices = np.load(os.path.join(df.DATA_MODELNET_DIR, 'cloud_' + class_idx + '_cluster_' + cluster + '.npy'))
                    # Split
                    new_cld = class_clds[cluster_indices]
                    new_lbl = np.ones((cluster_indices.shape[0],1), dtype=np.int32) * (int(class_idx)*100 + int(cluster))
                    new_clds.append(new_cld)
                    new_lbls.append(new_lbl)
    
                # Concatenate
                new_clds = np.concatenate(new_clds)
                new_lbls = np.concatenate(new_lbls)
                self.pointclouds_train = np.concatenate((nonclass_clds, new_clds))
                self.labels_train = np.concatenate((nonclass_lbls, new_lbls))
            # Align label numering
            labels_actual = np.unique(self.labels_train)
            labels_aligned = np.arange(len(labels_actual))
            labels_missing = np.setdiff1d(labels_aligned, labels_actual, assume_unique=True)
            labels_additional = np.setdiff1d(labels_actual, labels_aligned, assume_unique=True)
            for label_missing, label_additional in zip(labels_missing, labels_additional):
                self.labels_train[self.labels_train == label_additional] = label_missing
            self.CLASSES_COUNT = len(np.unique(self.labels_train))
        else:
            self.CLASSES_COUNT = 40
        # internal vars
        self.pointcloud_size = pointcloud_size
        with open(df.CLASS_NAMES_FILE) as f:
            self.class_names = f.readlines()
        self.class_names = [class_name.strip() for class_name in self.class_names]
        self.class_count_max = np.max([self.generate_class_clouds(True, idx)[1].shape[0] for idx in range(self.CLASSES_COUNT)])

    def generate_class_clouds(self,
                              train,
                              class_idx,
                              sampled=False,
                              shuffle_points=False,
                              jitter_points=False,
                              rotate_pointclouds=False,
                              rotate_pointclouds_up=False,
                              sampling_method='fps'):
        """
        Take all pointclouds with class_idx.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            sampling_method (str): How should I sample the modelnet cloud? Supported formats, are:
                random, uniform, via_graphs.
        Returns:
            (np.ndarray of size [B, N, 3], np.ndarray of size [B, N]): Representative batch and it's labels.
        """
        # Train or test data?
        if train:
            ptclds = self.pointclouds_train
            labels = self.labels_train
        else:
            ptclds = self.pointclouds_test
            labels = self.labels_test
 
        # INSTANCES = 40
        class_idcs = np.where(labels == class_idx)[0]
        class_lbls = labels[class_idcs]
        class_clds = ptclds[class_idcs]

        # RESIZE
        if sampled:
            class_clds = np.expand_dims(class_clds, 1)
            if self.pointcloud_size < 2048:
                class_clouds_resized = []
                for cloud_idx in range(len(class_clds)):                        
                    if sampling_method == 'random':
                        cloud_point_idxs = np.arange(len(class_clds[cloud_idx][0]))
                        cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        class_clouds_resized.append(class_clds[cloud_idx][0][cloud_randm_idxs])
                    elif sampling_method == 'uniform':
                        class_clouds_resized.append(pointcloud_downsample.uniform(class_clds[cloud_idx][0]))
                    elif sampling_method == 'via_graphs':
                        class_clouds_resized.append(pointcloud_downsample.via_graphs(class_clds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        class_clouds_resized.append(class_clds[cloud_idx][0][:self.pointcloud_size])
                class_clds = np.stack(class_clouds_resized, axis=0)

        # shuffle points
        class_clds = np.expand_dims(class_clds, axis=1)
        if shuffle_points:
            class_clds = ModelnetData._shuffle_points_in_batch(class_clds)
        # jitter
        if jitter_points:
            class_clds = ModelnetData._jitter_batch(class_clds)
        # rotate
        if rotate_pointclouds:
            class_clds = ModelnetData._rotate_batch(class_clds)
        elif rotate_pointclouds_up:
            class_clds = ModelnetData._rotate_batch(class_clds, random_axis=False)
        class_clds = np.squeeze(class_clds, axis=1)

        # yield
        return class_clds, class_lbls

    def generate_representative_batch_for_train(self,
                                                instances_number=2,
                                                shuffle_clouds=False,
                                                shuffle_points=False,
                                                jitter_points=False,
                                                rotate_pointclouds=False,
                                                rotate_pointclouds_up=False,
                                                sampling_method='fps'):
        """
        Take pointclouds with at least instances_number of each class.
    
        Args:
            instances_number (int): Size of a batch expressed in instances_number of each CLASSES_COUNT,
                resulting in batch_size equals instances_number * CLASSES_COUNT. Please assume every
                object should appear at least two times in the batch, so recommended batches_number is 2.
            shuffle_clouds (boolean): Should we shuffle clouds order?
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_points (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            sampling_method (str): How should I sample the modelnet cloud? Supported formats, are:
                random, uniform, via_graphs.
        Returns:
            (np.ndarray of size [B, N, 3], np.ndarray of size [B, N]): Representative batch and it's labels.
        """
        ptclds = self.pointclouds_train
        labels = self.labels_train
        
        # EXTEND TO 889 for each class (max class count)
        ptclds_ext = []
        labels_ext = []
        for class_idx in range(self.CLASSES_COUNT):
            class_clouds, class_labels = self.generate_class_clouds(train=True, class_idx=class_idx)
            if len(class_labels) < self.class_count_max:
                how_many = self.class_count_max - len(class_labels)
                cloud_idcs = np.random.choice(len(class_labels), how_many, replace=True)
                dupa = np.copy(class_clouds[cloud_idcs, ])
                ptclds_ext.append(dupa)
                labels_ext.append(np.ones((len(cloud_idcs), 1), dtype=np.int32)*class_idx)
        ptclds_ext = np.concatenate(ptclds_ext)
        labels_ext = np.concatenate(labels_ext)
        ptclds = np.concatenate((ptclds, ptclds_ext))
        labels = np.concatenate((labels, labels_ext))

        # shuffle point clouds order
        if shuffle_clouds:
            ptclds, labels, _ = ModelnetData._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)
        # get sorted pointclouds along labels
        ptclds_sorted = { k : [] for k in range(self.CLASSES_COUNT)}
        for cloud_idx in range(labels.shape[0]):
            ptclds_sorted[labels[cloud_idx]].append(ptclds[cloud_idx])
        # iterate over all batches in this file
        for batch_idx in range(int(ptclds.shape[0] / self.CLASSES_COUNT / instances_number)):

            batch_clouds = []
            batch_labels = []
            for class_idx in range(self.CLASSES_COUNT):
                batch_clouds.append(ptclds_sorted[class_idx][batch_idx*instances_number:(batch_idx+1)*instances_number])
                batch_labels.append(np.ones(instances_number, dtype=np.int32) * class_idx)

            # stack
            batch_clouds = np.concatenate(batch_clouds, axis=0) # B, N, 3
            batch_clouds = np.expand_dims(batch_clouds, axis=1) # B, 1, N, 3
            batch_labels = np.concatenate(batch_labels)         # B

            # RESIZE
            if self.pointcloud_size < 2048:
                batch_clouds_resized = []
                for cloud_idx in range(len(batch_clouds)):                        
                    if sampling_method == 'random':
                        cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    elif sampling_method == 'uniform':
                        batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'via_graphs':
                        batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)           # B, N', 3
                batch_clouds = np.expand_dims(batch_clouds_resized, axis=1)     # B, 1, N', 3

            # shuffle points
            if shuffle_points:
                batch_clouds = ModelnetData._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelnetData._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds, random_axis=False)

            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_representative_batch(self,
                                      train,
                                      instances_number=2,
                                      shuffle_clouds=False,
                                      shuffle_points=False,
                                      jitter_points=False,
                                      rotate_pointclouds=False,
                                      rotate_pointclouds_up=False,
                                      sampling_method='fps'):
        """
        Take pointclouds with at least instances_number of each class.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            instances_number (int): Size of a batch expressed in instances_number of each CLASSES_COUNT,
                resulting in batch_size equals instances_number * CLASSES_COUNT. Please assume every
                object should appear at least two times in the batch, so recommended batches_number is 2.
            shuffle_clouds (boolean): Should we shuffle clouds order?
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_points (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            sampling_method (str): How should I sample the modelnet cloud? Supported formats, are:
                random, uniform, via_graphs.
        Returns:
            (np.ndarray of size [B, N, 3], np.ndarray of size [B, N]): Representative batch and it's labels.
        """
        # Train or test data?
        if train:
            ptclds = self.pointclouds_train
            labels = self.labels_train
        else:
            ptclds = self.pointclouds_test
            labels = self.labels_test
 
        # shuffle point clouds order
        if shuffle_clouds:
            ptclds, labels, _ = ModelnetData._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)
        # get sorted pointclouds along labels
        ptclds_sorted = { k : [] for k in range(self.CLASSES_COUNT)}
        for cloud_idx in range(labels.shape[0]):
            ptclds_sorted[labels[cloud_idx]].append(ptclds[cloud_idx])        
        # iterate over all batches in this file
        for _ in range(int(math.floor(ptclds.shape[0] / self.CLASSES_COUNT / instances_number))):

            batch_clouds = []
            batch_labels = []
            for instance_idx in range(instances_number):
                for class_idx in range(self.CLASSES_COUNT):
                    cloud_class_idx = np.random.choice(np.arange(len(ptclds_sorted[class_idx])), 1)[0]
                    batch_clouds.append(ptclds_sorted[class_idx][cloud_class_idx])
                    batch_labels.append(class_idx)

            # stack
            batch_clouds = np.stack(batch_clouds, axis=0)
            batch_clouds = np.stack([batch_clouds], axis=1)
            batch_labels = np.stack(batch_labels, axis=0)

            # RESIZE
            if self.pointcloud_size < 2048:
                batch_clouds_resized = []
                for cloud_idx in range(len(batch_clouds)):                        
                    if sampling_method == 'random':
                        cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    elif sampling_method == 'uniform':
                        batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'via_graphs':
                        batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)
                batch_clouds = np.stack([batch_clouds_resized], axis=1)

            # shuffle points
            if shuffle_points:
                batch_clouds = ModelnetData._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelnetData._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds, random_axis=False)

            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_random_batch(self, train, batch_size=64,
                              shuffle_clouds=False,
                              shuffle_points=False,
                              jitter_points=False,
                              rotate_pointclouds=False,
                              rotate_pointclouds_up=False,
                              sampling_method='fps'):
        """ 
        Take random pointcloud, apply optional operations on each pointclouds and return 
        batch of such pointclouds with labels.
    
        Args:
            train (bool): Should we take pointclouds from train or test dataset?
            batch_size (int): Batch size we would split the dataset into (numer of triplets
                to be returned by the generator).
            shuffle_clouds (boolean): Should we shuffle clouds order?
            shuffle_points (boolean): Should we shuffle points in the pointclouds?
            jitter_pointclouds (boolean): Randomly jitter points with gaussian noise.
            rotate_pointclouds (boolean): Rotate pointclouds with random angle around
                random axis, but this axis has to contain (0,0) point.
            sampling_method (str): How should I sample the modelnet cloud? Supported formats, are:
                random, uniform, via_graphs.
        Returns:
            (np.ndarray of shape [B, N, 3]), where B: batch size, N: number of points in each
                pointcloud, 3: number of dimensions of each pointcloud (x,y,z).
        """
        # Train or test data?
        if train:
            ptclds = self.pointclouds_train
            labels = self.labels_train
        else:
            ptclds = self.pointclouds_test
            labels = self.labels_test

        # shuffle point clouds order
        if shuffle_clouds:
            ptclds, labels, _ = ModelnetData._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)
        # iterate over all batches in this file
        for batch_idx in range(int(math.floor(ptclds.shape[0] / batch_size))):
                
            # Get pointclouds and labels
            batch_clouds = ptclds[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_clouds = np.stack([batch_clouds], axis=1)
            batch_labels = labels[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_labels = np.squeeze(batch_labels)

            # RESIZE
            if self.pointcloud_size < 2048:
                batch_clouds_resized = []
                for cloud_idx in range(len(batch_clouds)):
                    if sampling_method == 'random':
                        cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    elif sampling_method == 'uniform':
                        batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'via_graphs':
                        batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)
                batch_clouds = np.stack([batch_clouds], axis=1)

            # shuffle points
            if shuffle_points:
                batch_clouds = ModelnetData._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelnetData._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelnetData._rotate_batch(batch_clouds, random_axis=False)
    
            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_random_batch_multiple_clouds(self, train, batch_size=64, shuffle_files=False,
                                          shuffle_points=False, jitter_points=False,
                                          rotate_pointclouds=False, rotate_pointclouds_up=False,
                                          sampling_method='random', samples = 1, reshape_flags=[]):
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
            samples (int): How many pointclouds should I sample from the original cloud?
            sampling_method (str): How should I sample the modelnet cloud? Supported formats, are:
                random, uniform, via_graphs.
            reshape_flags (list of str): Output pointclouds are in the default shape of
                [B, N, 3]. One can specify some reshape flags here:
                flatten_pointclouds -- Flat Nx3 pointcloud to N*3 array, so the output size
                    would be [B, N*3]
        Returns:
            (np.ndarray of shape [B, X, N, 3]), where B: batch size, X: number of pointclouds
                sampled from one original point cloud, N: number of points in each pointcloud,
                3: number of dimensions of each pointcloud (x,y,z).
        """

        raise NotImplementedError("This routine is depricated!")

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

                # SAMPLING
                if self.pointcloud_size < 2048:
                    batch_clouds_resized = []
                    for cloud_idx in range(len(batch_clouds)):
                        batch_clouds_resized_instance = []
                        for _ in range(samples):
                            if sampling_method == 'random':
                                cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                                cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                                batch_clouds_resized_instance.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                            elif sampling_method == 'uniform':
                                batch_clouds_resized_instance.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                            elif sampling_method == 'via_graphs':
                                batch_clouds_resized_instance.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                            elif sampling_method == 'fps':
                                batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                        batch_clouds_resized.append(batch_clouds_resized_instance)
                    batch_clouds = np.stack(batch_clouds_resized, axis=0)
#                    if self.pointcloud_size < 2048:
#                        batch_clouds_resized = []
#                        for cloud_idx in range(len(batch_clouds)):
#                            cloud_permuted = batch_clouds[cloud_idx][0].copy()
#                            np.random.shuffle(cloud_permuted)
#                            #batch_clouds_resized.append(np.split(cloud_permuted, 16))
#                            batch_clouds_resized.append(np.split(cloud_permuted, 16)[:samples])
#                        batch_clouds = np.stack(batch_clouds_resized, axis=0)

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
                yield batch_clouds, batch_labels

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

    def __init__(self, pointcloud_size, permuted=True, rotated_up=True, rotated_rand=False):
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
            self.regenerate_files(permuted=permuted, rotated_up=rotated_up, rotated_rand=rotated_rand)

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

    def regenerate_files(self, instances_per_class_train=100, instances_per_class_test=10,
                         permuted=True, rotated_up=True, rotated_rand=False):
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
            if rotated_up:
                clouds_new = self._rotate_batch(clouds_new, random_axis=False)              # rotate along random axis and random angle
            if rotated_rand:
                clouds_new = self._rotate_batch(clouds_new, random_axis=True)               # rotate along random axis and random angle
            clouds_new = self._jitter_batch(clouds_new)                 # jitter points
            if permuted:
                clouds_new = self._shuffle_points_in_batch(clouds_new)      # shuffle point in the pointcloud
            for cloud_idx in range(self.CLASSES_COUNT):                 # save pointclouds
                global_idx = instance_idx * self.CLASSES_COUNT + cloud_idx
                cloud_path = os.path.join(self.train_dir_path, format(global_idx, '04d') + '_' + format(cloud_idx, '02d') + '.npy')
                np.save(cloud_path, clouds_new[cloud_idx][0])

        # Augment dataset and save (test)
        for instance_idx in range(0, instances_per_class_test):
            clouds_new = np.copy(clouds)
            if rotated_up:
                clouds_new = self._rotate_batch(clouds_new, random_axis=False)              # rotate along random axis and random angle
            if rotated_rand:
                clouds_new = self._rotate_batch(clouds_new, random_axis=True)               # rotate along random axis and random angle
            clouds_new = self._jitter_batch(clouds_new)                 # jitter points
            if permuted:
                clouds_new = self._shuffle_points_in_batch(clouds_new)      # shuffle point in the pointcloud
            for cloud_idx in range(self.CLASSES_COUNT):                 # save pointclouds
                global_idx = instance_idx * self.CLASSES_COUNT + cloud_idx
                cloud_path = os.path.join(self.test_dir_path, format(global_idx, '04d') + '_' + format(cloud_idx, '02d') + '.npy')
                np.save(cloud_path, clouds_new[cloud_idx][0])

    def generate_representative_batch(self,
                                      train,
                                      instances_number=2,
                                      shuffle_points=False,
                                      jitter_points=False,
                                      rotate_pointclouds=False,
                                      rotate_pointclouds_up=False,
                                      sampling_method=None,
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
                              sampling_method=None,
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
