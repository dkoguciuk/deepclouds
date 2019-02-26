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
import numpy as np

# Exceptions
if sys.version_info[0] < 3:
    import exceptions as excpt
else: 
    import builtins as excpt
    
import numpy as np
#import pointcloud_downsample
import numpy.core.umath_tests as nm
import deepclouds.defines as df

class GenericDataset(object):
    """
    Generic (base) class for different input data manipulation. 
    """
    
    # Variables to use in the derived class
    pointclouds_train = None
    pointclouds_devel = None
    pointclouds_test = None
    labels_train = None
    labels_devel = None
    labels_test = None
    classes_count = None
    pointcloud_size = None
    class_names = None
    
    def __init__(self):
        """
        Default constructor, where os is checked.
        """
        # check platform
        if sys.platform != "linux" and sys.platform != "linux2":
            raise excpt.OSError("Your OS is not supported, please switch to linux")

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
            raise excpt.ValueError("Clip should be a positive number")
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
                batch[cloud_idx] = GenericDataset._rotate_pointcloud(batch[cloud_idx])
            else:
                batch[cloud_idx] = GenericDataset._rotate_pointcloud_up(batch[cloud_idx])
        return np.reshape(batch, shape)

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
            ptclds, labels, _ = GenericDataset._shuffle_data_with_labels(ptclds, labels)
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
                    #elif sampling_method == 'uniform':
                    #    batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    #elif sampling_method == 'via_graphs':
                    #    batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)
                batch_clouds = np.stack([batch_clouds], axis=1)

            # shuffle points
            if shuffle_points:
                batch_clouds = GenericDataset._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = GenericDataset._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = GenericDataset._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = GenericDataset._rotate_batch(batch_clouds, random_axis=False)
    
            # yield
            yield np.squeeze(batch_clouds, axis=1), batch_labels

    def generate_batch_c_i(self, classes_no, instances_no,
                           shuffle_clouds=False, shuffle_points=False,
                           jitter_points=False, rotate_pointclouds=False,
                           rotate_pointclouds_up=False, sampling_method='fps'):
        
        #######################################################################
        # Get train data
        #######################################################################
        ptclds = self.pointclouds_train
        labels = self.labels_train
        
        #######################################################################
        # shuffle point clouds order
        #######################################################################
        if shuffle_clouds:
            ptclds, labels, _ = ModelNet40._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)

        #######################################################################
        # Get classes_to_iterate
        #######################################################################
        unique_classes = np.unique(labels)
        classes = np.random.choice(unique_classes, size=(len(unique_classes)//classes_no) * classes_no, replace=False)
        
        #######################################################################
        # Get instances of each class
        #######################################################################
        def sample_k_point_clouds(class_idx, point_clouds_data, point_clouds_labels, instances_no):
            # Get point clouds
            point_clouds = point_clouds_data[point_clouds_labels == class_idx]
            if instances_no > len(point_clouds):
                print ("You want to much samples and I was to lazy to handle this.")
                exit(-1)
            # Get samples
            point_clouds_indices = np.random.choice(len(point_clouds), size=instances_no, replace=False)
            point_clouds = point_clouds[point_clouds_indices]
            return point_clouds

        instances = np.array([sample_k_point_clouds(x, ptclds, labels, instances_no) for x in classes])
        
        #######################################################################
        # Reshape data
        #######################################################################
        classes = np.reshape(classes, (-1, classes_no))                                 # L, C
        instances = np.stack(np.split(instances, len(classes)))                         # L, C, I, N, 3
        
        #######################################################################
        # iterate over all batches in this file
        #######################################################################
        for batch_idx in range(len(classes)):
            
            ###################################################################
            # Get batch clouds and labels
            ###################################################################
            batch_clouds = instances[batch_idx]                                         # C, I, N, 3
            batch_labels = np.tile(classes[batch_idx].reshape(-1, 1), instances_no)     # C, I
            batch_clouds = np.expand_dims(np.reshape(batch_clouds, (-1, batch_clouds.shape[-2], batch_clouds.shape[-1])), axis=1)       # -1, N, 3
            
            ###################################################################
            # Resize pointclouds
            ###################################################################
            if self.pointcloud_size < 2048:
                batch_clouds_resized = []
                for cloud_idx in range(len(batch_clouds)):                        
                    if sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                    else:
                        print("Can\'t handle this at the moment, cos my developer is a lazy cunt.")
                        exit()
                batch_clouds = np.stack(batch_clouds_resized, axis=0)           # -1, N', 3
                batch_clouds = np.expand_dims(batch_clouds_resized, axis=1)     # -1, 1, N', 3
            else:
                print("Can\'t handle this at the moment, cos my developer is a lazy cunt.")
                exit()
            
            ###################################################################
            # Augment point clouds
            ###################################################################
            
            # shuffle points
            if shuffle_points:
                batch_clouds = ModelNet40._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelNet40._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds, random_axis=False)
            
            ###################################################################
            # Reshape and yield
            ###################################################################
            
            batch_clouds = np.reshape(batch_clouds, (classes_no, instances_no, batch_clouds.shape[-2], batch_clouds.shape[-1]))     # C, I, N, 3
            yield batch_clouds, batch_labels
            

class ModelNet40(GenericDataset) : 
    """
    Class implementing all needed functionality with ModelNet40 data manipulation.
    """

    def __init__(self, pointcloud_size, clusterize=False, dev_fraction=False):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        
        # super
        super(ModelNet40, self).__init__()

        # download data if it's needed
        if not os.path.exists(df.DATA_MODELNET40_DIR):
            zipfile = os.path.basename(df.DATA_MODELNET40_URL)
            os.system('wget %s; unzip %s' % (df.DATA_MODELNET40_URL, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], df.DATA_DIR))
            os.system('rm %s' % (zipfile))
            print("Modelnet database downloading completed...")
        
        #######################################################################
        # load train data
        #######################################################################
        
        train_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelNet40._get_filenames(os.path.join(df.DATA_MODELNET40_DIR, "train_files.txt"))]
        train_data = [ModelNet40._load_h5_file(train_file) for train_file in train_files]
        pointclouds_train, labels_train = zip(*train_data)
        self.pointclouds_train = np.concatenate(pointclouds_train, axis=0)
        self.labels_train = np.concatenate(labels_train, axis=0)
        
        #######################################################################
        # train/dev split
        # 
        # For each class shuffle the original clouds order and then split into
        # train and dev set with def_fraction arg.
        #######################################################################
        
        if dev_fraction > 0:
            self.classes_count = 40
            ptclds_trn = []
            labels_trn = []
            ptclds_dev = []
            labels_dev = []
            for class_no in range(self.classes_count):
                # Get class clouds
                class_indices = np.squeeze(np.argwhere(np.squeeze(self.labels_train) == class_no))
                class_clouds = self.pointclouds_train[class_indices]
                class_labels = self.labels_train[class_indices]
                # Split idx
                split_idx = int(len(class_clouds) * (1-dev_fraction))
                ptclds_trn.append(class_clouds[:split_idx])
                labels_trn.append(class_labels[:split_idx])
                ptclds_dev.append(class_clouds[split_idx:])
                labels_dev.append(class_labels[split_idx:])
            self.pointclouds_devel = np.concatenate(ptclds_dev)
            self.labels_devel = np.squeeze(np.concatenate(labels_dev))
            self.pointclouds_train = np.concatenate(ptclds_trn)
            self.labels_train = np.squeeze(np.concatenate(labels_trn))
        
        #######################################################################
        # load test data
        #######################################################################
        
        test_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelNet40._get_filenames(os.path.join(df.DATA_MODELNET40_DIR, "test_files.txt"))]
        test_data = [ModelNet40._load_h5_file(test_file) for test_file in test_files]
        pointclouds_test, labels_test = zip(*test_data)
        self.pointclouds_test = np.concatenate(pointclouds_test, axis=0)
        self.labels_test = np.concatenate(labels_test, axis=0)
        
        #######################################################################
        # clusterize (not userd anymore I guess)
        #######################################################################
        
        if clusterize:
            subclasses_files = [f for f in os.listdir(df.DATA_MODELNET40_DIR) if 'npy' in f]
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
            self.classes_count = len(np.unique(self.labels_train))
        else:
            self.classes_count = 40
            
        #######################################################################
        # Internal help variables
        #######################################################################
            
        self.pointcloud_size = pointcloud_size
        with open(df.CLASS_NAMES_FILE) as f:
            self.class_names = f.readlines()
        self.class_names = [class_name.strip() for class_name in self.class_names]
        self.class_count_max = np.max([self.generate_class_clouds(True, idx)[1].shape[0] for idx in range(self.classes_count)])

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
                    #elif sampling_method == 'uniform':
                    #    class_clouds_resized.append(pointcloud_downsample.uniform(class_clds[cloud_idx][0]))
                    #elif sampling_method == 'via_graphs':
                    #    class_clouds_resized.append(pointcloud_downsample.via_graphs(class_clds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        class_clouds_resized.append(class_clds[cloud_idx][0][:self.pointcloud_size])
                class_clds = np.stack(class_clouds_resized, axis=0)

        # shuffle points
        class_clds = np.expand_dims(class_clds, axis=1)
        if shuffle_points:
            class_clds = ModelNet40._shuffle_points_in_batch(class_clds)
        # jitter
        if jitter_points:
            class_clds = ModelNet40._jitter_batch(class_clds)
        # rotate
        if rotate_pointclouds:
            class_clds = ModelNet40._rotate_batch(class_clds)
        elif rotate_pointclouds_up:
            class_clds = ModelNet40._rotate_batch(class_clds, random_axis=False)
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
                labels_ext.append(np.ones((len(cloud_idcs), 1), dtype=np.int32) * class_idx)
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
                batch_clouds.append(ptclds_sorted[class_idx][batch_idx * instances_number:(batch_idx + 1) * instances_number])
                batch_labels.append(np.ones(instances_number, dtype=np.int32) * class_idx)

            # stack
            batch_clouds = np.concatenate(batch_clouds, axis=0)  # B, N, 3
            batch_clouds = np.expand_dims(batch_clouds, axis=1)  # B, 1, N, 3
            batch_labels = np.concatenate(batch_labels)  # B

            # RESIZE
            if self.pointcloud_size < 2048:
                batch_clouds_resized = []
                for cloud_idx in range(len(batch_clouds)):                        
                    if sampling_method == 'random':
                        cloud_point_idxs = np.arange(len(batch_clouds[cloud_idx][0]))
                        cloud_randm_idxs = np.random.choice(cloud_point_idxs, self.pointcloud_size, replace=False)
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][cloud_randm_idxs])
                    # elif sampling_method == 'uniform':
                    #    batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    # elif sampling_method == 'via_graphs':
                    #    batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)  # B, N', 3
                batch_clouds = np.expand_dims(batch_clouds_resized, axis=1)  # B, 1, N', 3

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

    def generate_representative_batch_for_train_or_devel(self, train_or_devel = 'train',
                                                instances_number=2,
                                                shuffle_clouds=False,
                                                shuffle_points=False,
                                                jitter_points=False,
                                                rotate_pointclouds=False,
                                                rotate_pointclouds_up=False,
                                                sampling_method='fps',
                                                augment=720):
        """
        Take pointclouds with at least instances_number of each class.
    
        Args:
            instances_number (int): Size of a batch expressed in instances_number of each classes_count,
                resulting in batch_size equals instances_number * classes_count. Please assume every
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
        
        #######################################################################
        # Copy the original clouds
        #######################################################################
        
        ptclds = None
        labels = None
        if train_or_devel == 'train':
            ptclds = np.copy(self.pointclouds_train)
            labels = np.copy(self.labels_train)
        elif train_or_devel == 'devel':
            ptclds = np.copy(self.pointclouds_devel)
            labels = np.copy(self.labels_devel)
        else:
            raise ValueError('train_or_dev should be either train or devel');

        #######################################################################
        # augment dataset that all classes will be equally numbered 
        # this operation is specified by @augment_trn and @augment_dev
        # params
        #######################################################################
        
        def augment_data(ptclds, labels, class_count):
            ptclds_ext = []
            labels_ext = []
            for class_no in range(self.classes_count):
                # Get class clouds
                class_indices = np.squeeze(np.argwhere(np.squeeze(labels) == class_no))
                class_clouds = ptclds[class_indices]
                class_labels = labels[class_indices]
                # Shuffle class clouds
                if shuffle_clouds:
                    np.random.shuffle(class_clouds)
                # If there are to many clouds
                if len(class_labels) > class_count:
                    class_clouds = class_clouds[:class_count]
                    class_labels = class_labels[:class_count]
                # If there are to less clouds
                elif len(class_labels) < class_count:
                    how_many = class_count - len(class_labels)
                    aug_idx = np.random.choice(len(class_labels), how_many, replace=True)
                    clouds_tmp = np.copy(class_clouds[aug_idx, ])
                    labels_tmp = np.copy(class_labels[aug_idx, ])
                    class_clouds = np.concatenate((class_clouds, clouds_tmp))
                    class_labels = np.concatenate((class_labels, labels_tmp))
                # Remember
                ptclds_ext.append(class_clouds)
                labels_ext.append(class_labels)
            ptclds_ext = np.concatenate(ptclds_ext)
            labels_ext = np.concatenate(labels_ext)
            return ptclds_ext, labels_ext

        ptclds, labels = augment_data(ptclds, labels, augment)

        #######################################################################
        # shuffle the pointclouds and labels along first axis
        #######################################################################

        # shuffle point clouds order
        if shuffle_clouds:
            ptclds, labels, _ = ModelNet40._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)
        
        #######################################################################
        # Put pointclouds into dict with class_no as a key
        #######################################################################
        
        ptclds_sorted = { k : [] for k in range(self.classes_count)}
        for cloud_idx in range(labels.shape[0]):
            ptclds_sorted[labels[cloud_idx]].append(ptclds[cloud_idx])
        
        #######################################################################
        # Generate 
        #######################################################################
        
        for batch_idx in range(int(ptclds.shape[0] / self.classes_count / instances_number)):
            batch_clouds = []
            batch_labels = []
            for class_idx in range(self.classes_count):
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
                    #elif sampling_method == 'uniform':
                    #    batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    #elif sampling_method == 'via_graphs':
                    #    batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)           # B, N', 3
                batch_clouds = np.expand_dims(batch_clouds_resized, axis=1)     # B, 1, N', 3

            # shuffle points
            if shuffle_points:
                batch_clouds = ModelNet40._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelNet40._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds, random_axis=False)

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
            instances_number (int): Size of a batch expressed in instances_number of each classes_count,
                resulting in batch_size equals instances_number * classes_count. Please assume every
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
            ptclds, labels, _ = ModelNet40._shuffle_data_with_labels(ptclds, labels)
        labels = np.squeeze(labels)
        # get sorted pointclouds along labels
        ptclds_sorted = { k : [] for k in range(self.classes_count)}
        for cloud_idx in range(labels.shape[0]):
            ptclds_sorted[labels[cloud_idx]].append(ptclds[cloud_idx])        
        # iterate over all batches in this file
        for _ in range(int(math.floor(ptclds.shape[0] / self.classes_count / instances_number))):

            batch_clouds = []
            batch_labels = []
            for instance_idx in range(instances_number):
                for class_idx in range(self.classes_count):
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
                    #elif sampling_method == 'uniform':
                    #    batch_clouds_resized.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                    #elif sampling_method == 'via_graphs':
                    #    batch_clouds_resized.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
                    elif sampling_method == 'fps':
                        batch_clouds_resized.append(batch_clouds[cloud_idx][0][:self.pointcloud_size])
                batch_clouds = np.stack(batch_clouds_resized, axis=0)
                batch_clouds = np.stack([batch_clouds_resized], axis=1)

            # shuffle points
            if shuffle_points:
                batch_clouds = ModelNet40._shuffle_points_in_batch(batch_clouds)
            # jitter
            if jitter_points:
                batch_clouds = ModelNet40._jitter_batch(batch_clouds)
            # rotate
            if rotate_pointclouds:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds)
            elif rotate_pointclouds_up:
                batch_clouds = ModelNet40._rotate_batch(batch_clouds, random_axis=False)

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
            file_pointclouds, file_labels = ModelNet40._load_h5_file(pointcloud_files[pointcloud_files_idxs[idx]])
            # shuffle pointclouds from file
            if shuffle_files:
                file_pointclouds, file_labels, _ = ModelNet40._shuffle_data_with_labels(file_pointclouds, file_labels)
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
                            #elif sampling_method == 'uniform':
                            #    batch_clouds_resized_instance.append(pointcloud_downsample.uniform(batch_clouds[cloud_idx][0]))
                            #elif sampling_method == 'via_graphs':
                            #    batch_clouds_resized_instance.append(pointcloud_downsample.via_graphs(batch_clouds[cloud_idx][0]))
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


class ModelNet40Downsampled(GenericDataset) : 
    """
    Class implementing all needed functionality with ModelNet40Downsampled data manipulation.
    """
    
    def __init__(self, pointcloud_size):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        
        # super
        super(ModelNet40Downsampled, self).__init__()

        # Is data present?
        if not os.path.exists(df.DATA_MODELNET40_SAMPLED_DIR):
            raise AssertionError('No ModelNet40_downsampled dataset found, please move your hdf5 files into: ' + df.DATA_MODELNET40_SAMPLED_DIR)
        
        #######################################################################
        # load train data
        #######################################################################
        
        f = h5py.File(os.path.join(df.DATA_MODELNET40_SAMPLED_DIR, 'train_' + str(pointcloud_size) + '.hdf5'), 'r')
        self.pointclouds_train = np.array(f['tr_cloud'])
        self.labels_train = np.array(f['tr_labels'])
        
        #######################################################################
        # load test data
        #######################################################################
        
        f = h5py.File(os.path.join(df.DATA_MODELNET40_SAMPLED_DIR, 'test_' + str(pointcloud_size) + '.hdf5'), 'r')
        self.pointclouds_test = np.array(f['test_cloud'])
        self.labels_test = np.array(f['test_labels'])
        
        #######################################################################
        # load test data
        #######################################################################
        
        self.classes_count = np.max(np.unique(self.labels_train)) + 1
        self.pointcloud_size = pointcloud_size
    
        #######################################################################
        # SOME CHECKS
        #######################################################################        
        
#         for point_clouds in [self.pointclouds_train, self.pointclouds_test]:       
#             for idx, cloud in enumerate(point_clouds):
#     
#                 # ball unit?
#                 norms = [np.linalg.norm(p) for p in cloud]
#                 if np.max(norms) > 1.5:
#                     print ("INDEX: ", idx, " is not in unit-ball!")
#                 
#                 # Zero mean?
#                 if max(abs(np.mean(cloud[:,0])), abs(np.mean(cloud[:,1])), abs(np.mean(cloud[:,2]))) > 0.3:
#                     print ("INDEX: ", idx, " not zero-mean!")


class ShapeNetCore55(GenericDataset) : 
    """
    Class implementing all needed functionality with ShapeNetCore55 data manipulation.
    """

    def __init__(self, pointcloud_size):
        """
        Default constructor, where the check for modelnet files is performed and if there
        is no needed files, we would download them directly form the stanford website.
        """
        
        # super
        super(ShapeNetCore55, self).__init__()

        # Is data present?
        if not os.path.exists(df.DATA_SHAPENET55_DIR):
            raise AssertionError('No ShapeNetCoreV2 dataset found in the following directory: ' + df.DATA_SHAPENET55_DIR)
        
        #######################################################################
        # load train data
        #######################################################################
        
        # Classes dir
        classes_dir = [os.path.join(df.DATA_SHAPENET55_DIR, el) for el in os.listdir(df.DATA_SHAPENET55_DIR) if os.path.isdir(os.path.join(df.DATA_SHAPENET55_DIR, el))]
        for class_dir in classes_dir:
            
            # Objects dir
            objects_dir = [os.path.join(class_dir, el) for el in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, el))]
            for object_dir in objects_dir:
                
                # Load object
                object_path = os.path.join(os.path.join(object_dir, 'models'), 'model_normalized.obj')
                
            
            print (objects_dir)
            exit()
        
        print (train_classes_dir, len(train_classes_dir))
        exit()
        
        train_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelNet40._get_filenames(os.path.join(df.DATA_MODELNET40_DIR, "train_files.txt"))]
        train_data = [ModelNet40._load_h5_file(train_file) for train_file in train_files]
        pointclouds_train, labels_train = zip(*train_data)
        self.pointclouds_train = np.concatenate(pointclouds_train, axis=0)
        self.labels_train = np.concatenate(labels_train, axis=0)
        
        #######################################################################
        # train/dev split
        # 
        # For each class shuffle the original clouds order and then split into
        # train and dev set with def_fraction arg.
        #######################################################################
        
        if dev_fraction > 0:
            self.classes_count = 40
            ptclds_trn = []
            labels_trn = []
            ptclds_dev = []
            labels_dev = []
            for class_no in range(self.classes_count):
                # Get class clouds
                class_indices = np.squeeze(np.argwhere(np.squeeze(self.labels_train) == class_no))
                class_clouds = self.pointclouds_train[class_indices]
                class_labels = self.labels_train[class_indices]
                # Split idx
                split_idx = int(len(class_clouds) * (1-dev_fraction))
                ptclds_trn.append(class_clouds[:split_idx])
                labels_trn.append(class_labels[:split_idx])
                ptclds_dev.append(class_clouds[split_idx:])
                labels_dev.append(class_labels[split_idx:])
            self.pointclouds_devel = np.concatenate(ptclds_dev)
            self.labels_devel = np.squeeze(np.concatenate(labels_dev))
            self.pointclouds_train = np.concatenate(ptclds_trn)
            self.labels_train = np.squeeze(np.concatenate(labels_trn))
        
        #######################################################################
        # load test data
        #######################################################################
        
        test_files = [os.path.join(df.ROOT_DIR, elem) for elem in ModelNet40._get_filenames(os.path.join(df.DATA_MODELNET40_DIR, "test_files.txt"))]
        test_data = [ModelNet40._load_h5_file(test_file) for test_file in test_files]
        pointclouds_test, labels_test = zip(*test_data)
        self.pointclouds_test = np.concatenate(pointclouds_test, axis=0)
        self.labels_test = np.concatenate(labels_test, axis=0)
        
        #######################################################################
        # clusterize (not userd anymore I guess)
        #######################################################################
        
        if clusterize:
            subclasses_files = [f for f in os.listdir(df.DATA_MODELNET40_DIR) if 'npy' in f]
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
            self.classes_count = len(np.unique(self.labels_train))
        else:
            self.classes_count = 40
            
        #######################################################################
        # Internal help variables
        #######################################################################
            
        self.pointcloud_size = pointcloud_size
        with open(df.CLASS_NAMES_FILE) as f:
            self.class_names = f.readlines()
        self.class_names = [class_name.strip() for class_name in self.class_names]
        self.class_count_max = np.max([self.generate_class_clouds(True, idx)[1].shape[0] for idx in range(self.classes_count)])
