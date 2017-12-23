import numpy as np
from random import randint

def generate_next_data(data, labels, batch_size):
	""" 
	Generator returns 3 point clouds A, P, N.

	Args:
		data (numpy.ndarray): point clouds data of size [number_of_pointclouds, number_of_points_in_cloud, numer_of_coord_of_one_point]
		labels (numpy.ndarray): labels of each pointcloud of size [number_of_pointclouds, 0]
		batch_size (int): numer of triples to be returned
	Returns:
		tuple(A, P, N): where:
			A - random permutation of next cloud
			P - another random permutation of the same cloud
			N - random permutation of cloud from another class
	"""
	i = 0
	while i < data.shape[0]:
		j = find_point_cloud_from_another_class(data, labels, i)
		yield shuffle_point_cloud(data[i]), shuffle_point_cloud(data[i]),shuffle_point_cloud(j)
		i = i+1

def shuffle_point_cloud(point_cloud):
	index = np.arange(point_cloud.shape[0])
	np.random.shuffle(index)
	i, arr = 0, []
	while i < point_cloud.shape[0]:
		arr.append(point_cloud[index[i]])
		i = i+1
	return np.vstack(arr)

def find_point_cloud_from_another_class(data, labels, index):
	i = 0
	while True:
		j = randint(0, data.shape[0])
		if labels[j][0] != labels[index][0]:
			return np.vstack(data[j])
	
