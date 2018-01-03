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
	while i+batch_size-1 < data.shape[0]:
		j = 0
		A = np.empty([batch_size, data.shape[1], data.shape[2]])
		P = np.empty([batch_size, data.shape[1], data.shape[2]])
		N = np.empty([batch_size, data.shape[1], data.shape[2]])
		while j < batch_size:
			k = find_point_cloud_from_another_class(data, labels, i)
			A[j]=shuffle_point_cloud(data[i])
			P[j]=shuffle_point_cloud(data[i])
			N[j]=shuffle_point_cloud(k)
			i = i+1
			j = j+1
		yield A, P, N
		
		
def shuffle_point_cloud(point_cloud):
	np.random.shuffle(point_cloud)
	return np.vstack(point_cloud)


def find_point_cloud_from_another_class(data, labels, index):
	i = 0
	while True:
		j = randint(0, data.shape[0]-1)
		if labels[j][0] != labels[index][0]:
			return np.vstack(data[j])
