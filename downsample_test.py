import pypcd
import numpy as np
import pointcloud_downsample
import deepclouds.modelnet_data as modelnet_data

# filepaths
pcd_filepath_in = 'data/pcd/modelnet_orig.pcd'
pcd_filepath_out = 'data/pcd/modelnet_unif.pcd'
pcd_filepath_out_rand = 'data/pcd/modelnet_rand.pcd'

data_gen = modelnet_data.ModelnetData(2048)
for X, Y in data_gen.generate_random_batch(True):

  # Downsample uniform
  idx = 2
  output_pointcloud_raw = pointcloud_downsample.uniform(X[idx])
  print "NEW SHAPE [UNIFORM]: ", output_pointcloud_raw.shape

  # Save as pcd (original)
  cloud_in = pypcd.make_xyz_point_cloud(X[idx])
  cloud_in.save_pcd(pcd_filepath_in)

  # Save as pcd (uniform)
  cloud_out = pypcd.make_xyz_point_cloud(output_pointcloud_raw)
  cloud_out.save_pcd(pcd_filepath_out)

  # Save as pcd (random)
  cloud_point_idxs = np.arange(len(X[idx]))
  cloud_randm_idxs = np.random.choice(cloud_point_idxs, 128, replace=False)
  cloud_out_rand = X[idx][cloud_randm_idxs]
  cloud_out_rand = pypcd.make_xyz_point_cloud(cloud_out_rand)
  cloud_out_rand.save_pcd(pcd_filepath_out_rand)  

  break


