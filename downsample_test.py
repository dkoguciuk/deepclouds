import pypcd
import numpy as np
import downsample_uniform
import deepclouds.modelnet_data as modelnet_data

# filepaths
pcd_filepath_in = '/home/daniel/eclipse-workspace/siamese_pointnet/modelnet_in.pcd'
pcd_filepath_out = '/home/daniel/eclipse-workspace/siamese_pointnet/modelnet_downsampled.pcd'
pcd_filepath_out_rand = '/home/daniel/eclipse-workspace/siamese_pointnet/modelnet_downsampled_rand.pcd'

data_gen = modelnet_data.ModelnetData(2048)
for X, Y in data_gen.generate_random_batch(True):
  print X.shape
  idx = 2
  output_pointcloud_raw = downsample_uniform.downsample_uniform(X[idx])
  print "NEW SHAPE : ", output_pointcloud_raw.shape

  # Save as pcd
  cloud_in = pypcd.make_xyz_point_cloud(X[idx])
  cloud_in.save_pcd(pcd_filepath_in)

  # Save as pcd
  cloud_out = pypcd.make_xyz_point_cloud(output_pointcloud_raw)
  cloud_out.save_pcd(pcd_filepath_out)

  cloud_out_rand = X[idx][:128]
  cloud_out_rand = pypcd.make_xyz_point_cloud(cloud_out_rand)
  cloud_out_rand.save_pcd(pcd_filepath_out_rand)  

  break


