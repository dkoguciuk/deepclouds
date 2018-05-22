import pypcd
import numpy as np
import downsample_uniform

# filepaths
pcd_filepath_in = '/home/daniel/eclipse-workspace/siamese_pointnet/external/repo_asd/build/model.pcd'
pcd_filepath_out = '/home/daniel/eclipse-workspace/siamese_pointnet/external/repo_asd/build/model_downsampled_py.pcd'

# load
pc = pypcd.PointCloud.from_path(pcd_filepath_in)
ar = pc.pc_data

# conert to 'normal' numpy array
points = []
for point_idx in range(ar.shape[0]):
  point = []
  point.append(ar[point_idx][0])
  point.append(ar[point_idx][1])
  point.append(ar[point_idx][2])
  points.append(point)
input_pointcloud_raw = np.array(points)

# downsample
output_pointcloud_raw = downsample_uniform.downsample_uniform(input_pointcloud_raw)

# info
print "NEW SHAPE : ", output_pointcloud_raw.shape

# Save as pcd
cloud_out = pypcd.make_xyz_point_cloud(output_pointcloud_raw)
cloud_out.save_pcd(pcd_filepath_out)


