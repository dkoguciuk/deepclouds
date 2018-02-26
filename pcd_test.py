import pypcd
import numpy as np

# filepaths
npy_filepath = '/home/daniel/eclipse-workspace/siamese_pointnet/data/synthetic/train/0001_01.npy'
pcd_filepath = '/home/daniel/eclipse-workspace/siamese_pointnet/data/synthetic/train/pcd/0000_01.pcd'

# load
a = np.load(npy_filepath)

# augment
dist = a[1:] - a[:-1]
new_points = []
for idx in range(dist.shape[0]):
  act_dist = dist[idx]
  for ins in range(1, 10):
    incr = act_dist * ins / 10
    new_points.append(a[idx] + incr)

# concat
b = np.array(new_points)
c = np.concatenate((a, b), axis=0)


# Save as pcd
cloud_out = pypcd.make_xyz_point_cloud(c)
cloud_out.save_pcd(pcd_filepath)


