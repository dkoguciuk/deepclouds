#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <ctime>
#include <string>
#include <iostream>
#include <vector>

const int SIZE = 128;
std::vector<double> downsample_uniform(std::vector<double> input_cloud_raw, int output_cloud_size = 128,
                                       float leaf_size_start = 0.25, float leaf_size_step = 0.01);
