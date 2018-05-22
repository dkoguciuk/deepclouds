#include "pointcloud_downsample_lib.h"
#include <cstdlib>
#include <vector>

int main (int argc, char** argv)
{
    // Load
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("model.pcd", *input_cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    // Fill
    std::vector<double> input_cloud_raw;
    for (int i = 0; i<input_cloud->points.size(); ++i)
    {
        input_cloud_raw.push_back(input_cloud->points[i].x);
        input_cloud_raw.push_back(input_cloud->points[i].y);
        input_cloud_raw.push_back(input_cloud->points[i].z);
    }

    // Downsample
    std::vector<double> output_cloud_raw = downsample_uniform(input_cloud_raw, 128, 0.02);

    // Prepare
    output_cloud->width    = output_cloud_raw.size();
    output_cloud->height   = 1;
    output_cloud->is_dense = false;
    output_cloud->points.resize (output_cloud->width * output_cloud->height);

    // Fill
    for (int i = 0; i < output_cloud->points.size()/3; ++i)
    {
        output_cloud->points[i].x = output_cloud_raw[i*3 + 0];
        output_cloud->points[i].y = output_cloud_raw[i*3 + 1];
        output_cloud->points[i].z = output_cloud_raw[i*3 + 2];
    }

    // Save
    pcl::io::savePCDFileASCII("model_downsampled.pcd", *output_cloud);
}
