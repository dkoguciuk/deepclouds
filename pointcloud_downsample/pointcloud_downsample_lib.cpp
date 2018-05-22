#include "pointcloud_downsample_lib.h"

std::vector<double> downsample_uniform(std::vector<double> input_cloud_raw, int output_cloud_size,
                                       float leaf_size_start, float leaf_size_step)
{
    // asdasd
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Prepare
    input_cloud->width    = input_cloud_raw.size();
    input_cloud->height   = 1;
    input_cloud->is_dense = false;
    input_cloud->points.resize (input_cloud->width * input_cloud->height);

    // Fill
    for (int i = 0; i < input_cloud->points.size()/3; ++i)
    {
        input_cloud->points[i].x = input_cloud_raw[i*3 + 0];
        input_cloud->points[i].y = input_cloud_raw[i*3 + 1];
        input_cloud->points[i].z = input_cloud_raw[i*3 + 2];
    }

    // Downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(input_cloud);

    // Which direction
    float leaf_size = leaf_size_start;
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*filtered_cloud);
    bool direction_up = true;
    if (filtered_cloud->points.size() < output_cloud_size) direction_up = false;

    // Find the point
    while(true)
    {
        // Make leaf step
        if (direction_up) leaf_size += leaf_size_step;
        else leaf_size -= leaf_size_step;

        // Filter
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*filtered_cloud);

        // Check cond
        if (direction_up && filtered_cloud->points.size() <= output_cloud_size) break;
        if (!direction_up && filtered_cloud->points.size() >= output_cloud_size) break;
    }

    //bigger or equal points no
    if (direction_up) leaf_size -= leaf_size_step;

    // Randomly erase points to meet criteria
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*filtered_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::RandomSample<pcl::PointXYZ> sample(true); // Extract removed indices
    sample.setInputCloud(filtered_cloud);
    sample.setSample(output_cloud_size);
    sample.filter(*output_cloud);

    // Convert to vector of points
    std::vector<double> output_cloud_raw;
    double dist_max = 0;
    for (int i = 0; i<output_cloud->points.size(); ++i)
    {
        output_cloud_raw.push_back(output_cloud->points[i].x);
        output_cloud_raw.push_back(output_cloud->points[i].y);
        output_cloud_raw.push_back(output_cloud->points[i].z);

        double dist = (output_cloud->points[i].x*output_cloud->points[i].x + output_cloud->points[i].y*output_cloud->points[i].y + output_cloud->points[i].z*output_cloud->points[i].z);
        dist = sqrt(dist);
        dist_max = std::max(dist, dist_max);
    }
    //std::cout << "MAX_DIST = " << dist_max << std::endl;

    // Ret
    return output_cloud_raw;
}
