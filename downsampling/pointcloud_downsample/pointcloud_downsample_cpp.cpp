#include "pointcloud_downsample_lib.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;

int main (int argc, char** argv)
{
    string raw_input_path;
    string raw_output_path;
    string method;
    int size;

    //===============================================================================
    //======================= PROGRAM OPTIONS SPECIFICATION =========================
    //===============================================================================

    po::options_description desc("Options");
    desc.add_options()
            ("help,h", "Print this help.")
            ("input,i", po::value<string>(&raw_input_path)->required(), "Input point cloud path")
            ("output,o", po::value<string>(&raw_output_path), "Output point cloud path")
            ("method,m", po::value<string>(&method)->default_value("random"), "Point cloud sampling method, one of:\n"
                                                                       "random\n"
                                                                       "uniform\n"
                                                                       "graph")
            ("size,s ", po::value<int>(&size)->default_value(128), "Output point cloud size (defaults: 128)");

    po::variables_map vm;
    try{
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if ( vm.count("help")  )
        {
            cout << endl;
            cout << "======================================================" << endl;
            cout << "================ AEOLUS TEST PROGRAM =================" << endl;
            cout << "======================================================" << endl;
            cout << endl << desc << endl;
            return 0;
        }
        po::notify(vm);
    }
    catch( po::error& e )
    {
        cerr << endl << "ERROR: " << e.what() << endl << endl;
        cerr << desc << endl;
        return -1;
    }

    //===============================================================================
    //=========================== PROGRAM OPTIONS PARSING ===========================
    //===============================================================================

    // Input image
    fs::path input_path(raw_input_path);
    if (!fs::exists(input_path))
    {
        cerr << endl << "ERROR: " << "input point cloud file does not exist" << endl << endl;
        cerr << desc << endl;
        return -1;
    }

    fs::path output_path;
    if (raw_output_path.empty())
    {
        fs::path input_dir = input_path.parent_path();
        string input_stem = input_path.stem().string();
        string output_filename = input_stem + "_out.pcd";
        output_path = input_dir / output_filename;
    } else output_path = fs::path(raw_output_path);

    //===============================================================================
    //========================= LOAD PC AND PREPARE TO FEED =========================
    //===============================================================================

    // Load
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (input_path.string(), *input_cloud) == -1)
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    } else std::cout << "Point cloud loaded: " << input_path.string() << std::endl;

    // Fill
    std::vector<double> input_cloud_raw;
    for (int i = 0; i<input_cloud->points.size(); ++i)
    {
        input_cloud_raw.push_back(input_cloud->points[i].x);
        input_cloud_raw.push_back(input_cloud->points[i].y);
        input_cloud_raw.push_back(input_cloud->points[i].z);
    }

    //===============================================================================
    //================================= DOWNSAMPLE ==================================
    //===============================================================================

    // Downsample
    std::vector<double> output_cloud_raw;
    if (method == "random") { std::cout << "NOT IMPLEMENTED YET" << std::endl; exit(0); }
    else if (method == "uniform") output_cloud_raw = downsample_uniform(input_cloud_raw, size, 0.02);
    else if (method == "graph") output_cloud_raw = downsample_via_graphs(input_cloud_raw, size);

    //===============================================================================
    //============================= PREPARE AND SAVE PC =============================
    //===============================================================================

    // Prepare
    output_cloud->width    = output_cloud_raw.size()/3;
    output_cloud->height   = 1;
    output_cloud->is_dense = false;
    output_cloud->points.resize (output_cloud->width * output_cloud->height);

    // Fill
    for (int i = 0; i < output_cloud->points.size(); ++i)
    {
        output_cloud->points[i].x = output_cloud_raw[i*3 + 0];
        output_cloud->points[i].y = output_cloud_raw[i*3 + 1];
        output_cloud->points[i].z = output_cloud_raw[i*3 + 2];
    }

    // Save
    pcl::io::savePCDFileASCII(output_path.string(), *output_cloud);
    std::cout << "Point cloud saved: " << output_path.string() << std::endl;
}
