#include "pointcloud_downsample_lib.h"

#include <string>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

pybind11::array py_downsample_uniform(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> array,
                                   float output_cloud_size=128, float leaf_size_start=0.25, float leaf_size_step=0.01)
{
    // check input dimensions
    if (array.ndim() != 2 )//&& array.ndim() != 3)
        throw std::runtime_error("Input should be 2-D or batched 2-D (3-D) numpy array");
    if ( array.shape()[array.ndim()-1] != 3 )
        throw std::runtime_error("Input should be size of [..,3]");

    // allocate std::vector (to pass to the C++ function)
    std::vector<double> input(array.size());

    // copy py::array -> std::vector
    std::memcpy(input.data(),array.data(),array.size()*sizeof(double));

    // call downsample_uniform
    std::vector<double> result = downsample_uniform(input, output_cloud_size, leaf_size_start, leaf_size_step);

    // convert
    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { result.size()/3 , 3 };
    std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

    // return 2-D NumPy array
    return pybind11::array(pybind11::buffer_info(
                            result.data(),                           /* data as contiguous array  */
                            sizeof(double),                          /* size of one scalar        */
                            pybind11::format_descriptor<double>::format(), /* data type                 */
                            ndim,                                    /* number of dimensions      */
                            shape,                                   /* shape of the matrix       */
                            strides                                  /* strides for each axis     */
                            ));
}

pybind11::array py_downsample_via_graphs(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> array,
                                      float output_cloud_size=128, int neighbors_number=5)
{
    // check input dimensions
    if (array.ndim() != 2 )//&& array.ndim() != 3)
        throw std::runtime_error("Input should be 2-D or batched 2-D (3-D) NumPy array");
    if ( array.shape()[array.ndim()-1] != 3 )
        throw std::runtime_error("Input should be size of [..,3]");

    // allocate std::vector (to pass to the C++ function)
    std::vector<double> input(array.size());

    // copy py::array -> std::vector
    std::memcpy(input.data(),array.data(),array.size()*sizeof(double));

    // call downsample_uniform
    std::vector<double> result = downsample_via_graphs(input, output_cloud_size, neighbors_number);

    // convert
    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { result.size()/3 , 3 };
    std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

    // return 2-D NumPy array
    return pybind11::array(pybind11::buffer_info(
                            result.data(),                           /* data as contiguous array  */
                            sizeof(double),                          /* size of one scalar        */
                            pybind11::format_descriptor<double>::format(), /* data type                 */
                            ndim,                                    /* number of dimensions      */
                            shape,                                   /* shape of the matrix       */
                            strides                                  /* strides for each axis     */
                            ));
}

PYBIND11_MODULE(pointcloud_downsample, m)
{
    m.doc() = "downsampling module for deepcloud project";
    m.def("uniform", &py_downsample_uniform, "downsample uniformly using regular voxel grid",
          pybind11::arg("array"), pybind11::arg("output_cloud_size") = 128,
          pybind11::arg("leaf_size_start") = 0.25, pybind11::arg("leaf_size_step") = 0.01);
    m.def("via_graphs", &py_downsample_via_graphs, "downsample via graphs (check related papers)",
          pybind11::arg("array"), pybind11::arg("output_cloud_size") = 128,
          pybind11::arg("neighbors_number") = 5);
}
