#include "pointcloud_downsample_lib.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <iostream>


namespace numpy = pybind11;




std::vector<int> multiply(const std::vector<double>& input)
{
    std::vector<int> output(input.size());

    for ( size_t i = 0 ; i < input.size() ; ++i )
        output[i] = 10*static_cast<int>(input[i]);

    return output;
}

std::vector<double> length(const std::vector<double>& pos)
{
    size_t N = pos.size() / 2;

    std::vector<double> output(N*3);

    for ( size_t i = 0 ; i < N ; ++i ) {
        output[i*3+0] = pos[i*2+0];
        output[i*3+1] = pos[i*2+1];
        output[i*3+2] = std::pow(pos[i*2+0]*pos[i*2+1],.5);
    }

    return output;
}

// wrap C++ function with NumPy array IO
numpy::array_t<int> py_multiply(numpy::array_t<double, numpy::array::c_style | numpy::array::forcecast> array)
{
    // allocate std::vector (to pass to the C++ function)
    std::vector<double> array_vec(array.size());

    // copy py::array -> std::vector
    std::memcpy(array_vec.data(),array.data(),array.size()*sizeof(double));

    // call pure C++ function
    std::vector<int> result_vec = multiply(array_vec);

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result        = numpy::array_t<int>(array.size());
    auto result_buffer = result.request();
    int *result_ptr    = (int *) result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr,result_vec.data(),result_vec.size()*sizeof(int));

    return result;
}

numpy::array py_length(numpy::array_t<double, numpy::array::c_style | numpy::array::forcecast> array)
{
    // check input dimensions
    if ( array.ndim()     != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
    if ( array.shape()[1] != 2 )
        throw std::runtime_error("Input should have size [N,2]");

    // allocate std::vector (to pass to the C++ function)
    std::vector<double> pos(array.size());

    // copy py::array -> std::vector
    std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

    // call pure C++ function
    std::vector<double> result = length(pos);

    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { array.shape()[0] , 3 };
    std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

    // return 2-D NumPy array
    return numpy::array(numpy::buffer_info(
                            result.data(),                           /* data as contiguous array  */
                            sizeof(double),                          /* size of one scalar        */
                            numpy::format_descriptor<double>::format(), /* data type                 */
                            ndim,                                    /* number of dimensions      */
                            shape,                                   /* shape of the matrix       */
                            strides                                  /* strides for each axis     */
                            ));
}



numpy::array py_downsample_uniform(numpy::array_t<double, numpy::array::c_style | numpy::array::forcecast> array,
                                   float output_cloud_size=128, float leaf_size_start=0.25, float leaf_size_step=0.01)
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
    std::vector<double> result = downsample_uniform(input, output_cloud_size, leaf_size_start, leaf_size_step);

    // convert
    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { result.size()/3 , 3 };
    std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

    // return 2-D NumPy array
    return numpy::array(numpy::buffer_info(
                            result.data(),                           /* data as contiguous array  */
                            sizeof(double),                          /* size of one scalar        */
                            numpy::format_descriptor<double>::format(), /* data type                 */
                            ndim,                                    /* number of dimensions      */
                            shape,                                   /* shape of the matrix       */
                            strides                                  /* strides for each axis     */
                            ));
}

PYBIND11_MODULE(pointcloud_downsample, m)
{
    m.doc() = "downsampling module for siamese pointnet"; // optional module docstring
    m.def("uniform", &py_downsample_uniform, "asdasdas",
          numpy::arg("array"), numpy::arg("output_cloud_size") = 128,
          numpy::arg("leaf_size_start") = 0.25, numpy::arg("leaf_size_step") = 0.01);
    //m.def("add", &downsample_uniform_add, "A function which adds two numbers");
    m.def("multiply", &py_multiply, "Convert all entries of an 1-D NumPy-array to int and multiply by 10");
    m.def("length", &py_length, "Calculate the length of an array of vectors");
}
