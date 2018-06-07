##########################################################
##################### CMAKE RELATED ######################
##########################################################

cmake_minimum_required(VERSION 3.5)
project(pointcloud_downsample)

##########################################################
################## EXTERNAL LIBRARIES ####################
##########################################################

# PCL
find_package(PCL 1.7.1 REQUIRED)
if (PCL_FOUND)
    message("-- PCL ${PCL_VERSION} was found")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else(PCL_FOUND)
    message(FATAL_ERROR "PCL was not found")
endif(PCL_FOUND)

# VTK
if(NOT VTK_BINARY_DIR)
  find_package(VTK)
  if(NOT VTK_DIR)
    message(FATAL_ERROR "Please set VTK_DIR.")
  endif(NOT VTK_DIR)
  include(${VTK_USE_FILE})
endif(NOT VTK_BINARY_DIR)

# Find Boost
find_package(Boost COMPONENTS system filesystem python program_options REQUIRED)
if(Boost_FOUND)
    message("-- Boost ${Boost_VERSION} was found")
    include_directories(${Boost_INCLUDE_DIRS})
else(BOOST_FOUND)
    message(FATAL_ERROR "Boost was not found")
endif(Boost_FOUND)

##########################################################
######################## C++11 ###########################
##########################################################

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "-std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    message(STATUS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

##########################################################
##################### CMAKE RELATED ######################
##########################################################

add_subdirectory(pybind11)
include_directories(${PROJECT_SOURCE_DIR})

##########################################################
############## SIAMESEPOINTNET C++ MODULE ################
##########################################################

add_library(${PROJECT_NAME}_lib SHARED ${PROJECT_NAME}_lib.cpp ${PROJECT_NAME}_lib.h
            ${PROJECT_SOURCE_DIR}/pcseg/cplusplus/src/sampling.cpp)
target_link_libraries(${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${VTK_LIBRARIES})

##########################################################
############## SIAMESEPOINTNET PYTHON WRAPPER ################
##########################################################

pybind11_add_module(${PROJECT_NAME} ${PROJECT_NAME}_py.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_lib)

add_executable(${PROJECT_NAME}_cpp ${PROJECT_NAME}_cpp.cpp)
target_link_libraries(${PROJECT_NAME}_cpp ${PROJECT_NAME}_lib ${Boost_LIBRARIES})

##########################################################
############## SIAMESEPOINTNET INSTALL ################
##########################################################

install(TARGETS ${PROJECT_NAME}_lib DESTINATION lib)
install(FILES ${PROJECT_NAME}_lib.h DESTINATION include)
