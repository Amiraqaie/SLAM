#pragma once

#ifndef COMMON_INCLUDE_H
#define COMMON_INCLUDE_H

// std
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// typedef for eigen
// double matrices
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;

// float matrices
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;

// double vector
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

// float vector
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// for cv
#include <opencv2/core/core.hpp>

// glog
#include <glog/logging.h>

#endif