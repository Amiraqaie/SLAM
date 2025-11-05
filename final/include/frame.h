#pragma once

#ifndef FRAME_H
#define FRAME_H

#include <eigen3/Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <mutex>
#include <opencv2/core.hpp>
#include <vector>

struct Frame
{   
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;
    unsigned long key_frame_id_ = 0;
    bool is_keyframe_ = false;
    double time_stamp_;
    Sophus::SE3d pose_;
    std::mutex pose_mutex_;
    cv::Mat left_img_, right_img_;
    
};

#endif