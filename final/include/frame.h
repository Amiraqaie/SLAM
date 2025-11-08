#pragma once

#ifndef FRAME_H
#define FRAME_H

#include "camera.h"
#include "common_include.h"

struct Feature;

struct Frame
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;
    unsigned long keyframe_id_ = 0;
    bool is_keyframe_ = false;
    double time_stamp_;
    Sophus::SE3d pose_;
    std::mutex pose_mutex_;
    cv::Mat left_img_, right_img_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;

public:
    Frame() {}
    Frame(long id, double time_stamp, const Sophus::SE3d &pose, const cv::Mat &left, const cv::Mat &right);

    Sophus::SE3d Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }
    
    void SetPose(const Sophus::SE3d &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    void SetKeyFrame();

    static std::shared_ptr<Frame> CreateFrame();
};


#endif