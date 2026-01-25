#pragma once

#ifndef FINAL_PROJECT_FRAME_H
#define FINAL_PROJECT_FRAME_H

#include "camera.h"
#include "common_include.h"

namespace myslam{

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
    Sophus::SE3d pose_;     // T_c_w
    Sophus::SE3d gt_pose_;
    std::mutex pose_mutex_;
    std::mutex gt_pose_mutex_;
    std::mutex feature_mutex_;
    std::mutex keypoint_mutex_;
    cv::Mat left_img_, right_img_;

    std::vector<std::shared_ptr<Feature>> features_left_;
    std::vector<std::shared_ptr<Feature>> features_right_;
    cv::Mat orb_descriptors_left_;
    cv::Mat orb_descriptors_right_;
    std::vector<cv::KeyPoint> valid_keypoints_left;
    std::vector<cv::KeyPoint> valid_keypoints_right;

    
public:
    Frame() {}
    Frame(long id, double time_stamp, const Sophus::SE3d &pose, const cv::Mat &left, const cv::Mat &right);
    
    Sophus::SE3d Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    Sophus::SE3d GtPose() {
        std::unique_lock<std::mutex> lck(gt_pose_mutex_);
        return gt_pose_;
    }
    
    void SetPose(const Sophus::SE3d &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    void SetGtPose(const Sophus::SE3d &gt_pose) {
        std::unique_lock<std::mutex> lck(gt_pose_mutex_);
        gt_pose_ = gt_pose;
    }

    void SetKeyFrame();

    std::vector<cv::KeyPoint> GetKeypointsLeft();
    std::vector<cv::KeyPoint> GetKeypointsRight();

    std::vector<cv::KeyPoint> GetValidKeypointsLeft();
    std::vector<cv::KeyPoint> GetValidKeypointsRight();

    cv::Mat GetDescriptorsLeft();
    cv::Mat GetDescriptorsRight();
    
    static std::shared_ptr<Frame> CreateFrame();
};

}

#endif  //MYSLAM_FRAME_H