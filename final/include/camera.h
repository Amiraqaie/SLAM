#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "common_include.h"

class Camera {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;
        Sophus::SE3d pose_;     // pose is T_cam_cam0
        Sophus::SE3d pose_inv_; // pose_inv is T_cam0_cam

        Camera();
        Camera(double fx, double fy, double cx, double cy, double baseline, Sophus::SE3d &pose)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
            pose_inv_ = pose_.inverse();
        };

        Sophus::SE3d pose() { return pose_; }

        // return intrinsic matrix
        Eigen::Matrix3d K() const {
            Eigen::Matrix3d k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        // T_c_w is w to cam0 translation 
        Eigen::Vector3d world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w); 

        Eigen::Vector3d camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w);

        Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c);

        Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth = 1);

        Eigen::Vector3d pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth = 1);

        Eigen::Vector2d world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w);

};

#endif