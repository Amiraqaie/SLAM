//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "common_include.h"
#include <opencv2/calib3d.hpp>
#include <camera.h>

namespace myslam
{

    /**
     * linear triangulation with SVD
     * @param poses     poses,
     * @param points    points in normalized plane
     * @param pt_world  triangulated point in the world
     * @return true if success
     */
    inline bool triangulation(const std::vector<SE3> &poses,
                              const std::vector<Vec3> points, Vec3 &pt_world)
    {
        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();
        for (size_t i = 0; i < poses.size(); ++i)
        {
            Mat34 m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        // auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        // Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // Eigen::BDCSVD<Eigen::MatrixXd> svd;
        // svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);

        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2)
        {
            // 解质量不好，放弃
            return true;
        }
        return false;
    }

    inline bool triangulation_opencv(const std::vector<Mat34> &projections,
                                     const std::vector<Camera::Ptr> cameras,
                                     const std::vector<Vec3> &points,
                                     Vec3 &pt_world)
    {
        assert(projections.size() == 2 && points.size() == 2);

        // Get projection matrices: P = K * [R|t], but since points are normalized, K = I
        Mat34 P1 = projections[0];
        Mat34 P2 = projections[1];

        // Convert to OpenCV Mat
        cv::Mat proj1(3, 4, CV_64F), proj2(3, 4, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                proj2.at<double>(r, c) = P2(r, c);
                proj1.at<double>(r, c) = P1(r, c);
            }            
        }

        // Input points: normalized camera coordinates
        cv::Mat pts_1(2, 1, CV_64F), pts_2(2, 1, CV_64F);
        Vec2 ptc1 = cameras[0]->camera2pixel(points[0]);
        Vec2 ptc2 = cameras[1]->camera2pixel(points[1]);
        pts_1.at<double>(0, 0) = ptc1[0];
        pts_1.at<double>(1, 0) = ptc1[1];
        pts_2.at<double>(0, 0) = ptc2[0];
        pts_2.at<double>(1, 0) = ptc2[1];

        // Triangulate
        cv::Mat point_4d_homo;
        cv::triangulatePoints(proj1, proj2, pts_1, pts_2, point_4d_homo);
        // Convert from homogeneous to Euclidean
        cv::Mat pt_3d = point_4d_homo.col(0);
        pt_world = Vec3(pt_3d.at<double>(0) / pt_3d.at<double>(3),
                        pt_3d.at<double>(1) / pt_3d.at<double>(3),
                        pt_3d.at<double>(2) / pt_3d.at<double>(3));

        // Check if point is in front of camera
        if (pt_world[2] > 0)
        {
            return true;
        }

        return false;
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

} // namespace myslam

#endif // MYSLAM_ALGORITHM_H
