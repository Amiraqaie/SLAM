#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

#include "common_include.h"
#include <opencv2/calib3d.hpp>
#include <camera.h>

namespace myslam
{
    inline bool triangulation(const std::vector<Sophus::SE3d> &poses,
                            const std::vector<Eigen::Vector3d> points, 
                            Eigen::Vector3d &pt_world) {

        MatXX A(2 * poses.size(), 4);
        VecX b(2 * poses.size());
        b.setZero();
        for (size_t i = 0; i < poses.size(); ++i) {
            Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }
        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
            return true;
        }
        return false;
    }


    inline Eigen::Vector2d toVec2(const cv::Point2f p) { return Eigen::Vector2d(p.x, p.y); }

}

#endif // MYSLAM_ALGORITHM_H


