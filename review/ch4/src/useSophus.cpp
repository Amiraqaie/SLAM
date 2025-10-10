#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int, char **)
{

    Matrix3d R = AngleAxisd(M_PI / 4, Vector3d(0, 0, 1)).toRotationMatrix();

    // R is equal to this =
    // 0.707107 -0.707107         0
    // 0.707107  0.707107         0
    // 0         0                1

    Quaterniond q(R);

    // Sophus SO3
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);

    // logarithmic map on SO3
    Vector3d so3 = SO3_R.log();
    Matrix3d so3_hat = Sophus::SO3d::hat(so3);

    // update by perturbation model
    Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;

    // SE3
    Vector3d t(1, 0, 0);
    Sophus::SE3 SE3_R_t(R, t);
    Sophus::SE3 SE3_q_t(q, t);

    cout << "SE3_R_t = " << SE3_R_t.matrix() << endl;

    // logarithmic mapping
    typedef Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_R_t.log();
    cout << "se3 = " << se3 << endl;
    cout << "se3 hat = " << Sophus::SE3d::hat(se3) << endl;

    // update
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_R_t;

    return 0;
}
