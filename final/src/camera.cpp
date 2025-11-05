#include <camera.h>

Camera::Camera()
{

}

Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
    return pose_ * T_c_w * p_w;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w)
{
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
    double u = fx_ * (p_c[0] / p_c[2]) + cx_;
    double v = fy_ * (p_c[1] / p_c[2]) + cy_;

    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
    double X = depth * (p_p[0] - cx_) / fx_;
    double Y = depth * (p_p[1] - cy_) / fy_;
    double Z = depth;

    return Eigen::Vector3d(X, Y, Z);
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth)
{
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
    return camera2pixel(world2camera(p_w, T_c_w));
}
