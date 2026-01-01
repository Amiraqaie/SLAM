#ifndef G2O_TYPES_H
#define G2O_TYPES_H

#include "common_include.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_dogleg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> upd;
        for (int i = 0; i < 6; ++i) {
            upd[i] = update[i];
        }
        _estimate = Sophus::SE3d::exp(upd) * _estimate;
    }

    virtual bool read(std::istream& in) override {
        return true;
    }

    virtual bool write(std::ostream& out) const override {
        return true;
    }
};

class VertexXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d::Zero();
    }

    virtual void oplusImpl(const double* update) override {
        for (int i = 0; i < 3; ++i) {
            _estimate[i] += update[i];
        }
    }

    virtual bool read(std::istream& in) override {
        return true;
    }

    virtual bool write(std::ostream& out) const override {
        return true;
    }
};

class EdgeProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Matrix3d& K, const Sophus::SE3d& cam_ext) : _K(K)
    {
        _cam_ext = cam_ext;
    }

    void computeError() override {
        const VertexPose* v_pose = static_cast<const VertexPose*>(_vertices[0]);
        const VertexXYZ* v_point = static_cast<const VertexXYZ*>(_vertices[1]);

        Sophus::SE3d pose = v_pose->estimate();
        Eigen::Vector3d point_world = v_point->estimate();

        Eigen::Vector3d point_cam = _cam_ext * (pose * point_world);
        Eigen::Vector2d proj;
        proj[0] = _K(0, 0) * point_cam[0] / point_cam[2] + _K(0, 2);
        proj[1] = _K(1, 1) * point_cam[1] / point_cam[2] + _K(1, 2);

        _error = proj - _measurement;
    }

    virtual void linearizeOplus() override {
        const VertexPose* v_pose = static_cast<const VertexPose*>(_vertices[0]);
        const VertexXYZ* v_point = static_cast<const VertexXYZ*>(_vertices[1]);

        Sophus::SE3d T = v_pose->estimate();
        Eigen::Vector3d pw = v_point->estimate();
        Eigen::Vector3d pos_cam = _cam_ext * T * pw;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;

        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                           _cam_ext.rotationMatrix() * T.rotationMatrix();
    }

    virtual bool read(std::istream& in) override {
        return true;
    }

    virtual bool write(std::ostream& out) const override {
        return true;
    }

private:
    Eigen::Matrix3d _K;
    Sophus::SE3d _cam_ext;
};

class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectionPoseOnly(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K)
        : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }

   private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

#endif