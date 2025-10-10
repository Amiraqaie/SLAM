#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

struct PoseGraphEdge {
    int i, j;
    SE3d Tij;
    Matrix<double, 6, 6> information;
};

// Convert SE3 pose to 7D double array [tx, ty, tz, qx, qy, qz, qw]
void SE3ToDoubleArray(const SE3d &pose, double *pose_array) {
    Quaterniond q = pose.unit_quaternion();
    Vector3d t = pose.translation();
    pose_array[0] = t.x();
    pose_array[1] = t.y();
    pose_array[2] = t.z();
    pose_array[3] = q.x();
    pose_array[4] = q.y();
    pose_array[5] = q.z();
    pose_array[6] = q.w();
}

// Convert double array to SE3
SE3d DoubleArrayToSE3(const double *pose_array) {
    Vector3d t(pose_array[0], pose_array[1], pose_array[2]);
    Quaterniond q(pose_array[6], pose_array[3], pose_array[4], pose_array[5]);
    q.normalize();
    return SE3d(q, t);
}

// Templated version for Jet types (autodiff)
template <typename T>
Sophus::SE3<T> PoseToSE3(const T* pose_array) {
    Eigen::Matrix<T, 3, 1> t(pose_array[0], pose_array[1], pose_array[2]);
    Eigen::Quaternion<T> q(pose_array[6], pose_array[3], pose_array[4], pose_array[5]);
    q.normalize();
    return Sophus::SE3<T>(q, t);
}

// Ceres cost function
struct PoseGraphErrorTerm {
    PoseGraphErrorTerm(const SE3d &Tij, const Matrix<double, 6, 6> &info)
        : Tij_(Tij), sqrt_info_(info.llt().matrixL()) {}

    template <typename T>
    bool operator()(const T *pose_i, const T *pose_j, T *residuals) const {
        Sophus::SE3<T> Ti = PoseToSE3(pose_i);
        Sophus::SE3<T> Tj = PoseToSE3(pose_j);
        Sophus::SE3<T> Tij = Tij_.template cast<T>();

        Sophus::SE3<T> err = Tij.inverse() * Ti.inverse() * Tj;
        Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);
        res = sqrt_info_.template cast<T>() * err.log();
        return true;
    }

    static ceres::CostFunction *Create(const SE3d &Tij, const Matrix<double, 6, 6> &info) {
        return new ceres::AutoDiffCostFunction<PoseGraphErrorTerm, 6, 7, 7>(
            new PoseGraphErrorTerm(Tij, info));
    }

    const SE3d Tij_;
    Matrix<double, 6, 6> sqrt_info_;
};

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_ceres sphere.g2o" << endl;
        return -1;
    }

    // Read .g2o file
    ifstream fin(argv[1]);
    if (!fin) {
        cerr << "Cannot open file " << argv[1] << endl;
        return -1;
    }

    vector<double *> poses;
    vector<PoseGraphEdge> edges;

    while (!fin.eof()) {
        string tag;
        fin >> tag;
        if (tag == "VERTEX_SE3:QUAT") {
            int idx;
            double tx, ty, tz, qx, qy, qz, qw;
            fin >> idx >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            SE3d pose(q, Vector3d(tx, ty, tz));
            double *pose_arr = new double[7];
            SE3ToDoubleArray(pose, pose_arr);
            if (idx >= poses.size()) poses.resize(idx + 1);
            poses[idx] = pose_arr;
        } else if (tag == "EDGE_SE3:QUAT") {
            PoseGraphEdge edge;
            double tx, ty, tz, qx, qy, qz, qw;
            fin >> edge.i >> edge.j >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            Quaterniond q(qw, qx, qy, qz);
            q.normalize();
            edge.Tij = SE3d(q, Vector3d(tx, ty, tz));
            for (int i = 0; i < 6; ++i)
                for (int j = i; j < 6; ++j) {
                    fin >> edge.information(i, j);
                    if (i != j) edge.information(j, i) = edge.information(i, j);
                }
            edges.push_back(edge);
        }
    }

    // Build optimization problem
    ceres::Problem problem;
    for (const auto &edge : edges) {
        ceres::CostFunction *cost = PoseGraphErrorTerm::Create(edge.Tij, edge.information);
        problem.AddResidualBlock(cost, nullptr, poses[edge.i], poses[edge.j]);
    }

    // Fix the first pose
    problem.SetParameterBlockConstant(poses[0]);

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;

    // Save results
    ofstream fout("result_ceres.g2o");
    for (size_t i = 0; i < poses.size(); ++i) {
        SE3d pose = DoubleArrayToSE3(poses[i]);
        Quaterniond q = pose.unit_quaternion();
        Vector3d t = pose.translation();
        fout << "VERTEX_SE3:QUAT " << i << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    for (const auto &edge : edges) {
        Quaterniond q = edge.Tij.unit_quaternion();
        Vector3d t = edge.Tij.translation();
        fout << "EDGE_SE3:QUAT " << edge.i << " " << edge.j << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";
        for (int i = 0; i < 6; ++i)
            for (int j = i; j < 6; ++j)
                fout << edge.information(i, j) << " ";
        fout << endl;
    }
    fout.close();

    // Free memory
    for (auto *p : poses) delete[] p;

    return 0;
}
