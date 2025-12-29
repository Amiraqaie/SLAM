#include <iostream>
#include <memory>       // std::make_unique
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <glog/logging.h>
#include <pangolin/pangolin.h>

// Fix DBoW3 dynamic exception spec for C++17
#if __cplusplus >= 201703L
#define throw(...)
#endif
#include <DBoW3/Vocabulary.h>

// g2o includes
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

int main(int argc, char** argv)
{
    // ---------------- glog ----------------
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "glog is working";

    // ---------------- Eigen ----------------
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    std::cout << "Eigen OK:\n" << I << std::endl;

    // ---------------- OpenCV ----------------
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    std::cout << "OpenCV OK: " << img.rows << "x" << img.cols << std::endl;

    // ---------------- DBoW3 ----------------
    DBoW3::Vocabulary voc(10, 6, DBoW3::TF_IDF, DBoW3::L1_NORM);
    std::cout << "DBoW3 OK: vocab created" << std::endl;

    // ---------------- g2o ----------------
    using BlockSolverType  = g2o::BlockSolver<g2o::BlockSolverTraits<6,3>>;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver  = std::make_unique<BlockSolverType>(std::move(linearSolver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver)));
    std::cout << "g2o OK: optimizer created" << std::endl;

    // ---------------- Pangolin ----------------
    pangolin::CreateWindowAndBind("Test Window", 320, 240);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pangolin::FinishFrame();
    std::cout << "Pangolin OK: window created" << std::endl;

    return 0;
}
