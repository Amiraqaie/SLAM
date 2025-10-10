#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <chrono>
#include "sophus/se3.hpp"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

// BA by g2o
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// BA by gauss-newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose);

void convertMatToEigen(cv::Mat R, cv::Mat t, Eigen::Matrix3d &R_eigen, Eigen::Vector3d &t_eigen);

void bundleAdjustmentG2O(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose);

int main(int argc, char **argv)
{

    // −− Fetch images
    cv::Mat image_1 = cv::imread("../1.png", cv::IMREAD_COLOR);
    cv::Mat image_2 = cv::imread("../2.png", cv::IMREAD_COLOR);
    assert(image_1.data && image_2.data && "Can not load images!");

    cv::Mat d1 = imread("../1_depth.png", cv::IMREAD_UNCHANGED);
    cv::Mat d2 = imread("../1_depth.png", cv::IMREAD_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- initialize feature matching
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    cv::Mat descriptor_1, descriptor_2;
    std::vector<cv::DMatch> matches;

    find_feature_matches(image_1, image_2, keypoint_1, keypoint_2, matches);

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches)
    {
        ushort d = d1.ptr<unsigned short>(int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoint_2[m.trainIdx].pt);
    }

    std::cout << "3d-2d pairs: " << pts_3d.size() << std::endl;

    // solve PnP problem using opencv
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cv::Mat r, t;
    solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << std::endl;

    std::cout << "R=" << std::endl
              << R << std::endl;
    std::cout << "t=" << std::endl
              << t << std::endl;

    // solving BA by g-n
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    std::cout << "calling bundle adjustment by gauss newton" << std::endl;
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    Eigen::Matrix3d R_eigen;
    convertMatToEigen(R, t, R_eigen, t_eigen);
    Sophus::SE3d pose_gn(R_eigen, t_eigen);
    t1 = std::chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << std::endl;

    // // solve BA by g2o
    // Sophus::SE3d pose_g2o(R_eigen, t_eigen);
    // bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::KeyPoint> &keypoints_2, std::vector<cv::DMatch> &matches)
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::GFTTDetector::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= cv::max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void bundleAdjustmentGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose)
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    const int iteration = 1000;

    double cost = 0, lastCost = 0;

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 2);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int i = 0; i < iteration; i++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;

        // compute cost
        for (int i = 0; i < points_3d.size(); i++)
        {
            Eigen::Vector3d pc = pose * points_3d[i];       // convert world coordinate to camera
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            Eigen::Vector2d e = points_2d[i] - proj;
            cost += e.squaredNorm();

            // calculating jacobian matrix
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
            0,
            fx * pc[0] * inv_z2,
            fx * pc[0] * pc[1] * inv_z2,
            -fx - fx * pc[0] * pc[0] * inv_z2,
            fx * pc[1] * inv_z,
            0,
            -fy * inv_z,
            fy * pc[1] * inv_z,
            fy + fy * pc[1] * pc[1] * inv_z2,
            -fy * pc[0] * pc[1] * inv_z2,
            -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            std::cout << "result is nan!" << std::endl;
            break;
        }
        std::cout << "iteration " << i << " cost=" <<  cost << std::endl;

        if (i > 0 && cost >= lastCost) {
            // cost increase, update is not good
            std::cout << "cost: " << cost << ", last cost: " << lastCost << std::endl;
            break;
        }
        
        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }
    std::cout << "pose by g-n: \n" << pose.matrix() << std::endl;
}

void convertMatToEigen(cv::Mat R, cv::Mat t, Eigen::Matrix3d &R_eigen, Eigen::Vector3d &t_eigen)
{
    t_eigen(0) = t.at<double>(0);
    t_eigen(1) = t.at<double>(1);
    t_eigen(2) = t.at<double>(2);
    
    R_eigen(0, 0) = R.at<double>(0, 0);
    R_eigen(0, 1) = R.at<double>(0, 1);
    R_eigen(0, 2) = R.at<double>(0, 2);

    R_eigen(1, 0) = R.at<double>(1, 0);
    R_eigen(1, 1) = R.at<double>(1, 1);
    R_eigen(1, 2) = R.at<double>(1, 2);

    R_eigen(2, 0) = R.at<double>(2, 0);
    R_eigen(2, 1) = R.at<double>(2, 1);
    R_eigen(2, 2) = R.at<double>(2, 2);
}

void bundleAdjustmentG2O(const VecVector3d &points_3d, const VecVector2d &points_2d, const cv::Mat &K, Sophus::SE3d &pose);

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    // left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &is) override {}
    virtual bool write(std::ostream &os) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos),_K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
          << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
          0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
      }

    virtual bool read(std::istream& is) override { return true; }
    virtual bool write(std::ostream& os) const override { return true; }
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};
    
void bundleAdjustmentG2O(const VecVector3d &points_3d,const VecVector2d &points_2d,const cv::Mat &K,Sophus::SE3d &pose) {
  
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // Gradient descent method, you can choose from GN, LM, DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // Graph model
    optimizer.setAlgorithm(solver);   // Set up the solver
    optimizer.setVerbose(true);       // Turn on verbose output for debugging
  
    // vertex
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);
  
    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
      K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
      K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);
  
    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
      auto p2d = points_2d[i];
      auto p3d = points_3d[i];
      EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
      edge->setId(index);
      edge->setVertex(0, vertex_pose);
      edge->setMeasurement(p2d);
      edge->setInformation(Eigen::Matrix2d::Identity());
      optimizer.addEdge(edge);
      index++;
    }
  
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "optimization costs time: " << time_used.count() << " seconds." << std::endl;
    std::cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << std::endl;
    pose = vertex_pose->estimate();
  }