#include <iostream>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/block_solver.h>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

class VertexPose;
class EdgeProjection;

bool load_images(Mat &, Mat &, Mat &);

void DetectAndMatch(const cv::Mat &image1, const cv::Mat &image2,vector<cv::KeyPoint> &keypoints1, vector<cv::KeyPoint> &keypoints2,cv::Mat &descriptor1, cv::Mat &descriptor2, vector<cv::DMatch> &good_matches);

void SolvePnpGaussNewton(const vector<cv::KeyPoint> &keypoint1, const vector<cv::KeyPoint> &keypoint2, vector<cv::DMatch> matches, const Mat &depth, Mat &R, Mat &t);

void SolvePnpOpencv(const vector<cv::KeyPoint> &, const vector<cv::KeyPoint> &, vector<cv::DMatch>, const Mat &, Mat &, Mat &);

void SolvePnpG2O(const std::vector<cv::KeyPoint> &keypoints1,const std::vector<cv::KeyPoint> &keypoints2,const std::vector<cv::DMatch> &matches,const cv::Mat &depth1,cv::Mat &R, cv::Mat &t);

cv::Point2d pixel2cam(cv::KeyPoint, Mat);
Point2d pixel2cam(const Point2d &p, const Mat &K);
Eigen::Vector2d pixel2cam(const Eigen::Vector2d &p, const Mat &K);
int main()
{
    // load images
    Mat image1, image2, depth1;
    bool load_success = load_images(image1, image2, depth1);
    if (!load_success)
        return 1;

    // extract features and match
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;
    vector<cv::DMatch> good_matches;
    DetectAndMatch(image1, image2, keypoints1, keypoints2, descriptor1, descriptor2, good_matches);

    // run PnP solver in opencv
    cv::Mat R, t;
    SolvePnpOpencv(keypoints1, keypoints2, good_matches, depth1, R, t);
    cout << "Opencv R = " << R << endl;
    cout << "Opencv t = " << t << endl;

    // run SolvePnpGaussNewton
    cv::Mat R_g = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_g = cv::Mat::zeros(3, 1, CV_64F);
    SolvePnpGaussNewton(keypoints1, keypoints2, good_matches, depth1, R_g, t_g);
    cout << "Opencv R = " << R_g << endl;
    cout << "Opencv t = " << t_g << endl;

    // PnP with g2o (pose-only BA)
    cv::Mat R_g2o, t_g2o;
    SolvePnpG2O(keypoints1, keypoints2, good_matches, depth1, R_g2o, t_g2o);

    std::cout << "g2o BA result:    R= " << R_g2o << " t= " << t_g2o.t() << std::endl;

    // show images
    Mat result;
    drawMatches(image1, keypoints1, image2, keypoints2, good_matches, result);
    imshow("match result", result);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

bool load_images(Mat &image1, Mat &image2, Mat &depth1)
{
    image1 = imread("/home/amir/SLAM/review/ch7/images/1.png", IMREAD_COLOR);
    image2 = imread("/home/amir/SLAM/review/ch7/images/2.png", IMREAD_COLOR);
    depth1 = imread("/home/amir/SLAM/review/ch7/images/1_depth.png", IMREAD_UNCHANGED);

    if (image1.empty() || image2.empty() || depth1.empty())
    {
        cerr << "unable to find image path!!!";
        return 0;
    }
    return 1;
}

void DetectAndMatch(const cv::Mat &image1, const cv::Mat &image2,vector<cv::KeyPoint> &keypoints1, vector<cv::KeyPoint> &keypoints2,cv::Mat &descriptor1, cv::Mat &descriptor2, vector<cv::DMatch> &good_matches)
{
    // auto orb = cv::ORB::create();
    auto sift = cv::SIFT::create();
    // auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
    
    // Detect + Compute
    sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptor1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptor2);

    // KNN Match
    vector<vector<cv::DMatch>> matches_knn;
    matcher->knnMatch(descriptor1, descriptor2, matches_knn, 2);

    // Lowe's ratio test
    for (auto &m : matches_knn)
    {
        if (m.size() < 2)
            continue;
        if (m[0].distance < 0.1f * m[1].distance)
            good_matches.push_back(m[0]);
    }

    double min_dist = 10000, max_dist = 0;
    vector<DMatch> filtered_mathces;

    for (int i = 0; i < good_matches.size(); i++) {
        double dist = good_matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    for (int i = 0; i < good_matches.size(); i++) {
        if (good_matches[i].distance <= max(2 * min_dist, 30.0)) {
        filtered_mathces.push_back(good_matches[i]);
        }
    }
}

void SolvePnpOpencv(const vector<cv::KeyPoint> &keypoint1, const vector<cv::KeyPoint> &keypoint2, vector<cv::DMatch> matches, const Mat &depth, Mat &R, Mat &t)
{
    // Camera intrinsics (replace with actual calibration)
    const cv::Point2d principal_point(325.1, 249.7); // cx, cy
    const double focal_length = 521.0;
    const Mat K = (Mat_<double>(3, 3) << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1);

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches)
    {
        ushort d = depth.ptr<unsigned short>(int(keypoint1[m.queryIdx].pt.y))[int(keypoint1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint1[m.queryIdx], K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoint2[m.trainIdx].pt);
    }

    // solve pnp opencv
    cv::Mat distortion;
    vector<int> inliers;
    solvePnPRansac(pts_3d, pts_2d, K, distortion, R, t, false, 100, 3.0, 0.999, inliers, SOLVEPNP_ITERATIVE);
    Rodrigues(R, R);
    cout << "inliers = " << inliers.size() << " out of " << pts_2d.size() << " total points" << endl;
}

void SolvePnpGaussNewton(const vector<cv::KeyPoint> &keypoint1, const vector<cv::KeyPoint> &keypoint2, vector<cv::DMatch> matches, const Mat &depth, Mat &R, Mat &t)
{
    // Camera intrinsics (replace with actual calibration)
    const cv::Point2d principal_point(325.1, 249.7); // cx, cy
    const double focal_length = 521.0;
    const Mat K = (Mat_<double>(3, 3) << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1);
    Matrix3d K_e;
    K_e << focal_length, 0, principal_point.x, 0, focal_length, principal_point.y, 0, 0, 1;

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches)
    {
        ushort d = depth.ptr<unsigned short>(int(keypoint1[m.queryIdx].pt.y))[int(keypoint1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint1[m.queryIdx], K);
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoint2[m.trainIdx].pt);
    }

    // solve gaussian newton from scratch
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    Eigen::Matrix3d R_eigen;

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

    Sophus::SE3d pose(R_eigen, t_eigen);
    double cost = 0;

    for (int j = 0; j < 100; j++)
    {
        Matrix<double, 6, 1> b = Matrix<double, 6, 1>::Zero();
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        cost = 0;
        // loop through all measurements
        for (size_t i = 0; i < pts_2d.size(); i++)
        {
            Vector2d error = Vector2d::Zero();
            Vector2d pt2d(pts_2d[i].x, pts_2d[i].y);
            Vector3d pt3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z);
            pt3d = pose.rotationMatrix() * pt3d + pose.translation();

            // calculate reprojection error
            Vector2d proj(
                focal_length * pt3d(0) / pt3d(2) + principal_point.x,
                focal_length * pt3d(1) / pt3d(2) + principal_point.y);
            error = pt2d - proj;
            cost += error.transpose() * error;

            // calculate jacobian
            Matrix<double, 2, 6> J;
            J(0, 0) = (focal_length / pt3d(2));
            J(0, 1) = 0;
            J(0, 2) = -(focal_length * pt3d(0) / (pt3d(2) * pt3d(2)));
            J(0, 3) = -(focal_length * pt3d(0) * pt3d(1) / (pt3d(2) * pt3d(2)));
            J(0, 4) = focal_length + (focal_length * pt3d(0) * pt3d(0) / (pt3d(2) * pt3d(2)));
            J(0, 5) = -(focal_length * pt3d(1) / pt3d(2));

            J(1, 0) = 0;
            J(1, 1) = (focal_length / pt3d(2));
            J(1, 2) = -(focal_length * pt3d(1) / (pt3d(2) * pt3d(2)));
            J(1, 3) = -focal_length - (focal_length * pt3d(1) * pt3d(1) / (pt3d(2) * pt3d(2)));
            J(1, 4) = (focal_length * pt3d(0) * pt3d(1) / (pt3d(2) * pt3d(2)));
            J(1, 5) = (focal_length * pt3d(0) / pt3d(2));

            J = -J;

            b -= J.transpose() * error;
            H += J.transpose() * J;
        }

        Matrix<double, 6, 1> delta;
        delta = H.ldlt().solve(b);

        // update pose
        pose = Sophus::SE3d::exp(delta) * pose;

        // print results
        cout << "current cost function = " << cost << endl;
    }

    R_eigen = pose.rotationMatrix();
    t_eigen = pose.translation();

    cv::eigen2cv(R_eigen, R);
    cv::eigen2cv(t_eigen, t);
}

cv::Point2d pixel2cam(cv::KeyPoint kp, Mat K)

{
    cv::Point2d pt;

    pt.x = (kp.pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
    pt.y = (kp.pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1);

    return pt;
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
      (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
      );
  }

Eigen::Vector2d pixel2cam(const Eigen::Vector2d &p, const Eigen::Matrix3d &K) {
    return Eigen::Vector2d
      (
        (p.x() - K(0, 2)) / K(0, 0),
        (p.y() - K(1, 2)) / K(1, 1)
      );
}

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }
    void oplusImpl(const double *update) override {
        _estimate = Sophus::SE3d::exp(Eigen::Matrix<double, 6, 1>::Map(update)) * _estimate;
    }

    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeProjection(const Eigen::Vector3d& Pw, const Eigen::Matrix3d& K) : _Pw(Pw), _K(K) {}

    void computeError() override {
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d Pc = T * _Pw;
        Eigen::Vector3d pos_pixel = _K * Pc;
        pos_pixel /= pos_pixel[2];
        Eigen::Vector2d proj(pos_pixel[0], pos_pixel[1]);
        _error = _measurement - proj;
    }
    void linearizeOplus() override {
        VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d Pc = T * _Pw;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = Pc[0];
        double Y = Pc[1];
        double Z = Pc[2];
        double Z2 = Z * Z;
        _jacobianOplusXi << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z, 0, -fy / Z, fy * Y / Z2, fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }

private:
    Eigen::Vector3d _Pw;
    Eigen::Matrix3d _K;
};

void SolvePnpG2O(const std::vector<cv::KeyPoint> &keypoints1,const std::vector<cv::KeyPoint> &keypoints2,const std::vector<cv::DMatch> &matches,const cv::Mat &depth1,cv::Mat &R, cv::Mat &t)
{
    // 1. Build camera intrinsics
    double fx = 525.0, fy = 525.0, cx = 319.5, cy = 239.5;
    Eigen::Matrix3d K;
    K << fx, 0, cx,
    0, fy, cy,
    0,  0,  1;

    // 2. Collect 3D-2D correspondences
    VecVector3d pts_3d;
    VecVector2d pts_2d;
    for (auto &m : matches) {
    ushort d = depth1.ptr<unsigned short>(
    (int)keypoints1[m.queryIdx].pt.y)[(int)keypoints1[m.queryIdx].pt.x];
    if (d == 0) continue;
    double depth = d / 5000.0;
    Eigen::Vector2d p1_cam = pixel2cam(
    Eigen::Vector2d(keypoints1[m.queryIdx].pt.x,
                keypoints1[m.queryIdx].pt.y), K);
    pts_3d.emplace_back(p1_cam[0] * depth, p1_cam[1] * depth, depth);
    pts_2d.emplace_back(keypoints2[m.trainIdx].pt.x, keypoints2[m.trainIdx].pt.y);
    }

    // 3. Setup g2o optimizer
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
    std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 4. Add pose vertex
    auto v = new VertexPose();
    v->setId(0);
    v->setEstimate(Sophus::SE3d());
    optimizer.addVertex(v);

    // 5. Add edges
    for (size_t i = 0; i < pts_3d.size(); ++i) {
    auto e = new EdgeProjection(pts_3d[i], K);
    e->setVertex(0, v);
    e->setMeasurement(pts_2d[i]);
    e->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(e);
    }

    // 6. Optimize
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    Sophus::SE3d pose_est = v->estimate();
    Eigen::Matrix3d R_eigen = pose_est.rotationMatrix();
    Eigen::Vector3d t_eigen = pose_est.translation();
    cv::eigen2cv(R_eigen, R);
    cv::eigen2cv(t_eigen, t);
}
