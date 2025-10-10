#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pangolin/pangolin.h>

using namespace std;

void DetectAndMatch(
    const cv::Mat &image1, const cv::Mat &image2,
    vector<cv::KeyPoint> &keypoints1, vector<cv::KeyPoint> &keypoints2,
    cv::Mat &descriptor1, cv::Mat &descriptor2, vector<cv::DMatch> &good_matches)
{
    // auto orb = cv::ORB::create();
    auto orb = cv::SIFT::create();
    // auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);

    // Detect + Compute
    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptor1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptor2);

    // KNN Match
    vector<vector<cv::DMatch>> matches_knn;
    matcher->knnMatch(descriptor1, descriptor2, matches_knn, 2);

    // Lowe's ratio test
    for (auto &m : matches_knn)
    {
        if (m.size() < 2)
            continue;
        if (m[0].distance < 0.3f * m[1].distance)
            good_matches.push_back(m[0]);
    }
}

void pose_estimation_2d_2d(
    const vector<cv::KeyPoint> &keypoints1,
    const vector<cv::KeyPoint> &keypoints2,
    const vector<cv::DMatch> &good_matches,
    cv::Mat &R, cv::Mat &t)
{
    // Camera intrinsics (replace with actual calibration)
    const cv::Point2d principal_point(325.1, 249.7); // cx, cy
    const double focal_length = 521.0;

    // Extract matched points
    vector<cv::Point2f> points1, points2;
    points1.reserve(good_matches.size());
    points2.reserve(good_matches.size());

    for (const auto &match : good_matches)
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    if (points1.size() < 5 || points2.size() < 5)
    {
        cerr << "Not enough points for pose estimation!";
        return;
    }

    // Find Essential Matrix with RANSAC to remove outliers
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(
        points1, points2,
        focal_length, principal_point,
        cv::RANSAC, 0.9999, 1.0, inlier_mask);

    // Recover relative camera rotation and translation
    cv::recoverPose(E, points1, points2, R, t, focal_length, principal_point, inlier_mask);

    cout << "Number of inliers: " << cv::countNonZero(inlier_mask) << " / " << good_matches.size() << endl;
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void triangulation(const std::vector<cv::KeyPoint> &keypoint_1, const std::vector<cv::KeyPoint> &keypoint_2, const std::vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t, std::vector<cv::Point3d> &points)
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2f> pts_1, pts_2;
    for (cv::DMatch m : matches)
    {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++)
    {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        cv::Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        points.push_back(p);
    }
}

inline cv::Scalar get_color(float depth)
{
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th)
        depth = up_th;
    if (depth < low_th)
        depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &points)
{
    if (points.empty())
    {
        cerr << "Point cloud is empty" << endl;
        return;
    }

    // create a point cloud pointer
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    cloud->reserve(points.size());

    for (const auto &p : points)
    {
        pcl::PointXYZ pt;
        if (p[3] != 0)
        {
            pt.x = static_cast<float>(p[0] / p[3]);
            pt.y = static_cast<float>(p[1] / p[3]);
            pt.z = static_cast<float>(p[2] / p[3]);
        }
        else
        {
            pt.x = static_cast<float>(p[0]);
            pt.y = static_cast<float>(p[1]);
            pt.z = static_cast<float>(p[2]);
        }
        cloud->points.push_back(pt);
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::visualization::CloudViewer viewer("point cloud");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

void show_point_cloud(const vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &points, const string &window_name)
{
    pangolin::CreateWindowAndBind(window_name, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (const auto &point : points)
        {
            // Color based on Z-depth (blue to red)
            float r = point(2) / 10.0f;
            float b = 1.0f - r;
            glColor3f(r, 0.0f, b);
            glVertex3f(point(0), point(1), point(2));
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);
    }
}

int main()
{
    cv::Mat image1;
    cv::Mat image2;

    image1 = cv::imread("/home/amir/SLAM/review/ch7/images/1.png");
    image2 = cv::imread("/home/amir/SLAM/review/ch7/images/2.png");

    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;
    vector<cv::DMatch> good_matches;
    cv::Mat R, t;

    DetectAndMatch(image1, image2, keypoints1, keypoints2, descriptor1, descriptor2, good_matches);
    pose_estimation_2d_2d(keypoints1, keypoints2, good_matches, R, t);

    // show results without filtering
    cv::Mat result;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, result);
    cv::imshow("good_mathces", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // print result
    cout << "rotation matrix is = " << R << endl;
    cout << "translation vector is = " << t << endl;

    // triangulation
    std::vector<cv::Point3d> points;
    triangulation(keypoints1, keypoints2, good_matches, R, t, points);

    // convert to point cloud and show it
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;
    for (cv::Point3d p : points)
    {
        if (p.z > 0)
            pointcloud.emplace_back(p.x, p.y, p.z, 1.0);
    }

    // showPointCloud
    show_point_cloud(pointcloud, "point cloud");
    return 0;
}