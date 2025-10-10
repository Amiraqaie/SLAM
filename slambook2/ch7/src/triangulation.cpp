#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <pangolin/pangolin.h>
#include <unistd.h>

void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> key_point1, std::vector<cv::KeyPoint> key_point2, std::vector<cv::DMatch> matches, cv::Mat &R, cv::Mat &t){
    // Camera Intrinsics,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //−− Convert the matching point to the form of vector<Point2f>
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++)
    {
        points1.push_back(key_point1[matches[i].queryIdx].pt);
        points2.push_back(key_point2[matches[i].trainIdx].pt);
    }

    //−− Calculate fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    std::cout << "fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    //−− Calculate essential matrix
    cv::Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    cv::Mat essential_matrix;
    // essential_matrix = cv::findEssentialMat(points1, points2, focal_length,
    //     principal_point);
    essential_matrix = cv::findEssentialMat(points1, points2, K);
    std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    //−− Calculate homography matrix
    //−− But the scene is not planar, and calculating the homography matrix here is of
    // little significance
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    //−− Recover rotation and translation from the essential matrix.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t << std::endl;
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
      (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
      );
}

void triangulation( const std::vector<cv::KeyPoint> &keypoint_1, const std::vector<cv::KeyPoint> &keypoint_2, const std::vector<cv::DMatch> &matches, const cv::Mat &R, const cv::Mat &t, std::vector<cv::Point3d> &points) {
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
                                            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                                            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                                            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
                                          );    
                                          
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    
    std::vector<cv::Point2f> pts_1, pts_2;
    for (cv::DMatch m:matches) {
      pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
      pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
  
    for (int i = 0; i < pts_4d.cols; i++) {
      cv::Mat x = pts_4d.col(i);
      x /= x.at<float>(3, 0); // 归一化
      cv::Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
      );
      points.push_back(p);
    }
}

inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
  }

int main(int argc, char **argv)
{

    //−− Fetch images
    cv::Mat image_1 = cv::imread("../1.png", cv::IMREAD_COLOR);
    cv::Mat image_2 = cv::imread("../2.png", cv::IMREAD_COLOR);

    //-- initialize feature matching
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    cv::Mat descriptor_1, descriptor_2;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::FeatureDetector> detector = cv::GFTTDetector::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    // find features in both images
    detector->detect(image_1, keypoint_1);
    detector->detect(image_2, keypoint_2);

    // extract descriptors
    descriptor->compute(image_1, keypoint_1, descriptor_1);
    descriptor->compute(image_2, keypoint_2, descriptor_2);

    // match extracted descriptors
    matcher->match(descriptor_1, descriptor_2, matches);

    auto min_max = minmax_element(matches.begin(), matches.end(),
    [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);


    // remove bad matchs
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptor_1.rows; i++) {
      if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
      }
    }

    cv::Mat img_goodmatch;
    cv::drawMatches(image_1, keypoint_1, image_2, keypoint_2, good_matches, img_goodmatch);
    imshow("good matches", img_goodmatch);

    //−− Estimate the motion between two frames
    cv::Mat R, t;
    pose_estimation_2d2d(keypoint_1, keypoint_2, good_matches, R, t);

    //−− Check E=t^R∗scale
    cv::Mat t_x =
    (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    std::cout << "t^R=" << std::endl << t_x * R << std::endl;


    //−− Check epipolar constraints
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m: good_matches) {
        cv::Point2d pt1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint = " << d << std::endl;
    }

    // triangulation
    std::vector<cv::Point3d> points;
    triangulation(keypoint_1, keypoint_2, good_matches, R, t, points);

    cv::Mat img1_plot = image_1.clone();
    cv::Mat img2_plot = image_2.clone();
    for (int i = 0; i < good_matches.size(); i++) {
      float depth1 = points[i].z;
      std::cout << "depth: " << depth1 << std::endl;
      cv::Point2d pt1_cam = pixel2cam(keypoint_1[good_matches[i].queryIdx].pt, K);
      cv::circle(img1_plot, keypoint_1[good_matches[i].queryIdx].pt, 2, get_color(depth1), 2);
  
      // 第二个图
      cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
      float depth2 = pt2_trans.at<double>(2, 0);
      cv::circle(img2_plot, keypoint_2[good_matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }

    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey(0);

    // convert to point cloud and show it
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud;
    for (cv::Point3d p: points)
    {  
        if (p.z > 0)
            pointcloud.emplace_back(p.x, p.y, p.z, 0.5);
    }
    showPointCloud(pointcloud);
    return 0;
}