#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

void pose_estimation_2d2d(std::vector<cv::KeyPoint> key_point1, 
                          std::vector<cv::KeyPoint> key_point2, 
                          std::vector<cv::DMatch> matches,
                          cv::Mat &R, cv::Mat &t)
{
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

int main(int argc, char **argv)
{
    // if (argc != 3)
    // {
    //     std::cout << "usage: pose_estimation_2d2d img1 img2" << std::endl;
    //     return -1;
    // }

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
    cv::waitKey(0);

    //−− Estimate the motion between two frames
    cv::Mat R, t;
    pose_estimation_2d2d(keypoint_1, keypoint_2, matches, R, t);

    //−− Check E=t^R∗scale
    cv::Mat t_x =
    (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    std::cout << "t^R=" << std::endl << t_x * R << std::endl;


    //−− Check epipolar constraints
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m: matches) {
        cv::Point2d pt1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint = " << d << std::endl;
    }
    return 0;
}