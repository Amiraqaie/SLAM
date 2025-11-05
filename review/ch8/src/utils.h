#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<bool> &success, bool inverse = false, bool has_initial_guess = false);


void OpticalFlowMultiLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,vector<bool> &success,bool inverse = false);

bool load_images(Mat &image1, Mat &image2)
{
    image1 = imread("/home/havaie/Git/SLAM/review/ch8/images/LK1.png", IMREAD_GRAYSCALE);
    image2 = imread("/home/havaie/Git/SLAM/review/ch8/images/LK2.png", IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty())
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