#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>

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
        if (m[0].distance < 0.1f * m[1].distance)
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

    return 0;
}