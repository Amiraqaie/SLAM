#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

int main()
{
    cv::Mat image1;
    cv::Mat image2;

    image1 = cv::imread("/home/havaie/Codes/SLAM/SLAM/images/1615.png");
    image2 = cv::imread("/home/havaie/Codes/SLAM/SLAM/images/171.png");
    
    // GFTT detector
    cv::Ptr<cv::GFTTDetector> gftt_detector_ = cv::GFTTDetector::create( 150, 0.01, 20);
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;
    // cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    // detect keypoints
    gftt_detector_->detect(image1, keypoints1);
    gftt_detector_->detect(image2, keypoints2);

    // compute descriptor
    extractor->compute(image1, keypoints1, descriptor1);
    extractor->compute(image2, keypoints2, descriptor2);

    // match descriptors
    vector<cv::DMatch> matches;
    vector<vector<cv::DMatch>> matches_knn;
    matcher->knnMatch(descriptor1, descriptor2, matches_knn, 2);

    // Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < matches_knn.size(); i++) {
        if (matches_knn[i][0].distance < ratio_thresh * matches_knn[i][1].distance) {
            matches.push_back(matches_knn[i][0]);
        }
    }

    // show results without filtering
    cv::Mat result;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, result);
    cv::imshow("good_mathcer", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}