#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

int main()
{
    cv::Mat image1;
    cv::Mat image2;

    image1 = cv::imread("/home/amir/SLAM/review/ch7/images/1.png");
    image2 = cv::imread("/home/amir/SLAM/review/ch7/images/2.png");

    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    // detect keypoints
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    // compute descriptor
    extractor->compute(image1, keypoints1, descriptor1);
    extractor->compute(image2, keypoints2, descriptor2);

    // match descriptors
    vector<cv::DMatch> matches;
    vector<vector<cv::DMatch>> matches_knn;
    matcher->match(descriptor1, descriptor2, matches);
    matcher->knnMatch(descriptor1, descriptor2, matches_knn, 2);

    // filter out outliers
    vector<cv::DMatch> good_matches;
    for (auto matches : matches_knn)
    {
        int first_distance = matches[0].distance;
        int second_distance = matches[1].distance;

        if (first_distance * (1 / 0.75) < second_distance)
            good_matches.push_back(matches[0]);
    }

    // show results without filtering
    cv::Mat result;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, result);
    cv::imshow("good_mathcer", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}