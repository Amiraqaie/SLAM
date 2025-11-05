#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

    // read images
    Mat image_1 = imread("../1.png");
    Mat image_2 = imread("../2.png");

    // initialization
    std::vector<KeyPoint> keypoint_1, keypoint_2;
    Mat descriptor_1, descriptor_2;
    Ptr<FeatureDetector> detector = GFTTDetector::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // detect Oriented fast keypoints
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(image_1, keypoint_1);
    detector->detect(image_2, keypoint_2);

    // compute BRIEF descriptors
    descriptor->compute(image_1, keypoint_1, descriptor_1);
    descriptor->compute(image_2, keypoint_2, descriptor_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(image_1, keypoint_1, outimg1, Scalar::all(-1));
    imshow("ORB features", outimg1);

    // use Hamming distance to brute force matching
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptor_1, descriptor_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;


    auto min_max = minmax_element(matches.begin(), matches.end(),
    [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);


    // remove bad matchs
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptor_1.rows; i++) {
      if (matches[i].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
      }
    }

    // show results
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(image_1, keypoint_1, image_2, keypoint_2, matches, img_match);
    drawMatches(image_1, keypoint_1, image_2, keypoint_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);
}