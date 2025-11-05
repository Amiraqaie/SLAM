#include <utils.h>

int main()
{
    // load images
    Mat image1, image2;
    bool load_success = load_images(image1, image2);
    if (!load_success)
        return 1;

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(image1, kp1);
    vector<Point2f> pt1, pt2;
    for (auto &kp : kp1)
        pt1.push_back(kp.pt);

    // use Opencv optical flow to track feature
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(image1, image2, pt1, pt2, status, err);

    // show optical flow result
    Mat vis_cv;
    cv::cvtColor(image1, vis_cv, COLOR_GRAY2BGR);
    for (int i = 0; i < pt1.size(); i++)
    {
        if (status[i] == 1)
        {
            circle(vis_cv, pt2[i], 3, Scalar(0, 255, 0), -1);
            line(vis_cv, pt1[i], pt2[i], Scalar(0, 255, 0));
        }
    }
    imshow("optical flow by opencv", vis_cv);
    waitKey(0);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(image1, image2, kp1, kp2_single, success_single);

    // show optical flow result
    Mat vis_single;
    cv::cvtColor(image1, vis_single, COLOR_GRAY2BGR);
    for (int i = 0; i < kp1.size(); i++)
    {
        circle(vis_single, kp2_single[i].pt, 3, Scalar(0, 255, 0), -1);
        line(vis_single, kp1[i].pt, kp2_single[i].pt, Scalar(0, 255, 0));
    }
    imshow("optical flow single level", vis_single);
    waitKey(0);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(image1, image2, kp1, kp2_multi, success_multi, true);

    // show optical flow result
    Mat vis_multi;
    cv::cvtColor(image1, vis_multi, COLOR_GRAY2BGR);
    for (int i = 0; i < kp1.size(); i++)
    {
        circle(vis_multi, kp2_multi[i].pt, 3, Scalar(0, 255, 0), -1);
        line(vis_multi, kp1[i].pt, kp2_multi[i].pt, Scalar(0, 255, 0));
    }
    imshow("optical flow multi level", vis_multi);
    waitKey(0);
}
