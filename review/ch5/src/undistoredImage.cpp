#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;;

int main()
{

    double k1 = -0.2834, k2 = 0.0739, p1 = 0.00019, p2 = 1.76e-5;
    // double k1 = -0.0, k2 = 0.0, p1 = 0.0, p2 = 0;
    double fx = 458.65, fy = 457.29, cx = 367.21, cy = 248.37;

    cv::Mat image = cv::imread("/home/amir/SLAM/review/ch5/images/distorted.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_undistorted = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (size_t u = 0; u < image.cols; u++)
    {
        for(size_t v = 0; v < image.rows; v++)
        {
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);

            double x_distorted = x * (1 + k1 * r * r + k2 * pow(r, 4)) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * pow(r, 4)) + 2 * p2 * x * y + p1 * (r * r + 2 * y * y);

            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < image.cols && v_distorted < image.rows)
                image_undistorted.at<int8_t>(v, u) = image.at<int8_t>((int) v_distorted, (int) u_distorted);
        }
    }

    cv::imshow("distorted image", image);
    cv::imshow("undistorted image", image_undistorted);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}