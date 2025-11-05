#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;

vector<Eigen::Vector4d> filterStatisticalOutliers(const vector<Eigen::Vector4d> &points);
void show_point_cloud(const vector<Eigen::Vector4d> &points, const string& window_name);

int main()
{
    double fx = 718.85, fy = 718.85, cx = 607.19, cy = 185.21;
    double baseline = 0.573;
    double max_depth = 60 * baseline;

    // Load images
    cv::Mat image_left = cv::imread("/home/amir/SLAM/review/ch5/images/left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image_right = cv::imread("/home/amir/SLAM/review/ch5/images/right.png", cv::IMREAD_GRAYSCALE);

    if(image_left.empty() || image_right.empty()) {
        cerr << "Error: Could not load images!" << endl;
        return -1;
    }

    // Calculate stereo disparity
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 96, 3, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);

    cv::Mat disparity;
    stereo->compute(image_left, image_right, disparity);
    disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);

    // Show disparity
    cv::imshow("Disparity", disparity);
    cv::waitKey(0);

    // Convert disparity to point cloud
    vector<Eigen::Vector4d> points;
    for (int v = 0; v < disparity.rows; v++) {
        for (int u = 0; u < disparity.cols; u++) {
            float disp = disparity.at<float>(v, u);
            if (disp > 0) {
                double depth = fx * baseline / disp;
                if (depth <= max_depth) {
                    points.emplace_back(
                        (u - cx) * depth / fx, 
                        (v - cy) * depth / fy, 
                        depth, 
                        1.0
                    );
                }
            }
        }
    }

    cout << "Original points: " << points.size() << endl;

    // Filter and show results
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Eigen::Vector4d> filtered_points = filterStatisticalOutliers(points);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> duration = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "Filtered points: " << filtered_points.size() << endl;
    cout << "Filtered time: " << duration.count() << endl;

    show_point_cloud(points, "Original Point Cloud");
    show_point_cloud(filtered_points, "Filtered Point Cloud");

    return 0;
}

void show_point_cloud(const vector<Eigen::Vector4d> &points, const string& window_name)
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

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (const auto &point : points) {
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

vector<Eigen::Vector4d> filterStatisticalOutliers(const vector<Eigen::Vector4d> &points)
{
    if (points.empty()) return {};

    // Convert to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(points.size());
    
    for (const auto &p : points) {
        cloud->push_back(pcl::PointXYZ(p[0], p[1], p[2]));
    }
    cloud->width = cloud->size();
    cloud->height = 1;

    // Apply statistical outlier removal
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);            // Number of neighbors to analyze
    sor.setStddevMulThresh(1.0); // Standard deviation multiplier
    sor.filter(*filtered);

    // Convert back to Eigen vectors
    vector<Eigen::Vector4d> result;
    result.reserve(filtered->size());
    for (const auto &p : *filtered) {
        result.emplace_back(p.x, p.y, p.z, 1.0);
    }

    return result;
}