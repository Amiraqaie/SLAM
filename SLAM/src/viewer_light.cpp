#include "viewer_light.h"
#include "frame.h"
#include "map.h"
#include "feature.h"
#include "loop_closing.h"
#include "algorithm.h"
#include <opencv2/opencv.hpp>
#include <deque>
#include <numeric>                 // >>> ADDED
#include <algorithm>               // >>> ADDED

namespace myslam {

static constexpr int TRAJ_SIZE = 800;
static constexpr float SCALE = 20.0f;   // meters → pixels

Viewer::Viewer() {
    viewer_thread_ = std::thread(&Viewer::ThreadLoop, this);
    viewer_running_ = true;
}

void Viewer::Close() {
    viewer_running_ = false;
    viewer_thread_.join();
}

void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lock(viewer_data_mutex_);
    current_frame_ = current_frame;
}

void Viewer::UpdateMap() {
    std::unique_lock<std::mutex> lock(viewer_data_mutex_);
    landmarks_= map_->GetActiveMapPoints();
    keyframes_ = map_->GetAllKeyFrames();
    loop_constraints_ = map_->GetLoopConstraints();
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    cv::namedWindow("MySLAM Track", cv::WINDOW_FULLSCREEN);
    cv::namedWindow("Pose Translation Error", cv::WINDOW_NORMAL);   // >>> ADDED

    std::deque<cv::Point2f> trajectory;
    std::deque<cv::Point2f> gt_trajectory;

    std::vector<float> translation_errors;   // >>> ADDED

    float max_plot_error = 20.0f;  // >>> ADDED: fixed max Y-axis range (meters)

    while (viewer_running_) {
        Frame::Ptr current_frame;
        Map::LandmarksType landmarks;
        std::map<unsigned long, Frame::Ptr> keyframes;
        std::vector<LoopConstraint> constraints;
        {
            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            keyframes = std::map<unsigned long, Frame::Ptr>(keyframes_.begin(), keyframes_.end());
            current_frame = current_frame_;
            landmarks = landmarks_;
            constraints = loop_constraints_;
        }

        if (!current_frame) {
            cv::waitKey(5);
            continue;
        }

        cv::Mat traj = cv::Mat::zeros(TRAJ_SIZE, TRAJ_SIZE, CV_8UC3);
        trajectory.clear();
        gt_trajectory.clear();
        translation_errors.clear();           // >>> ADDED

        for (auto frame : keyframes)
        {
            SE3 Twc = frame.second->Pose().inverse();
            Eigen::Vector3d t = Twc.translation();
            trajectory.emplace_back(t.x(), t.z());

            SE3 Gt_Twc = frame.second->GtPose().inverse();
            Eigen::Vector3d gt_t = Gt_Twc.translation();
            gt_trajectory.emplace_back(gt_t.x(), gt_t.z());

            // === Pose translation error ===
            float err = (t - gt_t).norm();    // >>> ADDED
            translation_errors.push_back(err); // >>> ADDED
        }

        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();

        for (auto& p : trajectory) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_z = std::min(min_z, p.y);
            max_z = std::max(max_z, p.y);
        }

        for (auto& p : gt_trajectory) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_z = std::min(min_z, p.y);
            max_z = std::max(max_z, p.y);
        }

        float margin = 50;
        float scale_x = (TRAJ_SIZE - 2*margin) / std::max(1e-5f, max_x - min_x);
        float scale_z = (TRAJ_SIZE - 2*margin) / std::max(1e-5f, max_z - min_z);
        float scale = std::min(scale_x, scale_z);

        for (size_t i = 1; i < trajectory.size(); ++i) {
            cv::Point2f p1(
                margin + (trajectory[i-1].x - min_x) * scale,
                TRAJ_SIZE - margin - (trajectory[i-1].y - min_z) * scale
            );
            cv::Point2f p2(
                margin + (trajectory[i].x - min_x) * scale,
                TRAJ_SIZE - margin - (trajectory[i].y - min_z) * scale
            );
            cv::line(traj, p1, p2, cv::Scalar(255,255,0), 1);
        }

        for (size_t i = 1; i < gt_trajectory.size(); ++i) {
            cv::Point2f p1(
                margin + (gt_trajectory[i-1].x - min_x) * scale,
                TRAJ_SIZE - margin - (gt_trajectory[i-1].y - min_z) * scale
            );
            cv::Point2f p2(
                margin + (gt_trajectory[i].x - min_x) * scale,
                TRAJ_SIZE - margin - (gt_trajectory[i].y - min_z) * scale
            );
            cv::line(traj, p1, p2, cv::Scalar(0,255,255), 1);
        }

        // === Pose Error Plot with fixed range ===
        if (!translation_errors.empty()) {                      // >>> ADDED
            cv::Mat err_plot(300, TRAJ_SIZE, CV_8UC3, cv::Scalar(0,0,0)); // >>> ADDED

            float max_err = max_plot_error;  // >>> MODIFIED: fixed Y-axis range

            for (size_t i = 1; i < translation_errors.size(); ++i) {
                cv::Point p1(
                    (i-1) * TRAJ_SIZE / translation_errors.size(),
                    280 - (translation_errors[i-1] / max_err) * 260
                );
                cv::Point p2(
                    i * TRAJ_SIZE / translation_errors.size(),
                    280 - (translation_errors[i] / max_err) * 260
                );
                cv::line(err_plot, p1, p2, cv::Scalar(0,255,0), 2);
            }
            cv::imshow("Pose Translation Error", err_plot);   // >>> ADDED
        }

        cv::imshow("MySLAM Track", traj);
        cv::waitKey(5);
    }
}

}  // namespace myslam
