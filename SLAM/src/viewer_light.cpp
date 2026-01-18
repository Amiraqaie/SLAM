#include "viewer_light.h"
#include "frame.h"
#include "map.h"
#include "feature.h"

#include <opencv2/opencv.hpp>
#include <deque>

namespace myslam {

static constexpr int TRAJ_SIZE = 800;
static constexpr float SCALE = 20.0f;   // meters → pixels

Viewer::Viewer() {
    viewer_thread_ = std::thread(&Viewer::ThreadLoop, this);
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
    active_keyframes_ = map_->GetAllKeyFrames();
    active_landmarks_ = map_->GetAllMapPoints();
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    cv::namedWindow("MySLAM Track", cv::WINDOW_NORMAL);
    cv::namedWindow("MySLAM Frame", cv::WINDOW_NORMAL);

    std::deque<cv::Point2f> trajectory;

    while (viewer_running_) {
        Frame::Ptr frame;
        {
            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            frame = current_frame_;
        }

        if (!frame) {
            cv::waitKey(5);
            continue;
        }

        // =========================
        // 1. Draw current image
        // =========================
        cv::Mat img;
        cv::cvtColor(frame->left_img_, img, cv::COLOR_GRAY2BGR);

        for (auto& feat : frame->features_left_) {
            std::unique_lock<std::mutex> lck(frame->feature_mutex_);
            if (feat->map_point_.lock()) {
                cv::circle(img, feat->position_.pt, 2,
                           cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("MySLAM Frame", img);

        // =========================
        // 2. Draw trajectory (dynamic scale)
        // =========================
        cv::Mat traj = cv::Mat::zeros(TRAJ_SIZE, TRAJ_SIZE, CV_8UC3);

        SE3 Twc = frame->Pose().inverse();
        Eigen::Vector3d t = Twc.translation();

        // Add current point to trajectory
        cv::Point2f pt(t.x(), t.z()); // store in real-world coords
        trajectory.push_back(pt);

        // Compute bounding box of all trajectory points
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();

        for (auto& p : trajectory) {
            if (p.x < min_x) min_x = p.x;
            if (p.x > max_x) max_x = p.x;
            if (p.y < min_z) min_z = p.y;
            if (p.y > max_z) max_z = p.y;
        }

        // Compute dynamic scale to fit window
        float margin = 20; // pixels
        float scale_x = (TRAJ_SIZE - 2*margin) / std::max(1e-5f, max_x - min_x);
        float scale_z = (TRAJ_SIZE - 2*margin) / std::max(1e-5f, max_z - min_z);
        float scale = std::min(scale_x, scale_z);

        // Draw all points
        for (size_t i = 1; i < trajectory.size(); ++i) {
            cv::Point2f p1 = cv::Point2f(
                margin + (trajectory[i-1].x - min_x) * scale,
                TRAJ_SIZE - margin - (trajectory[i-1].y - min_z) * scale
            );
            cv::Point2f p2 = cv::Point2f(
                margin + (trajectory[i].x - min_x) * scale,
                TRAJ_SIZE - margin - (trajectory[i].y - min_z) * scale
            );
            cv::line(traj, p1, p2, cv::Scalar(0, 255, 0), 2);
        }

        // Draw all landmarks
        // for (auto &landmark : active_landmarks_) {
        //     auto pose = landmark.second->Pos();
        //     cv::Point2f landmark_2d = cv::Point2f(
        //         margin + (pose[0] - min_x) * scale,
        //         TRAJ_SIZE - margin - (pose[2] - min_z) * scale
        //     );
        //     cv::circle(traj, landmark_2d, 1, cv::Scalar(0, 255, 0), -1);
        // }

        // Draw current position
        cv::Point2f cur_pt(
            margin + (pt.x - min_x) * scale,
            TRAJ_SIZE - margin - (pt.y - min_z) * scale
        );
        cv::circle(traj, cur_pt, 3, cv::Scalar(0, 0, 255), -1);

        cv::imshow("MySLAM Track", traj);


        cv::waitKey(5);
    }
}

}  // namespace myslam
