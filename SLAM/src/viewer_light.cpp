#include "viewer_light.h"
#include "frame.h"
#include "map.h"
#include "feature.h"
#include "loop_closing.h"
#include "algorithm.h"
#include <opencv2/opencv.hpp>
#include <deque>

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
    // active_landmarks_ = map_->GetAllMapPoints();
    landmarks_= map_->GetActiveMapPoints();
    keyframes_ = map_->GetAllKeyFrames();
    loop_constraints_ = map_->GetLoopConstraints();
    map_updated_ = true;
}

void Viewer::ThreadLoop() {
    cv::namedWindow("MySLAM Track", cv::WINDOW_FULLSCREEN);
    // cv::namedWindow("MySLAM Frame", cv::WINDOW_NORMAL);

    std::deque<cv::Point2f> trajectory;

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

        // 2. Draw trajectory (dynamic scale)
        cv::Mat traj = cv::Mat::zeros(TRAJ_SIZE, TRAJ_SIZE, CV_8UC3);

        trajectory.clear();
        for (auto frame : keyframes)
        {
            SE3 Twc = frame.second->Pose().inverse();
            Eigen::Vector3d t = Twc.translation();
            cv::Point2f pt(t.x(), t.z()); // store in real-world coords
            trajectory.push_back(pt);
        }

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
        float margin = 50; // pixels
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
            cv::line(traj, p1, p2, cv::Scalar(255, 255, 0), 1);
        }

        // Draw all landmarks
        int index = 0;
        for (auto &landmark : landmarks) {
            index++;
            if (index % 1000 == 0)
                continue;
            auto pose = landmark.second->Pos();
            cv::Point2f landmark_2d = cv::Point2f(
                margin + (pose[0] - min_x) * scale,
                TRAJ_SIZE - margin - (pose[2] - min_z) * scale
            );
            cv::circle(traj, landmark_2d, 1, cv::Scalar(0, 255, 0), -1);
        }

        // Draw current position
        SE3 Twc = current_frame->Pose().inverse();
        Eigen::Vector3d t = Twc.translation();
        cv::Point2f pt(t.x(), t.z()); // store in real-world coords
        cv::Point2f cur_pt(
            margin + (pt.x - min_x) * scale,
            TRAJ_SIZE - margin - (pt.y - min_z) * scale
        );
        cv::circle(traj, cur_pt, 3, cv::Scalar(0, 0, 255), -1);

        
        // Draw Loop Constraints
        if (!constraints.empty()){
            for (auto constraint : constraints)
            {
                int keyframe1_id = constraint.keyframe1_id;
                int keyframe2_id = constraint.keyframe2_id;
                
                Frame::Ptr kf1 = map_->GetByKeyFrameId(keyframe1_id);
                Frame::Ptr kf2 = map_->GetByKeyFrameId(keyframe2_id);
                
                SE3 Twc1 = kf1->Pose().inverse();
                SE3 Twc2 = kf2->Pose().inverse();
                
                Eigen::Vector3d t1 = Twc1.translation();
                Eigen::Vector3d t2 = Twc2.translation();
                
                cv::Point2f p1 = cv::Point2f(
                    margin + (t1.x() - min_x) * scale,
                    TRAJ_SIZE - margin - (t1.z() - min_z) * scale
                );
                cv::Point2f p2 = cv::Point2f(
                    margin + (t2.x() - min_x) * scale,
                    TRAJ_SIZE - margin - (t2.z() - min_z) * scale
                );
                cv::line(traj, p1, p2, cv::Scalar(0, 0, 255), 1);   
            }
        }
        cv::imshow("MySLAM Track", traj);
        cv::waitKey(5);
    }
}

}  // namespace myslam
