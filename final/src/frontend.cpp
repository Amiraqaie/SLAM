#include "algorithm.h"
#include "frontend.h"
#include "backend.h"
#include "config.h"
#include "feature.h"
#include "opencv2/opencv.hpp"
#include "viewer.h"
#include "map.h"
#include "g2o_types.h"


Frontend::Frontend() {
    // GFTT detector
    gftt_detector_ = cv::GFTTDetector::create(
        Config::Get<int>("num_features"),
        0.01,       // qualityLevel = 0.01
        20          // minDistance = 20
    );
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();        // Track is called in both good and bad states
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    if (last_frame_)
    {
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = EstimateCurrentPose();

    if (tracking_inliers_ >= num_features_tracking_) {
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ >= num_features_bad_) {
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        status_ = FrontendStatus::LOST;
    }

    InsertKeyFrame();
    relative_motion_ = last_frame_->Pose().inverse() * current_frame_->Pose();
    if (viewer_)
    {
        viewer_->AddCurrentFrame(current_frame_);
    }
    return true;
}

bool Frontend::InsertKeyFrame() {
    if (tracking_inliers_ >= num_features_for_keyframe_) {
        return false;
    }
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    LOG(INFO) << "Insert a new keyframe " << current_frame_->keyframe_id_
              << ", total keyframes: " << map_->GetAllKeyFrames().size();
    SetObservationsForKeyFrame();
    DetectFeatures();
    FindFeaturesInRight();
    TriangulateNewPoints();
    if (backend_) {
        backend_->UpdateMap();
    }
    if (viewer_) {
        viewer_->UpdateMap();
    }
    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto& feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->AddObservation(feat);
        }
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
    Sophus::SE3d current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            std::vector<Eigen::Vector3d> points{
                camera_left_->pixel2camera(
                    Eigen::Vector2d(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Eigen::Vector2d(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Eigen::Vector3d pworld = Eigen::Vector3d::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();
                pworld = current_pose_Twc * pworld;
                new_map_point->SetPos(pworld);
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}