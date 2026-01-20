//
// Created by gaoxiang on 19-5-4.
//
#include "visual_odometry.h"
#include <chrono>
#include "config.h"

namespace myslam {

VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    dataset_ =
        Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    // backend_->Stop();
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);
    // viewer_ = nullptr;
    loop_closing_ = LoopClosing::Ptr(new LoopClosing);
    pose_graph_optimizer_ = PoseGraphOptimization::Ptr(new PoseGraphOptimization);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    backend_->SetPoseGraph(pose_graph_optimizer_);

    loop_closing_->SetMap(map_);
    loop_closing_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));
    
    pose_graph_optimizer_->SetMap(map_);
    pose_graph_optimizer_->SetBackend(backend_);

    viewer_->SetMap(map_);

    return true;
}

void VisualOdometry::Run() {
    while (1) {
        LOG(INFO) << "VO is running";
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    loop_closing_->Stop();
    while (1)
    {
        usleep(300);
    }
    
    LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    
    // Add keyframe to loop closing if it's a keyframe
    if (new_frame->is_keyframe_) {
        loop_closing_->AddKeyFrame(new_frame);
        keyframes_since_last_pgo_++;
        
        // Trigger pose graph optimization periodically
        if (keyframes_since_last_pgo_ >= loop_closure_frequency_) {
            TriggerPoseGraphOptimization();
            keyframes_since_last_pgo_ = 0;
        }
    }
    
    return success;
}

void VisualOdometry::TriggerPoseGraphOptimization() {
    auto loop_constraints = loop_closing_->GetLoopConstraints();
    
    if (loop_constraints.size() >= min_loop_constraints_for_pgo_) {
        LOG(INFO) << "Triggering pose graph optimization with " 
                 << loop_constraints.size() << " loop constraints";
        
        // Perform pose graph optimization
        bool success = pose_graph_optimizer_->OptimizePoseGraph(loop_constraints);
        
        if (success) {
            // Update viewer after optimization
            if (viewer_) {
                viewer_->UpdateMap();
            }
            
            LOG(INFO) << "Pose graph optimization completed successfully";
        } else {
            LOG(WARNING) << "Pose graph optimization failed";
        }
    } else {
        LOG(INFO) << "Not enough loop constraints (" << loop_constraints.size() 
                 << "/" << min_loop_constraints_for_pgo_ << ") for pose graph optimization";
    }
}

}  // namespace myslam
