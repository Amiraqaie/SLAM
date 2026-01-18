#pragma once
#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "backend.h"
#include "common_include.h"
#include "dataset.h"
#include "frontend.h"
#include "viewer_light.h"
#include "loop_closing.h"
#include "pose_graph.h"

namespace myslam {

/**
 * VO 对外接口
 */
class VisualOdometry {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualOdometry> Ptr;

    /// constructor with config file
    VisualOdometry(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

    /// Trigger pose graph optimization when enough loop closures are detected
    void TriggerPoseGraphOptimization();

private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;
    LoopClosing::Ptr loop_closing_ = nullptr;
    PoseGraphOptimization::Ptr pose_graph_optimizer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;

    // Loop closure parameters
    int loop_closure_frequency_ = 10;  // Check for PGO every N keyframes
    int keyframes_since_last_pgo_ = 0;
    int min_loop_constraints_for_pgo_ = 3;
};
}  // namespace myslam

#endif  // MYSLAM_VISUAL_ODOMETRY_H
