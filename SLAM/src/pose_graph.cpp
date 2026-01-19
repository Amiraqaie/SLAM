#include "pose_graph.h"
#include "feature.h"

namespace myslam {

PoseGraphOptimization::PoseGraphOptimization() {
    max_iterations_ = 20;
    verbose_ = true;
    
    LOG(INFO) << "Pose Graph Optimization initialized";
}

bool PoseGraphOptimization::OptimizePoseGraph(const std::vector<LoopConstraint>& loop_constraints) {
    if (loop_constraints.empty()) {
        LOG(WARNING) << "No loop constraints for pose graph optimization";
        return false;
    }
    
    LOG(INFO) << "Starting pose graph optimization with " << loop_constraints.size() << " loop constraints";
    
    // Create g2o optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto linearSolver = std::make_unique<LinearSolverType>();
    auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(verbose_);
    
    // Store original poses
    auto keyframes = map_->GetAllKeyFrames();
    for (const auto& kf_pair : keyframes) {
        original_poses_[kf_pair.first] = kf_pair.second->Pose();
    }
    
    // Build pose graph
    BuildPoseGraph(loop_constraints, optimizer);
    
    // Perform optimization
    optimizer.initializeOptimization();
    int iterations = optimizer.optimize(max_iterations_);
    
    if (iterations > 0) {
        LOG(INFO) << "Pose graph optimization completed in " << iterations << " iterations";
        
        // Extract pose corrections
        ExtractPoseCorrections(optimizer);
        
        // Update map with corrected poses and map points
        UpdateMapAfterOptimization();
        
        return true;
    } else {
        LOG(ERROR) << "Pose graph optimization failed";
        return false;
    }
}

void PoseGraphOptimization::BuildPoseGraph(const std::vector<LoopConstraint>& loop_constraints,
                                          g2o::SparseOptimizer& optimizer) {
    // Add keyframe vertices
    AddKeyframeVertices(optimizer);
    
    // Add odometry edges
    AddOdometryEdges(optimizer);
    
    // Add loop closure edges
    AddLoopClosureEdges(loop_constraints, optimizer);
}

void PoseGraphOptimization::AddKeyframeVertices(g2o::SparseOptimizer& optimizer) {
    auto keyframes = map_->GetAllKeyFrames();
    
    for (const auto& kf_pair : keyframes) {
        auto keyframe = kf_pair.second;
        
        // Create SE3 vertex
        g2o::VertexSE3* vertex = new g2o::VertexSE3();
        vertex->setId(keyframe->keyframe_id_);
        
        // Convert Sophus SE3 to g2o SE3
        Eigen::Isometry3d pose_iso = Eigen::Isometry3d::Identity();
        pose_iso.linear() = keyframe->Pose().rotationMatrix();
        pose_iso.translation() = keyframe->Pose().translation();
        vertex->setEstimate(pose_iso);
        
        // Fix the first keyframe
        if (keyframe->keyframe_id_ == 0) {
            vertex->setFixed(true);
        }
        
        optimizer.addVertex(vertex);
    }
    
    LOG(INFO) << "Added " << keyframes.size() << " keyframe vertices to pose graph";
}

void PoseGraphOptimization::AddOdometryEdges(g2o::SparseOptimizer& optimizer) {
    auto keyframes = map_->GetAllKeyFrames();
    
    // Sort keyframes by ID
    std::vector<std::pair<unsigned long, Frame::Ptr>> sorted_keyframes(keyframes.begin(), keyframes.end());
    std::sort(sorted_keyframes.begin(), sorted_keyframes.end());
    
    int edge_count = 0;
    for (size_t i = 1; i < sorted_keyframes.size(); ++i) {
        auto kf1 = sorted_keyframes[i-1].second;
        auto kf2 = sorted_keyframes[i].second;
        
        // Compute relative pose between consecutive keyframes
        // T_kf1_kf2
        SE3 relative_pose = kf1->Pose().inverse() * kf2->Pose();
        
        // Create SE3 edge
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->setId(edge_count++);
        edge->setVertex(0, optimizer.vertex(kf1->keyframe_id_));
        edge->setVertex(1, optimizer.vertex(kf2->keyframe_id_));
        
        // Convert to g2o format
        Eigen::Isometry3d relative_iso = Eigen::Isometry3d::Identity();
        relative_iso.linear() = relative_pose.rotationMatrix();
        relative_iso.translation() = relative_pose.translation();
        edge->setMeasurement(relative_iso);
        
        // Set information matrix (odometry is usually quite accurate)
        Mat66 information = Mat66::Identity() * 1000.0;  // High confidence
        edge->setInformation(information);
        
        optimizer.addEdge(edge);

        // Print odometry edge info
        const auto& t = relative_iso.translation();
        Eigen::Quaterniond q(relative_iso.rotation());

        // LOG(INFO) << "[OdomEdge] "
        //         << "KF " << kf1->keyframe_id_
        //         << " -> " << kf2->keyframe_id_
        //         << " | t = [" << t.transpose() << "]"
        //         << " | q = [" << q.w() << ", "
        //                         << q.x() << ", "
        //                         << q.y() << ", "
        //                         << q.z() << "]";

        // LOG(INFO) << "[OdomEdge] Information:\n" << information;
    }
    
    LOG(INFO) << "Added " << edge_count << " odometry edges to pose graph";
}

void PoseGraphOptimization::AddLoopClosureEdges(const std::vector<LoopConstraint>& loop_constraints,
                                               g2o::SparseOptimizer& optimizer) {
    int edge_count = 10000;  // Start from high ID to avoid conflicts
    
    for (const auto& constraint : loop_constraints) {
        // Create SE3 edge for loop closure
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->setId(edge_count++);
        edge->setVertex(0, optimizer.vertex(constraint.keyframe1_id));
        edge->setVertex(1, optimizer.vertex(constraint.keyframe2_id));
        
        // Convert relative pose to g2o format
        // relative_iso should be T_kf1_kf2
        Eigen::Isometry3d relative_iso = Eigen::Isometry3d::Identity();
        relative_iso.linear() = constraint.relative_pose.rotationMatrix();
        relative_iso.translation() = constraint.relative_pose.translation();
        edge->setMeasurement(relative_iso);
        
        // Set information matrix
        edge->setInformation(constraint.information);
        
        // Add robust kernel for loop closures
        g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber();
        robust_kernel->setDelta(1.0);
        edge->setRobustKernel(robust_kernel);
        
        optimizer.addEdge(edge);
        
        LOG(INFO) << "Added loop closure edge between keyframes " 
                 << constraint.keyframe1_id << " and " << constraint.keyframe2_id;

        // Print loopclosure edge info
        const auto& t = relative_iso.translation();
        Eigen::Quaterniond q(relative_iso.rotation());

        LOG(INFO) << "[LOOP CLOSURE EDGE] "
                << "KF " << constraint.keyframe1_id
                << " -> " << constraint.keyframe2_id
                << " | t = [" << t.transpose() << "]"
                << " | q = [" << q.w() << ", "
                                << q.x() << ", "
                                << q.y() << ", "
                                << q.z() << "]";

        LOG(INFO) << "[LOOP CLOSURE] Information:\n" << constraint.information;
    }
}

void PoseGraphOptimization::ExtractPoseCorrections(g2o::SparseOptimizer& optimizer) {
    pose_corrections_.clear();
    
    auto keyframes = map_->GetAllKeyFrames();
    
    for (const auto& kf_pair : keyframes) {
        auto keyframe = kf_pair.second;
        unsigned long kf_id = keyframe->keyframe_id_;
        
        // Get optimized vertex
        g2o::VertexSE3* vertex = static_cast<g2o::VertexSE3*>(optimizer.vertex(kf_id));
        if (!vertex) continue;
        
        // Extract optimized pose
        Eigen::Isometry3d optimized_iso = vertex->estimate();
        SE3 optimized_pose(optimized_iso.linear(), optimized_iso.translation());
        
        // Compute correction: T_corrected = T_correction * T_original
        // Therefore: T_correction = T_corrected * T_original^(-1)
        // T_original = T_w_c (old)
        // T_crrected = T_w_c (new)
        // T_correction = look like non sense
        // should be T_correction = T_original(-1) * T_crrected = T_c(old)_c(new)
        // or T_correction = T_crrected(-1) * T_original = T_c(new)_c(old) this is better
        
        // SE3 correction = optimized_pose * original_poses_[kf_id].inverse();
        SE3 correction = optimized_pose.inverse() * original_poses_[kf_id];
        pose_corrections_[kf_id] = correction;
        
        LOG(INFO) << "Keyframe " << kf_id << " correction: " 
                 << correction.translation().transpose() << " | "
                 << correction.so3().log().transpose();
    }
}

void PoseGraphOptimization::UpdateMapAfterOptimization() {
    // Update keyframe poses
    auto keyframes = map_->GetAllKeyFrames();
    for (const auto& kf_pair : keyframes) {
        auto keyframe = kf_pair.second;
        unsigned long kf_id = keyframe->keyframe_id_;
        
        if (pose_corrections_.find(kf_id) != pose_corrections_.end()) {
            // T_w_c(new) = T_w_c (old)  *  T_c(new)_c(old).inverse()
            SE3 corrected_pose = pose_corrections_[kf_id] * original_poses_[kf_id].inverse();
            keyframe->SetPose(corrected_pose);
        }
    }
    
    // Correct map point positions
    /// TODO : after the pose graph optimization the backend will adjust last 7 poses again to the before of pose graph optimization
    CorrectMapPointPositions();
    
    LOG(INFO) << "Map updated after pose graph optimization";
}

void PoseGraphOptimization::CorrectMapPointPositions() {
    auto landmarks = map_->GetAllMapPoints();
    int corrected_points = 0;
    int64 all_points = landmarks.size();
    int64 no_observation_points = 0;
    
    for (const auto& lm_pair : landmarks) {
        auto landmark = lm_pair.second;
        auto observations = landmark->GetObs();
        
        if (observations.empty()) 
        {
            no_observation_points++;
            continue;
        }
        // Find the keyframe that observed this landmark with the smallest correction
        // (to minimize correction error propagation)
        SE3 best_correction;
        bool found_correction = false;
        double min_correction_magnitude = std::numeric_limits<double>::max();
        
        for (const auto& obs_weak : observations) {
            auto obs = obs_weak.lock();
            if (!obs) continue;
            
            auto frame = obs->frame_.lock();
            if (!frame || !frame->is_keyframe_) continue;
            
            unsigned long kf_id = frame->keyframe_id_;
            if (pose_corrections_.find(kf_id) != pose_corrections_.end()) {
                SE3 correction = pose_corrections_[kf_id];
                double magnitude = correction.translation().norm() + correction.so3().log().norm();
                
                if (magnitude < min_correction_magnitude) {
                    min_correction_magnitude = magnitude;
                    best_correction = correction;
                    found_correction = true;
                }
            }
        }
        
        // Apply correction to map point position
        if (found_correction) {

            // original_pos = coordinate of landmark in World coordinate
            Vec3 original_pos = landmark->Pos();

            // original pose in old camera coordinate
            Vec3 corrected_pos =  best_correction * original_pos;
            landmark->SetPos(corrected_pos);
            corrected_points++;
            
            LOG_EVERY_N(INFO, 100) << "Corrected landmark " << landmark->id_ 
                                  << " position by " << (corrected_pos - original_pos).norm() << " meters";
        }
    }
    
    LOG(INFO) << "Corrected positions of " << corrected_points << " map points out of " << all_points - no_observation_points << " available map points!!!";
}

}  // namespace myslam
