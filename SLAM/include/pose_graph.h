#ifndef MYSLAM_POSE_GRAPH_OPTIMIZATION_H
#define MYSLAM_POSE_GRAPH_OPTIMIZATION_H

#include "common_include.h"
#include "frame.h"
#include "map.h"
#include "mappoint.h"
#include "loop_closing.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>

namespace myslam {
    class Backend;
}

namespace myslam {

class PoseGraphOptimization {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<PoseGraphOptimization> Ptr;
    
    PoseGraphOptimization();
    
    // Set map for pose graph construction
    void SetMap(Map::Ptr map) { map_ = map; }

    void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

    // Perform full pose graph optimization
    bool OptimizePoseGraph(const std::vector<LoopConstraint>& loop_constraints);
    
    // Get pose corrections for map point update
    std::unordered_map<unsigned long, SE3> GetPoseCorrections() const {
        return pose_corrections_;
    }
    
private:
    // Build pose graph from keyframes and constraints
    void BuildPoseGraph(const std::vector<LoopConstraint>& loop_constraints,
                       g2o::SparseOptimizer& optimizer);
    
    // Add keyframe vertices to graph
    void AddKeyframeVertices(g2o::SparseOptimizer& optimizer);
    
    // Add odometry edges between consecutive keyframes
    void AddOdometryEdges(g2o::SparseOptimizer& optimizer);
    
    // Add loop closure edges
    void AddLoopClosureEdges(const std::vector<LoopConstraint>& loop_constraints,
                            g2o::SparseOptimizer& optimizer);
    
    // Extract optimized poses and compute corrections
    void ExtractPoseCorrections(g2o::SparseOptimizer& optimizer);
    
    // Update map after pose graph optimization
    void UpdateMapAfterOptimization();
    
    // Correct map point positions based on pose corrections
    void CorrectMapPointPositions();
        
private:
    Map::Ptr map_;
    
    // Store original poses before optimization
    std::unordered_map<unsigned long, SE3> original_poses_;
    
    std::shared_ptr<Backend> backend_;

    // Store pose corrections (T_corrected = correction * T_original)
    std::unordered_map<unsigned long, SE3> pose_corrections_;
    
    // Optimization parameters
    int max_iterations_ = 100;
    bool verbose_ = true;
};

}  // namespace myslam

#endif  // MYSLAM_POSE_GRAPH_OPTIMIZATION_H
