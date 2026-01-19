#ifndef MYSLAM_LOOP_CLOSING_H
#define MYSLAM_LOOP_CLOSING_H

#include "common_include.h"
#include "frame.h"
#include "map.h"
#include "camera.h"
#include "feature.h"
#include "mappoint.h"
#include "g2o_types.h"
#include <queue>
#include <DBoW3/DBoW3.h>

#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>

namespace myslam {

struct LoopConstraint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    unsigned long keyframe1_id;  // Current keyframe
    unsigned long keyframe2_id;  // Loop keyframe
    SE3 relative_pose;           // T_12 (from frame1 to frame2)
    Mat66 information;           // Information matrix
    double confidence;           // Loop confidence score
    
    LoopConstraint(unsigned long kf1, unsigned long kf2, const SE3& pose, 
                   const Mat66& info, double conf)
        : keyframe1_id(kf1), keyframe2_id(kf2), relative_pose(pose), 
          information(info), confidence(conf) {}
};

/**
 * Loop Closure Detection and Pose Graph Optimization
 * Detects loops using visual bag-of-words and optimizes pose graph
 */
class LoopClosing {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<LoopClosing> Ptr;
    
    
    LoopClosing();
    ~LoopClosing();
    
    // Set components
    void SetMap(Map::Ptr map) { map_ = map; }
    void SetCameras(Camera::Ptr camera_left, Camera::Ptr camera_right) 
    { 
        camera_left_ = camera_left;
        camera_right_ = camera_right;
    }
    
    // Main interface - process keyframe for loop detection
    void AddKeyFrame(Frame::Ptr keyframe);
    
    // Stop the loop detection thread
    void Stop();
    
    // Get loop constraints for pose graph optimization
    std::vector<LoopConstraint> GetLoopConstraints();
    
private:
    // Main loop detection thread
    void LoopDetectionThread();
    
    // Detect loop candidates using visual similarity
    std::vector<Frame::Ptr> DetectLoopCandidates(Frame::Ptr current_frame);
    
    // Triangulate the extracted orb feature in left and right image of frame
    std::map<int, Vec3> TriangulateFeatures(Frame::Ptr frame);

    // Verify loop closure through geometric verification
    bool VerifyLoopClosure(Frame::Ptr current_frame, Frame::Ptr candidate_frame,
                          SE3& relative_pose, Mat66& information);
    
    // Extract ORB features and descriptors
    void ExtractORBDescriptors(Frame::Ptr frame);
    
    // Compute visual similarity between frames using bag-of-words
    double ComputeVisualSimilarity(Frame::Ptr frame1, Frame::Ptr frame2);
    
    // Perform feature matching between two frames
    std::vector<cv::DMatch> MatchFeatures(Frame::Ptr frame1, Frame::Ptr frame2);
    std::vector<cv::DMatch> MatchFeatures(cv::Mat descriptors1, cv::Mat descriptors2);
    
    // Estimate relative pose using matched features
    bool EstimateRelativePose(Frame::Ptr frame1, Frame::Ptr frame2,
                             const std::vector<cv::DMatch>& matches,
                             SE3& relative_pose, Mat66& information);
    
    int EstimateCandidatePose(std::vector<cv::Point3d> &objectPoints, std::vector<cv::Point2d> &imagePoints, SE3& relative_pose);
    
private:
    Map::Ptr map_;
    Camera::Ptr camera_left_;
    Camera::Ptr camera_right_;

    // dbow3 vocab
    DBoW3::Vocabulary vocab_;
    DBoW3::Database db_;

    // Thread management
    std::thread loop_thread_;
    std::atomic<bool> loop_running_;
    std::mutex keyframe_mutex_;
    std::condition_variable keyframe_updated_;
    std::queue<Frame::Ptr> keyframe_queue_;
    
    // ORB feature extractor
    cv::Ptr<cv::GFTTDetector> gftt_detector_;
    cv::Ptr<cv::DescriptorExtractor> orb_extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    
    // Loop constraints storage
    std::vector<LoopConstraint> loop_constraints_;
    std::mutex constraints_mutex_;
    
    // Database for visual place recognition
    std::vector<Frame::Ptr> keyframe_database_;
    
    // Parameters
    double visual_similarity_threshold_ = 0.7;
    int min_loop_interval_ = 10;  // Minimum frames between loops
    int ransac_iterations_ = 1000;
    double ransac_threshold_ = 0.02;  // 2cm
    double minimum_match_ratio_ = 0.4;
    int min_inliers_ = 20;
    int dbow_query_size_ = 10.0;
    bool show_result_ = true;
};

}  // namespace myslam

#endif  // MYSLAM_LOOP_CLOSING_H