#include "algorithm.h"
#include "config.h"
#include "loop_closing.h"

namespace myslam {

LoopClosing::LoopClosing() {
    // Initialize ORB feature extractor
    orb_extractor_ = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, 
                                     cv::ORB::HARRIS_SCORE, 31, 20);
    
    // Initialize feature matcher
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    
    // Load parameters from config
    visual_similarity_threshold_ = Config::Get<double>("loop.visual_similarity_threshold");
    min_loop_interval_ = Config::Get<int>("loop.min_loop_interval");
    ransac_iterations_ = Config::Get<int>("loop.ransac_iterations");
    ransac_threshold_ = Config::Get<double>("loop.ransac_threshold");
    min_inliers_ = Config::Get<int>("loop.min_inliers");
    
    // Start loop detection thread
    loop_running_.store(true);
    loop_thread_ = std::thread(&LoopClosing::LoopDetectionThread, this);
    
    LOG(INFO) << "Loop closure detection initialized";
}

LoopClosing::~LoopClosing() {
    Stop();
}

void LoopClosing::Stop() {
    loop_running_.store(false);
    keyframe_updated_.notify_all();
    if (loop_thread_.joinable()) {
        loop_thread_.join();
    }
    LOG(INFO) << "Loop closure detection stopped";
}

void LoopClosing::AddKeyFrame(Frame::Ptr keyframe) {
    std::unique_lock<std::mutex> lock(keyframe_mutex_);
    keyframe_queue_.push(keyframe);
    keyframe_updated_.notify_one();
}

std::vector<LoopConstraint> LoopClosing::GetLoopConstraints() {
    std::unique_lock<std::mutex> lock(constraints_mutex_);
    return loop_constraints_;
}

void LoopClosing::LoopDetectionThread() {
    while (loop_running_.load()) {
        Frame::Ptr current_keyframe = nullptr;
        
        {
            std::unique_lock<std::mutex> lock(keyframe_mutex_);
            keyframe_updated_.wait(lock, [this] { return !keyframe_queue_.empty() || !loop_running_.load(); });
            
            if (!loop_running_.load()) break;
            
            if (!keyframe_queue_.empty()) {
                current_keyframe = keyframe_queue_.front();
                keyframe_queue_.pop();
            }
        }
        
        if (!current_keyframe) continue;
        
        // Extract ORB features for the current keyframe
        ExtractORBFeatures(current_keyframe);
        
        // Add to database
        keyframe_database_.push_back(current_keyframe);
        
        // Skip loop detection for early keyframes
        if (keyframe_database_.size() < min_loop_interval_) continue;
        
        // Detect loop candidates
        auto candidates = DetectLoopCandidates(current_keyframe);
        
        // Verify loop closures
        for (auto candidate : candidates) {
            /// TODO : releative position should be calculated
            // relative_pose = T_kf1_kf2
            SE3 relative_pose;
            Mat66 information;
            
            if (VerifyLoopClosure(current_keyframe, candidate, relative_pose, information)) {
                // Create loop constraint
                double confidence = 1.0;  // Could be based on inlier ratio
                LoopConstraint constraint(current_keyframe->keyframe_id_, 
                                        candidate->keyframe_id_,
                                        relative_pose, information, confidence);
                
                {
                    std::unique_lock<std::mutex> lock(constraints_mutex_);
                    loop_constraints_.push_back(constraint);
                }
                
                LOG(INFO) << "Loop closure detected between keyframes " 
                         << current_keyframe->keyframe_id_ << " and " 
                         << candidate->keyframe_id_
                         << "  Relative Position : "
                         << relative_pose.translation().norm() << std::endl;

                break;  // Only one loop per keyframe
            }
        }
    }
}

void LoopClosing::ExtractORBFeatures(Frame::Ptr frame) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    orb_extractor_->detectAndCompute(frame->left_img_, cv::Mat(), keypoints, descriptors);
    
    // Store ORB features in frame (extend Frame class if needed)
    frame->orb_keypoints_ = keypoints;
    frame->orb_descriptors_ = descriptors;
}

std::vector<Frame::Ptr> LoopClosing::DetectLoopCandidates(Frame::Ptr current_frame) {
    std::vector<Frame::Ptr> candidates;
    std::vector<std::pair<double, Frame::Ptr>> similarity_scores;
    
    // Compare with frames that are far enough in time
    for (size_t i = 0; i < keyframe_database_.size(); ++i) {
        auto candidate = keyframe_database_[i];
        
        // Skip recent frames
        // TODO :  why 1->2 and 4->0 are detected as candidate
        if (current_frame->keyframe_id_ - candidate->keyframe_id_ < min_loop_interval_) {
            continue;
        }
        
        double similarity = ComputeVisualSimilarity(current_frame, candidate);
        if (similarity > visual_similarity_threshold_) {
            similarity_scores.push_back(std::make_pair(similarity, candidate));
        }
    }
    
    // Sort by similarity and return top candidates
    std::sort(similarity_scores.begin(), similarity_scores.end(), 
              std::greater<std::pair<double, Frame::Ptr>>());
    
    int max_candidates = 3;
    for (int i = 0; i < std::min(max_candidates, (int)similarity_scores.size()); ++i) {
        candidates.push_back(similarity_scores[i].second);
    }
    
    return candidates;
}

double LoopClosing::ComputeVisualSimilarity(Frame::Ptr frame1, Frame::Ptr frame2) {
    // Simple bag-of-words similarity using descriptor matching
    if (frame1->orb_descriptors_.empty() || frame2->orb_descriptors_.empty()) {
        return 0.0;
    }
    
    std::vector<cv::DMatch> matches;
    matcher_->match(frame1->orb_descriptors_, frame2->orb_descriptors_, matches);
    
    if (matches.empty()) return 0.0;
    
    // Filter good matches
    std::sort(matches.begin(), matches.end());
    int num_good_matches = 0;
    double distance_threshold = matches[0].distance * 2.0;
    
    for (const auto& match : matches) {
        if (match.distance < distance_threshold) {
            num_good_matches++;
        }
    }
    
    // Similarity based on ratio of good matches
    double similarity = double(num_good_matches) / std::max(frame1->orb_keypoints_.size(), 
                                                           frame2->orb_keypoints_.size());
    return std::min(similarity, 1.0);
}

bool LoopClosing::VerifyLoopClosure(Frame::Ptr current_frame, Frame::Ptr candidate_frame,
                                   SE3& relative_pose, Mat66& information) {
    // Match features between frames
    auto matches = MatchFeatures(current_frame, candidate_frame);
    
    if (matches.size() < min_inliers_) {
        return false;
    }
    
    // Estimate relative pose
    return EstimateRelativePose(current_frame, candidate_frame, matches, 
                               relative_pose, information);
}

std::vector<cv::DMatch> LoopClosing::MatchFeatures(Frame::Ptr frame1, Frame::Ptr frame2) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
    matcher_->knnMatch(frame1->orb_descriptors_, frame2->orb_descriptors_, 
                      knn_matches, 2);
    
    // Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }
    
    return matches;
}

bool LoopClosing::EstimateRelativePose(Frame::Ptr frame1, Frame::Ptr frame2,
                                      const std::vector<cv::DMatch>& matches,
                                      SE3& relative_pose, Mat66& information) {
    if (matches.size() < min_inliers_) {
        return false;
    }
    
    // Prepare 3D points for pose estimation
    std::vector<cv::Point3f> pts1, pts2;
    
    for (const auto& match : matches) {
        // Get 3D points from map points associated with features
        int idx1 = match.queryIdx;
        int idx2 = match.trainIdx;
        
        if (idx1 >= frame1->features_left_.size() || idx2 >= frame2->features_left_.size()) {
            continue;
        }
        
        auto mp1 = frame1->features_left_[idx1]->map_point_.lock();
        auto mp2 = frame2->features_left_[idx2]->map_point_.lock();
        
        if (!mp1 || !mp2) continue;
        
        Vec3 pos1 = mp1->Pos();
        Vec3 pos2 = mp2->Pos();
        
        pts1.push_back(cv::Point3f(pos1[0], pos1[1], pos1[2]));
        pts2.push_back(cv::Point3f(pos2[0], pos2[1], pos2[2]));
    }
    
    if (pts1.size() < min_inliers_) {
        return false;
    }
    
    // Solve for relative pose using RANSAC
    std::vector<bool> inliers;
    if (!SolveRANSAC(pts1, pts2, relative_pose, inliers)) {
        return false;
    }
    
    // Count inliers
    int inlier_count = std::count(inliers.begin(), inliers.end(), true);
    if (inlier_count < min_inliers_) {
        return false;
    }
    
    // Set information matrix based on inlier ratio
    double inlier_ratio = double(inlier_count) / matches.size();
    information = Mat66::Identity() * inlier_ratio * 100.0;  // Scale by confidence
    
    LOG(INFO) << "Loop verification successful with " << inlier_count 
             << " inliers out of " << matches.size() << " matches";
    
    return true;
}

bool LoopClosing::SolveRANSAC(const std::vector<cv::Point3f>& pts1,
                              const std::vector<cv::Point3f>& pts2,
                              SE3& pose, std::vector<bool>& inliers) {
    if (pts1.size() < 3 || pts1.size() != pts2.size()) {
        return false;
    }
    
    int best_inliers = 0;
    SE3 best_pose;
    inliers.resize(pts1.size(), false);
    
    for (int iter = 0; iter < ransac_iterations_; ++iter) {
        // Sample 3 random correspondences
        std::vector<int> sample_indices;
        for (int i = 0; i < 3; ++i) {
            int idx;
            do {
                idx = rand() % pts1.size();
            } while (std::find(sample_indices.begin(), sample_indices.end(), idx) != sample_indices.end());
            sample_indices.push_back(idx);
        }
        
        // Estimate pose from 3 point correspondences using Procrustes analysis
        Eigen::Matrix3Xd P1(3, 3), P2(3, 3);
        for (int i = 0; i < 3; ++i) {
            P1.col(i) = Vec3(pts1[sample_indices[i]].x, pts1[sample_indices[i]].y, pts1[sample_indices[i]].z);
            P2.col(i) = Vec3(pts2[sample_indices[i]].x, pts2[sample_indices[i]].y, pts2[sample_indices[i]].z);
        }
        
        // Compute centroids
        Vec3 c1 = P1.rowwise().mean();
        Vec3 c2 = P2.rowwise().mean();
        
        // Center the point sets
        P1.colwise() -= c1;
        P2.colwise() -= c2;
        
        // Compute rotation using SVD
        Eigen::Matrix3d H = P1 * P2.transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
        
        if (R.determinant() < 0) {
            Eigen::Matrix3d V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        
        // Compute translation
        Vec3 t = c2 - R * c1;
        
        SE3 candidate_pose(R, t);
        
        // Count inliers
        int current_inliers = 0;
        std::vector<bool> current_inlier_mask(pts1.size(), false);
        
        for (size_t i = 0; i < pts1.size(); ++i) {
            Vec3 p1(pts1[i].x, pts1[i].y, pts1[i].z);
            Vec3 p2(pts2[i].x, pts2[i].y, pts2[i].z);
            
            Vec3 p1_transformed = candidate_pose * p1;
            double error = (p1_transformed - p2).norm();
            
            if (error < ransac_threshold_) {
                current_inliers++;
                current_inlier_mask[i] = true;
            }
        }
        
        if (current_inliers > best_inliers) {
            best_inliers = current_inliers;
            best_pose = candidate_pose;
            inliers = current_inlier_mask;
        }
    }
    
    if (best_inliers >= min_inliers_) {
        pose = best_pose;
        return true;
    }
    
    return false;
}

}  // namespace myslam
