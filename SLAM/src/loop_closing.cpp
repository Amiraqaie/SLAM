#include "loop_closing.h"
#include "algorithm.h"
#include "config.h"
#include <opencv2/core/eigen.hpp>
#include "mappoint.h"
#include <DBoW3/DBoW3.h>

namespace myslam {

LoopClosing::LoopClosing() {
    // Initialize ORB feature extractor
    orb_extractor_ = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, 
                                     cv::ORB::HARRIS_SCORE, 31, 20);
    
    // Initialize feature matcher
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    
    // Initialize DBOW3 Vacabulary
    std::string vocab_path = Config::Get<std::string>("loop.vocab_path");
    vocab_.load(vocab_path);
    
    if (vocab_.empty()) {
        std::cerr << "Vocabulary does not exist." << std::endl;
        std::cerr << "LOOP CLOSURE Thread will not execute !!!! ." << std::endl;
        return;
    }
    db_.setVocabulary(vocab_, false, 0);

    // Load parameters from config
    visual_similarity_threshold_ = Config::Get<double>("loop.visual_similarity_threshold");
    min_loop_interval_ = Config::Get<int>("loop.min_loop_interval");
    ransac_iterations_ = Config::Get<int>("loop.ransac_iterations");
    ransac_threshold_ = Config::Get<double>("loop.ransac_threshold");
    min_inliers_ = Config::Get<int>("loop.min_inliers");
    dbow_query_size_ = Config::Get<int>("loop.dbow_query_size");
    show_result_ = static_cast<bool>(Config::Get<int>("loop.show_result"));

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
        
        // Extract ORB Descriptors for the current keyframe feature
        ExtractORBDescriptors(current_keyframe);
        
        // Add to database
        // TODO : use DBOW3 to add frames to DataBase
        keyframe_database_.push_back(current_keyframe);
        db_.add(current_keyframe->GetDescriptorsLeft());

        // Skip loop detection for early keyframes
        if (keyframe_database_.size() < static_cast<size_t>(min_loop_interval_)) continue;
        
        // Detect loop candidates
        auto candidates = DetectLoopCandidates(current_keyframe);
        
        // show result of loop closure
        if (show_result_ && candidates.size() > 0)
        {   
            cv::imshow("current keyframe", current_keyframe->left_img_);
            cv::imshow("candidate with higher score", candidates[0]->left_img_);
        }

        // Verify loop closures
        for (auto candidate : candidates) {

            /// TODO : releative position should be calculated
            // relative_pose = T_kf1_kf2
            SE3 relative_pose;
            Mat66 information;
            
            /* TODO : we must triangulate features in current_keyframe coordinate 
                To pass it to pnp solver, if 3d positions of mappoint is in world
                coordinate then pnp ransac will fail


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
            */
        }
    }
}

void LoopClosing::ExtractORBDescriptors(Frame::Ptr frame) {
    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;
    cv::Mat descriptors_left;
    cv::Mat descriptors_right;
    
    for (Feature::Ptr feat : frame->features_left_)
    {
        if (!feat)
            continue;
        std::unique_lock<std::mutex> lck(frame->feature_mutex_);
        keypoints_left.push_back(feat->position_);
    }
    for (Feature::Ptr feat : frame->features_right_)
    {
        if (!feat)
            continue;
        std::unique_lock<std::mutex> lck(frame->feature_mutex_);
        keypoints_right.push_back(feat->position_);
    }        

    orb_extractor_->compute(frame->left_img_, keypoints_left, descriptors_left);
    orb_extractor_->compute(frame->left_img_, keypoints_right, descriptors_right);

    // Store ORB Descriptors
    frame->orb_descriptors_left_ = descriptors_left;
    frame->orb_descriptors_right_ = descriptors_right;
}

std::vector<Frame::Ptr> LoopClosing::DetectLoopCandidates(Frame::Ptr current_frame) {
    std::vector<Frame::Ptr> candidates;
    std::vector<std::pair<double, Frame::Ptr>> similarity_scores;

    // Get query of top four candidates
    DBoW3::QueryResults ret;
    db_.query(current_frame->GetDescriptorsLeft(), ret, dbow_query_size_);
    std::cout << "searching for KeyFrame " << current_frame->keyframe_id_ << " returns " << ret << std::endl;
    
    // calculate similarity score with previous keyframe
    double gt_score = ComputeVisualSimilarity(current_frame, keyframe_database_[current_frame->keyframe_id_ - 1]);

    for (auto result : ret)
    {
        if ((current_frame->keyframe_id_ - result.Id) > min_loop_interval_)
        {

            double score = result.Score;
            double similarity = score / gt_score;
            if (similarity > visual_similarity_threshold_)
            {
                int id = result.Id;
                candidates.push_back(keyframe_database_[id]);
            }
        }
    }

    return candidates;
}

double LoopClosing::ComputeVisualSimilarity(Frame::Ptr frame1, Frame::Ptr frame2) {

    // calculate similarity score with previous keyframe
    DBoW3::BowVector frame1_vector;
    vocab_.transform(frame1->GetDescriptorsLeft(), frame1_vector);
    DBoW3::BowVector frame2_vector;
    vocab_.transform(keyframe_database_[frame2->keyframe_id_ - 1]->GetDescriptorsLeft(), frame2_vector);
    double score = vocab_.score(frame1_vector, frame2_vector);
    return score;
}

bool LoopClosing::VerifyLoopClosure(Frame::Ptr current_frame, Frame::Ptr candidate_frame,
                                   SE3& relative_pose, Mat66& information) {
    // Match features between frames
    auto matches = MatchFeatures(current_frame, candidate_frame);
    
    // Estimate relative pose
    // TODO :  we should add spacial loop consistency check to this pipline
    return EstimateRelativePose(current_frame, candidate_frame, matches, 
                               relative_pose, information);
}

std::vector<cv::DMatch> LoopClosing::MatchFeatures(Frame::Ptr frame1, Frame::Ptr frame2) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
    matcher_->knnMatch(frame1->orb_descriptors_left_, frame2->orb_descriptors_left_, 
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

std::vector<cv::DMatch> LoopClosing::MatchFeatures(cv::Mat descriptors1, cv::Mat descriptors2) {
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> knn_matches;

    matcher_->knnMatch(descriptors1, descriptors2, 
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

std::vector<Vec3> LoopClosing::TriangulateFeatures(Frame::Ptr frame) {
    std::vector<Vec3> pts3d;

    if (frame->GetKeypointsLeft().empty() || frame->GetKeypointsRight().empty()) {
        LOG(WARNING) << "No ORB features found in right image!";
        return pts3d;
    }



    return pts3d;
}

bool LoopClosing::EstimateRelativePose(Frame::Ptr frame1, Frame::Ptr frame2,
                                       const std::vector<cv::DMatch>& matches,
                                       SE3& relative_pose, Mat66& information) {
    if (matches.size() < static_cast<size_t>(min_inliers_))
        return false;

    // Step 1: Triangulate ORB features in frame1 using stereo
    std::vector<Vec3> pts3d_frame1 = TriangulateFeatures(frame1);

    // Step 2: Prepare PnP correspondences
    std::vector<cv::Point3f> objectPoints; // 3D points in frame1 (left cam)
    std::vector<cv::Point2f> imagePoints;  // 2D points in frame2 (left image)

    for (const auto& m : matches) {
        int idx1 = m.queryIdx;  // left frame ORB index
        int idx2 = m.trainIdx;  // right frame ORB index

        if (idx1 >= pts3d_frame1.size() || idx2 >= frame2->GetKeypointsLeft().size())
            continue;

        Vec3 p3d = pts3d_frame1[idx1];
        if (p3d[2] <= 0) continue; // ignore points behind camera

        objectPoints.push_back(cv::Point3f(p3d[0], p3d[1], p3d[2]));
        imagePoints.push_back(frame2->GetKeypointsLeft()[idx2].pt);
    }

    if (objectPoints.size() < static_cast<size_t>(min_inliers_))
        return false;

    // Step 3: Solve PnP RANSAC
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    cv::Mat K = (cv::Mat_<double>(3,3) << camera_left_->fx_, 0, camera_left_->cx_,
                                           0, camera_left_->fy_, camera_left_->cy_,
                                           0, 0, 1);

    bool success = cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(),
                                      rvec, tvec, false,
                                      100,          // max iterations
                                      8.0,          // reprojection error threshold in pixels
                                      min_inliers_, // minimum inliers
                                      inliers, cv::SOLVEPNP_ITERATIVE);

    if (!success || inliers.size() < static_cast<size_t>(min_inliers_))
        return false;

    // Step 4: Convert rvec/tvec to SE3
    cv::Mat R;
    cv::Rodrigues(rvec, R); // 3x3 rotation matrix

    Eigen::Matrix3d rot;
    Eigen::Vector3d trans;
    cv::cv2eigen(R, rot);
    cv::cv2eigen(tvec, trans);

    // relative_pose = T_kf1_kf2
    relative_pose = SE3(rot, trans);

    // Step 5: Set information matrix proportional to inlier ratio
    double inlier_ratio = double(inliers.size()) / objectPoints.size();
    information = Mat66::Identity() * inlier_ratio * 100.0;

    LOG(INFO) << "Loop verification successful with " << inliers.size()
              << " inliers out of " << objectPoints.size() << " matches";

    return true;
}


}  // namespace myslam
