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
        if (show_result_)
        {   
            cv::imshow("current keyframe", current_keyframe->left_img_);
            cv::imshow("candidate with higher score", candidates[1]->left_img_);
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

    /* 
    std::vector<std::pair<double, myslam::Frame::Ptr>> clossest_keyframes_database;
    for (size_t i = 0; i < keyframe_database_.size(); ++i) {
        auto candidate = keyframe_database_[i];
        Sophus::SE3d estimated_distance =  current_frame->Pose().inverse() * candidate->Pose();
        double distance = estimated_distance.translation().norm();
        clossest_keyframes_database.push_back({distance, candidate});
        }
    // Sort by distance
    std::sort(clossest_keyframes_database.begin(), clossest_keyframes_database.end(), [](auto& a, auto& b){
        return a.first < b.first;
    });

    std::vector<myslam::Frame::Ptr> top10_keyframes;
    
    int count = 0;
    for (auto it = clossest_keyframes_database.begin(); it != clossest_keyframes_database.end() && count < 10; ++it, ++count) {
        top10_keyframes.push_back(it->second);
    }

    // Optional: print their IDs
    for (auto& kf : top10_keyframes) {
        std::cout << "Top candidate KF ID: " << kf->keyframe_id_ << std::endl;
    }

    // Compare with frames that are far enough in time
    for (size_t i = 0; i < top10_keyframes.size(); ++i) {
        auto candidate = top10_keyframes[i];
        
        // Skip recent frames
        // TODO :  why 1->2 and 4->0 are detected as candidate
        if (current_frame->keyframe_id_ - candidate->keyframe_id_ < static_cast<size_t>(min_loop_interval_)) {
            continue;
        }
        
        double similarity = ComputeVisualSimilarity(current_frame, candidate);
        // std::cout << "Similarity Between : KF" << current_frame->keyframe_id_ << " AND KF" << candidate->keyframe_id_ << " is equal to = " << similarity << std::endl;
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
    */
   
    // TODO : write a for loop to calculate similarity of the current frame with 
    // all keyframe database frames and cout the results one-by-one

    // Get query of top four candidates
    DBoW3::QueryResults ret;
    db_.query(current_frame->GetDescriptorsLeft(), ret, 4);
    std::cout << "searching for KeyFrame " << current_frame->keyframe_id_ << " returns " << ret << std::endl;
    for (auto result : ret)
    {
        int id = result.Id;
        candidates.push_back(keyframe_database_[id]);
    }

    // // Show the results of candidates
    // cv::imshow("current KeyFrame", current_frame->left_img_);
    // std::string window_name = "Candidate ";
    // cv::imshow(window_name, candidates[1]->left_img_);
    // cv::waitKey(0);

    return candidates;
}

double LoopClosing::ComputeVisualSimilarity(Frame::Ptr frame1, Frame::Ptr frame2) {

    double clipped_similarity = 0.0;

    /*
    // Simple bag-of-words similarity using descriptor matching
    if (frame1->orb_descriptors_left_.empty() || frame2->orb_descriptors_left_.empty()) {
        return 0.0;
    }
    
    std::vector<cv::DMatch> matches = MatchFeatures(frame1, frame2);    
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
    double similarity = double(num_good_matches) / std::max(frame1->features_left_.size(), 
    frame2->features_left_.size());
    cv::Mat match_result;
    cv::drawMatches(frame1->left_img_, frame1->GetKeypointsLeft(), frame2->left_img_, frame2->GetKeypointsLeft(), matches, match_result);
    static double max_similarity = 0;
    clipped_similarity = std::min(similarity, 1.0);
    max_similarity = std::max(clipped_similarity, max_similarity);
    std::string window_name = "Similarity Checking !!! ";
    std::string explanation = std::to_string(clipped_similarity) + "Max similarity Was = " + std::to_string(max_similarity);
    std::string ids_exp = "id current : " + std::to_string(frame1->keyframe_id_);
    // Draw text on match_result
    cv::putText(match_result, explanation, cv::Point(20, 30), 
    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 1);
    cv::putText(match_result, ids_exp, cv::Point(20, 100), 
    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 1);    
    cv::imshow(window_name, match_result);
    cv::waitKey(1);
    // cv::destroyWindow(window_name);
    */

    // TODO : we must use dbow3 to calculate the similarity
    return clipped_similarity;
}

bool LoopClosing::VerifyLoopClosure(Frame::Ptr current_frame, Frame::Ptr candidate_frame,
                                   SE3& relative_pose, Mat66& information) {
    // Match features between frames
    auto matches = MatchFeatures(current_frame, candidate_frame);
    
    // Estimate relative pose
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

/* TODO : its better to move ransac solving in a seprate function (used in EstimateRelativePosition)
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
*/

}  // namespace myslam
