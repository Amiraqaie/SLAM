#include "loop_closing.h"
#include "algorithm.h"
#include "config.h"
#include <opencv2/core/eigen.hpp>
#include "mappoint.h"
#include <DBoW3/DBoW3.h>

namespace myslam {

LoopClosing::LoopClosing() {
    // Initialize ORB feature extractor
    orb_extractor_ = cv::ORB::create();
    gftt_detector_ = cv::GFTTDetector::create(
        Config::Get<int>("num_features"),
        0.01,
        20
    );

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
    minimum_match_ratio_ = Config::Get<double>("loop.minimum_match_ratio");
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
        
        
        // Verify loop closures
        for (auto candidate : candidates) {
            
            /// TODO : releative position should be calculated
            // relative_pose = T_kf1_kf2
            SE3 relative_pose;
            Mat66 information;
            
            /* TODO : we must triangulate features in current_keyframe coordinate 
            To pass it to pnp solver, if 3d positions of mappoint is in world
            coordinate then pnp ransac will fail
            */
           
           
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

void LoopClosing::ExtractORBDescriptors(Frame::Ptr frame) {

    std::vector<cv::KeyPoint> keypoints_left = frame->GetKeypointsLeft();
    std::vector<cv::KeyPoint> keypoints_right = frame->GetKeypointsRight();
    cv::Mat descriptors_left;
    cv::Mat descriptors_right;    
    
    orb_extractor_->compute(frame->left_img_, keypoints_left, descriptors_left);
    orb_extractor_->compute(frame->right_img_, keypoints_right, descriptors_right);

    // std::vector<cv::KeyPoint> keypoints_left;
    // std::vector<cv::KeyPoint> keypoints_right;
    // cv::Mat descriptors_left;
    // cv::Mat descriptors_right;    
    
    // gftt_detector_->detect(frame->left_img_, keypoints_left);
    // gftt_detector_->detect(frame->right_img_, keypoints_right);

    // orb_extractor_->compute(frame->left_img_, keypoints_left, descriptors_left);
    // orb_extractor_->compute(frame->right_img_, keypoints_right, descriptors_right);
    
    // Store ORB Descriptors and valid keypoints
    frame->orb_descriptors_left_ = descriptors_left;
    frame->orb_descriptors_right_ = descriptors_right;

    frame->valid_keypoints_left = keypoints_left;
    frame->valid_keypoints_right = keypoints_right;

    if (frame->GetDescriptorsLeft().rows != frame->GetValidKeypointsLeft().size())
        std::cout << "there is a bug!!!" << std::endl;
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

    std::vector<cv::DMatch> matches = MatchFeatures(current_frame, candidate_frame);

    // Estimate relative pose
    return EstimateRelativePose(current_frame, candidate_frame, matches, 
                               relative_pose, information);
}

std::vector<cv::DMatch> LoopClosing::MatchFeatures(Frame::Ptr frame1, Frame::Ptr frame2) {   

    // match descriptors
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> matches_knn;
    matcher_->knnMatch(frame1->GetDescriptorsLeft(), frame2->GetDescriptorsLeft(), matches_knn, 2);

    // Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < matches_knn.size(); i++) {
        if (matches_knn[i][0].distance < ratio_thresh * matches_knn[i][1].distance) {
            matches.push_back(matches_knn[i][0]);
        }
    }

    return matches;
}

std::vector<cv::DMatch> LoopClosing::MatchFeatures(cv::Mat descriptors1, cv::Mat descriptors2) {   

    // match descriptors
    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>> matches_knn;
    matcher_->knnMatch(descriptors1, descriptors2, matches_knn, 2);

    // Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < matches_knn.size(); i++) {
        if (matches_knn[i][0].distance < ratio_thresh * matches_knn[i][1].distance) {
            matches.push_back(matches_knn[i][0]);
        }
    }

    return matches;
}

std::map<int, Vec3> LoopClosing::TriangulateFeatures(Frame::Ptr frame) {
    
    std::map<int, Vec3> triangulated_points;
    std::vector<cv::KeyPoint> key_points_left = frame->GetValidKeypointsLeft();
    std::vector<cv::KeyPoint> key_points_right = frame->GetValidKeypointsRight();
    std::vector<cv::DMatch> matches = MatchFeatures(frame->GetDescriptorsLeft(), frame->GetDescriptorsRight());

    if ( key_points_left.empty() || key_points_right.empty() ) {
        LOG(WARNING) << "No ORB features found in right image!";
        return triangulated_points;
    }
    
    if (show_result_)
    {   
        std::string file_name = std::to_string(frame->id_) + "_LeftToRight.png";
        std::cout << file_name << std::endl;
        cv::Mat outImg;
        cv::drawMatches(frame->left_img_, key_points_left, frame->right_img_, key_points_right, matches, outImg);
        cv::imshow("left to right triangulation", outImg);
        // cv::imwrite(file_name, outImg);
    }


    // TODO : Check if feature have map point available
    // then use them in camera coordinate as triangulated result

    std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;


    std::unique_lock<std::mutex> lck(frame->keypoint_mutex_);
    for (auto match : matches)
    {
        // triangulation
        std::vector<Eigen::Vector3d> points{
            camera_left_->pixel2camera(Eigen::Vector2d(key_points_left[match.queryIdx].pt.x, key_points_left[match.queryIdx].pt.y)),
            camera_right_->pixel2camera(Eigen::Vector2d(key_points_right[match.trainIdx].pt.x, key_points_right[match.trainIdx].pt.y))
            };


        Eigen::Vector3d p_camera_left = Eigen::Vector3d::Zero();

        if (triangulation(poses, points, p_camera_left) && p_camera_left[2] > 0) {
            triangulated_points.insert(std::make_pair((int) match.queryIdx, p_camera_left));
            cnt_init_landmarks++;
        }
    }
                
    LOG(INFO) << "Loop Closure Triangulated " << cnt_init_landmarks << " 3D points";

    return triangulated_points;
}

std::vector<cv::DMatch> LoopClosing::TrackFeaturesLK(Frame::Ptr frame)
{
    std::vector<cv::DMatch> matches;
    
    std::vector<cv::KeyPoint> key_points_left = frame->GetValidKeypointsLeft();
    
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : key_points_left) {
        kps_left.push_back(kp.pt);
        kps_right.push_back(kp.pt);
    }

    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(
        frame->left_img_, frame->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // 4. Filtering and Creating DMatch objects
    for (size_t i = 0; i < key_points_left.size(); i++) {
        if (status[i] == 1) {
            // A successful track: prevPoints[i] -> nextPoints[i]
            
            cv::DMatch match;
            // queryIdx corresponds to the index in the input KeyPoint vector
            match.queryIdx = (int)i; 
            // trainIdx is set to the same index for tracking consistency
            match.trainIdx = (int)i; 
            
            // Use the tracking error as the 'distance'
            if (i < error.size()) {
                match.distance = error[i]; 
            } else {
                match.distance = 0.0f; 
            }
            
            matches.push_back(match);
        }
    }

    return matches;
}

std::map<int, Vec3> LoopClosing::TriangulateFeaturesLK(Frame::Ptr frame) {

    std::map<int, Vec3> triangulated_points;
    std::vector<cv::KeyPoint> key_points_left = frame->GetValidKeypointsLeft();
    std::vector<cv::KeyPoint> key_points_right = frame->GetValidKeypointsRight();
    std::vector<cv::DMatch> matches = TrackFeaturesLK(frame);

    if (show_result_)
    {   
        std::string file_name = std::to_string(frame->id_) + "_LeftToRight.png";
        std::cout << file_name << std::endl;
        cv::Mat outImg;
        cv::drawMatches(frame->left_img_, key_points_left, frame->right_img_, key_points_right, matches, outImg);
        cv::imshow("left to right triangulation", outImg);
        // cv::imwrite(file_name, outImg);
    }

    std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;

    std::unique_lock<std::mutex> lck(frame->keypoint_mutex_);
    for (auto match : matches)
    {
        // triangulation
        std::vector<Eigen::Vector3d> points{
            camera_left_->pixel2camera(Eigen::Vector2d(key_points_left[match.queryIdx].pt.x, key_points_left[match.queryIdx].pt.y)),
            camera_right_->pixel2camera(Eigen::Vector2d(key_points_right[match.trainIdx].pt.x, key_points_right[match.trainIdx].pt.y))
            };


        Eigen::Vector3d p_camera_left = Eigen::Vector3d::Zero();

        if (triangulation(poses, points, p_camera_left) && p_camera_left[2] > 0) {
            triangulated_points.insert(std::make_pair((int) match.queryIdx, p_camera_left));
            cnt_init_landmarks++;
        }
    }
                
    LOG(INFO) << "Loop Closure Triangulated " << cnt_init_landmarks << " 3D points";

    return triangulated_points;
}

bool LoopClosing::EstimateRelativePose(Frame::Ptr frame1, Frame::Ptr frame2,
                                       const std::vector<cv::DMatch>& matches,
                                       SE3& relative_pose, Mat66& information) {
    if (matches.size() < static_cast<size_t>(min_inliers_))
        return false;

    // show result of loop closure
    if (show_result_)
    {   
        // Match features between frames
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        keypoints1 = frame1->GetValidKeypointsLeft();
        keypoints2 = frame2->GetValidKeypointsLeft();
        std::string file_name1 = std::to_string(frame1->id_) + "_" + std::to_string(frame2->id_) + ".png";
        std::cout << "Number of matches: " << matches.size() << std::endl;
        cv::Mat outImg;
        cv::drawMatches(frame1->left_img_, keypoints1, frame2->left_img_, keypoints2, matches, outImg);
        cv::imshow("loop closure result", outImg);
        // cv::imwrite(file_name1, outImg);
    }

    // Step 1: Triangulate ORB features in frame1 using stereo
    std::map<int, Vec3> pts3d_frame1 = TriangulateFeatures(frame1);

    // Step 2: Prepare PnP correspondences
    std::vector<cv::Point3d> objectPoints; // 3D points in frame1 (left cam)
    std::vector<cv::Point2d> imagePoints;  // 2D points in frame2 (left image)

    for (const auto& m : matches) {
        int idx1 = m.queryIdx;  // left frame1 ORB index
        int idx2 = m.trainIdx;  // left frame2 ORB index

        if (idx1 >= (int) pts3d_frame1.size() || idx2 >= (int) frame2->GetValidKeypointsLeft().size())
            continue;

        if (pts3d_frame1.find(idx1) == pts3d_frame1.end())
            continue;
        
        Vec3 p3d = pts3d_frame1[idx1];
        if (p3d[2] <= 0) continue; // ignore points behind camera

        objectPoints.push_back(cv::Point3d(p3d[0], p3d[1], p3d[2]));
        imagePoints.push_back(frame2->GetValidKeypointsLeft()[idx2].pt);
    }

    if (objectPoints.size() < static_cast<size_t>(min_inliers_))
        return false;

    // Step 3: Solve PnP RANSAC
    cv::Mat rvec, tvec;
    cv::Mat K = (cv::Mat_<double>(3,3) << camera_left_->fx_, 0, camera_left_->cx_,
    0, camera_left_->fy_, camera_left_->cy_,
    0, 0, 1);
    
    std::cout << "objectPoints size : " << objectPoints.size() << std::endl;
    std::cout << "imagePoints size : " << imagePoints.size() << std::endl;
    
    // Step 5: Set information matrix proportional to inlier ratio
    int inliers = EstimateCandidatePose(objectPoints, imagePoints, relative_pose);

    double inlier_ratio = double(inliers) / objectPoints.size();
    information = Mat66::Identity() * inlier_ratio * 100.0;

    LOG(INFO) << "Loop verification successful with " << inliers << " inliers out of " << objectPoints.size() << " matches";

    return true;
}

int LoopClosing::EstimateCandidatePose(std::vector<cv::Point3d> &objectPoints, std::vector<cv::Point2d> &imagePoints, SE3& relative_pose)
{
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(
            std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Vertex
    VertexPose *vertex_pose = new VertexPose(); 
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    // K 
    Eigen::Matrix3d K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<bool> imagePointsStatus;
    for (cv::Point2d pt: imagePoints)
    {
        imagePointsStatus.push_back(false);
    }

    for (size_t i = 0; i < objectPoints.size(); i++) {
        Eigen::Vector3d pos3d(objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        Eigen::Vector2d measurement(imagePoints[i].x, imagePoints[i].y);
        auto edge = new EdgeProjectionPoseOnly(pos3d, K);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(measurement);
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        edge->setInformation(Eigen::Matrix2d::Identity());
        edges.push_back(edge);
        optimizer.addEdge(edge);
        index++;
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(Sophus::SE3d());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            if (imagePointsStatus[i]) {
                e->computeError();
            }
            if (e->chi2() > chi2_th) {
                imagePointsStatus[i] = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {
                imagePointsStatus[i] = false;
                e->setLevel(0);
            };

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << imagePointsStatus.size() - cnt_outlier;

    relative_pose = vertex_pose->estimate();

    LOG(INFO) << "Candidate Pose = \n" << vertex_pose->estimate().matrix();

    return imagePointsStatus.size() - cnt_outlier;
}


}