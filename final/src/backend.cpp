//
// Created by gaoxiang on 19-5-2.
//

#include "backend.h"
#include "algorithm.h"
#include "feature.h"
#include "g2o_types.h"
#include "map.h"
#include "mappoint.h"

namespace myslam {

Backend::Backend()
{
    backend_running_.store(true);
    backend_thread_ = std::thread(&Backend::BackendLoop, this);
}

void Backend::UpdateMap()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

void Backend::Stop()
{
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);

        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);
    }
}

void Backend::Optimize(Map::KeyframesType& keyframes, 
                       Map::LandmarksType& landmarks)
{
    // Optimization code here
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // Add vertices for keyframes
    std::unordered_map<unsigned long, VertexPose*> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes)
    {
        auto kf = keyframe.second;
        VertexPose* v = new VertexPose();
        v->setId(kf->id_);
        v->setEstimate(kf->pose_);
        optimizer.addVertex(v);
        if (kf->id_ > max_kf_id)
            max_kf_id = kf->id_;
        vertices.insert({kf->keyframe_id_, v});
    }

    // add vertices for mappoints
    std::unordered_map<unsigned long, VertexXYZ*> vertices_landmark;
    
    // K
    Eigen::Matrix3d K = cam_left_->K();
    Sophus::SE3d left_ext = cam_left_->pose();
    Sophus::SE3d right_ext = cam_right_->pose();

    // add edges
    int index = 1;
    double chi2_th = 5.991; // 95% threshold for chi2 with 2 DOF
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    for (auto &landmark : landmarks)
    {
        if (landmark.second->is_outlier_)
            continue;
        unsigned long landmark_id = landmark.second->id_;
        auto mp = landmark.second;
        auto obs = mp->GetObs();
        for (auto &weak_feature : obs)
        {
            auto feature = weak_feature.lock();
            if (feature == nullptr || feature->is_outlier_)
                continue;
            auto kf = feature->frame_.lock();
            if (kf == nullptr)
                continue;
            EdgeProjection* edge = nullptr;
            if (feature->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
            } else {
                edge = new EdgeProjection(K, right_ext);
            }

            // add landmark vertex if not yet added
            if (vertices_landmark.find(landmark_id) == vertices_landmark.end())
            {
                VertexXYZ* v = new VertexXYZ();
                v->setId(landmark_id + max_kf_id + 1); // ensure unique id
                v->setEstimate(mp->Pos());
                v->setMarginalized(true);
                vertices_landmark.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            if (vertices.find(kf->keyframe_id_) != vertices.end()
                && vertices_landmark.find(landmark_id) != vertices_landmark.end())
            {

                edge->setId(index);
                edge->setVertex(0, vertices.at(kf->keyframe_id_));
                edge->setVertex(1, vertices_landmark.at(landmark_id));
                edge->setMeasurement(toVec2(feature->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feature});
                optimizer.addEdge(edge);
                index++;
            }
            else delete edge;
            
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmark) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}
}  // namespace myslam