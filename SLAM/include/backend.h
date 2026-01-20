#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "common_include.h"
#include "frame.h"
#include "map.h"

namespace myslam {
    class PoseGraphOptimization;
    class Map;
}


namespace myslam {

class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend();

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        cam_left_ = left;
        cam_right_ = right;
    }

    void SetMap(Map::Ptr map) { map_ = map; }

    void SetPoseGraph(std::shared_ptr<PoseGraphOptimization> pose_graph) { pose_graph_ = pose_graph; }

    void UpdateMap();

    void Stop();


    void RequestPause();
    void WaitUntilPaused();
    void Resume();
private:
    void BackendLoop();
    
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

private:
    enum class BackendState {
        IDLE,
        OPTIMIZING,
        PAUSED
    };

    BackendState state_ = BackendState::IDLE;
    bool pause_requested_ = false;
    
    std::mutex state_mutex_;
    std::condition_variable state_cv_;

    Map::Ptr map_ = nullptr;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;


    Camera::Ptr cam_left_ = nullptr;
    Camera::Ptr cam_right_ = nullptr;

    std::shared_ptr<PoseGraphOptimization> pose_graph_;
};

}

#endif