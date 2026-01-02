// #ifdef BACKEND_H
#define BACKEND_H

#include "common_include.h"
#include "frame.h"
#include "map.h"


class Backend {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend();

    void SetCameras(Camera::Ptr left, Camera::Ptr right)
    {
        camera_left_ = left;
        camera_right_ = right;
    }

    void SetMap(Map::Ptr map) { map_ = map; }

    void UpdateMap();

    void Stop();

private:

    void BackendLoop();

    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    Map::Ptr map_ = nullptr;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr camera_left_ = nullptr;
    Camera::Ptr camera_right_ = nullptr;
};

// #endif  // BACKEND_H