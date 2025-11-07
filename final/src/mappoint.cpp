#include <mappoint.h>
#include <feature.h>

MapPoint::MapPoint() {}

MapPoint::MapPoint(unsigned long id, Eigen::Vector3d p)
    : id_(id), pos_(p) {}

// void MapPoints::RemoveObservation(std::shared_ptr<Feature> feature) {
//     std::unique_lock<std::mutex> lck(data_mutex_);
//     observations_.remove(feature);
//     observed_times_--;
// }

void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end();
         iter++) {
        if (iter->lock() == feat) {
            observations_.erase(iter);
            feat->map_point_.reset();
            observed_times_--;
            break;
        }
    }
}

MapPoint::Ptr MapPoint::CreateNewMappoint() {
    static unsigned long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}