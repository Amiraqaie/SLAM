#pragma once

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <common_include.h>

struct Feature;

struct MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<MapPoint> Ptr;


    unsigned long id_;
    bool is_outlier_ = false;
    Eigen::Vector3d pos_;
    std::mutex data_mutex_;
    int observed_times_ = 0;
    std::list<std::weak_ptr<Feature>> observations_;
 
    MapPoint();
    MapPoint(unsigned long id, Eigen::Vector3d p);

    Eigen::Vector3d Pos() {
        // doesn't need lock ?? 
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;        
    }

    void SetPos(Eigen::Vector3d &p) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = p;
    }

    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feature);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    static MapPoint::Ptr CreateNewMappoint();
};


#endif