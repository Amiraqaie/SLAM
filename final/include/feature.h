#pragma once

#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/features2d.hpp>
#include "common_include.h"

struct Frame;
struct MapPoint;

struct Feature
{
public:
    
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;
    std::weak_ptr<MapPoint> map_point_;

    cv::KeyPoint position_;

    bool is_outlier_ = false;
    bool is_on_left_ = false;

public:
    Feature() {};

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &keypoint)
    : frame_(frame), position_(keypoint) {};

};


#endif