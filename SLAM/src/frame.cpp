#include "frame.h"
#include "feature.h"

namespace myslam{

Frame::Frame(long id, double time_stamp, const Sophus::SE3d &pose, const cv::Mat &left, const cv::Mat &right) 
      : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

Frame::Ptr Frame::CreateFrame() 
{
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::SetKeyFrame() 
{
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

std::vector<cv::KeyPoint> Frame::GetKeypointsLeft()
{
    std::vector<cv::KeyPoint> keypoints;
    
    for (Feature::Ptr feat : features_left_)
    {
        if (!feat)
            continue;
        std::unique_lock<std::mutex> lck(feature_mutex_);
        keypoints.push_back(feat->position_);
    }

    return keypoints;
}

std::vector<cv::KeyPoint> Frame::GetKeypointsRight()
{
    std::vector<cv::KeyPoint> keypoints;
    
    for (Feature::Ptr feat : features_left_)
    {
        if (!feat)
            continue;
        std::unique_lock<std::mutex> lck(feature_mutex_);
        keypoints.push_back(feat->position_);
    }

    return keypoints;
}

cv::Mat Frame::GetDescriptorsLeft()
{
    return orb_descriptors_left_;
}

cv::Mat Frame::GetDescriptorsRight()
{
    return orb_descriptors_right_;
}

}