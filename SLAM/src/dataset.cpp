#include "dataset.h"
#include "frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace myslam {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    // read camera intrinsics and extrinsics
    std::ifstream fin(dataset_path_ + "/calib.txt");
    
    if (!fin) {
        LOG(ERROR) << "connot find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    for (int i = 0; i < 4; i++) {
        char camera_name[3];
        for (int k = 0; k < 3; k++) {
            fin >> camera_name[k];
        }

        double projection_data[12];
        for (int j = 0; j < 12; j++) {
            fin >> projection_data[j];
        }
        
        Eigen::Matrix3d K;
        K << projection_data[0], projection_data[1], projection_data[2],
             projection_data[4], projection_data[5], projection_data[6],
             projection_data[8], projection_data[9], projection_data[10];
        
        Eigen::Vector3d t;
        t << projection_data[3], projection_data[7], projection_data[11];

        t = K.inverse() * t;
        K = K * 0.5;  // image is resized to half size

        Sophus::SE3d pose(Eigen::Matrix3d::Identity(), t);
        Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                          t.norm(), pose));
        cameras_.push_back(new_camera);
        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();

    // ================= LOAD POSES =================
    std::ifstream pose_file(dataset_path_ + "/poses.txt");
    if (!pose_file) {
        LOG(WARNING) << "No poses.txt found, running without ground truth";
        return true;  // allow VO-only mode
    }

    while (true) {
        Eigen::Matrix<double, 3, 4> T;
        for (int i = 0; i < 12; i++) {
            if (!(pose_file >> T(i / 4, i % 4))) {
                break;  // <<< just break the inner for loop
            }
        }

        // Check if reading failed (end of file)
        if (pose_file.eof() || pose_file.fail()) {
            break;  // <<< break the while loop
        }

        Eigen::Matrix3d R = T.block<3,3>(0,0);
        Eigen::Vector3d t = T.block<3,1>(0,3);

        // ---- FORCE R INTO SO(3) ----
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            R, Eigen::ComputeFullU | Eigen::ComputeFullV);

        R = svd.matrixU() * svd.matrixV().transpose();

        if (R.determinant() < 0) {
            R = -R;  // handle reflection edge case
        }

        Sophus::SE3d T_w_c(R, t);
        poses_.push_back(T_w_c);
    }

    // Logging after all poses are read
    LOG(INFO) << "Loaded " << poses_.size() << " ground-truth poses.";
    pose_file.close();

    double total_dist = 0;
    for (int i = 1; i < poses_.size(); ++i) {
        total_dist += (poses_[i].translation() - poses_[i-1].translation()).norm();
    }
    LOG(INFO) << "Total Trajectory length of sequence " << total_dist << " meters.";

    current_images_index_ = 0;
    return true;
}

Frame::Ptr Dataset::NextFrame() {
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;
    // read images
    image_left =
        cv::imread((fmt % dataset_path_ % 0 % current_images_index_).str(),
                   cv::IMREAD_GRAYSCALE);
    image_right =
        cv::imread((fmt % dataset_path_ % 1 % current_images_index_).str(),
                   cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "cannot find images at index " << current_images_index_;
        return nullptr;
    }

    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->left_img_ = image_left_resized;
    new_frame->right_img_ = image_right_resized;

    // ================= SET GT POSE =================
    if (current_images_index_ < poses_.size()) {
        new_frame->SetGtPose(poses_[current_images_index_].inverse());
    }

    current_images_index_++;
    return new_frame;
}

}  // namespace myslam