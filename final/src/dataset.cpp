#include "dataset.h"
#include <fstream>

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
        
    }
}