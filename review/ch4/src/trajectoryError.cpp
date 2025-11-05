#include <iostream>
#include <fstream>
#include <sstream>
// #include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;
using namespace Sophus;

std::string estimated_path = "/home/amir/SLAM/review/ch4/estimated.txt";
std::string groundtruth_path = "/home/amir/SLAM/review/ch4/groundtruth.txt";


int main( )
{

    std::vector<Sophus::SE3d> estimated_poses;
    std::vector<Sophus::SE3d> groundtruth_poses;

    ifstream estimated_file(estimated_path);
    ifstream groundtruth_file(groundtruth_path);
    std::string line;

    while (getline(estimated_file, line))
    {
        // cout << line << endl;
        std::istringstream iss(line);
        double time, tx, ty, tz, qx, qy, qz, qw;
        iss >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Vector3d t(tx, ty, tz);
        Sophus::SE3d T_w_r(q, t);
        estimated_poses.push_back(T_w_r);
    }
    
    
    while (getline(groundtruth_file, line))
    {
        // cout << line << endl;
        std::istringstream iss(line);
        double time, tx, ty, tz, qx, qy, qz, qw;
        iss >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Vector3d t(tx, ty, tz);
        Sophus::SE3d T_w_r(q, t);
        // cout << T_w_r.matrix() << endl;
        groundtruth_poses.push_back(T_w_r);
    }

    cout << "estimated poses size : " << estimated_poses.size() << endl;
    cout << "groudtruth poses size : " << groundtruth_poses.size() << endl;

    // calculate error:
    std::vector<double> pose_errors;
    std::vector<Sophus::SE3d>::iterator estimate_pose_iterator = estimated_poses.begin();
    std::vector<Sophus::SE3d>::iterator groundtruth_pose_iterator = groundtruth_poses.begin();
    double rmse = 0;

    while (estimate_pose_iterator != estimated_poses.end() || groundtruth_pose_iterator != groundtruth_poses.end())
    {
        double error = ((*groundtruth_pose_iterator).inverse() * (*estimate_pose_iterator)).log().norm();
        estimate_pose_iterator++;
        groundtruth_pose_iterator++;
        // cout << "estimated poses  : " << estimate_pose_iterator->matrix() << endl;
        // cout << "groudtruth poses size : " << groundtruth_pose_iterator->matrix() << endl;
        cout << "error  =  " << error << endl;
        rmse += error * error;
    }
    
    rmse = rmse / double(estimated_poses.size());
    rmse = sqrt(rmse);
    cout << "rmse = " << rmse << endl;


    
    return 0;
}