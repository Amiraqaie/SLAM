#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unistd.h>
#include <iostream>

using namespace Eigen;
using namespace std;
using namespace pangolin;

// path to trajectory file
string trajectory_file = "./trajectory.txt";

void DrwTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv){
    // vector of all poses
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
    
    // open input file
    ifstream fin(trajectory_file);

    // check file exists
    if (!fin){
        cout << "trajectory file not found at " << trajectory_file << endl;
        return 1;
    }

    // read poses
    while (!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz));
        poses.push_back(Twr);
    }
    cout << "read " << poses.size() << " poses" << endl;
    fin.close();

    // draw trajectory in pangolin
    DrwTrajectory(poses);

    return 0;
}

void DrwTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses){
    // create pangolin display
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // set camera
    pangolin::OpenGlRenderState d_cam(
        pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.2, 0.2, 0.2)
    );

    pangolin::View &s_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, 1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(d_cam));

    while (pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        s_cam.Activate(d_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses.size() - 1; i++){
            // draw three axes of each pose
            Vector3d Ow = poses[i].translation();
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));

            glBegin(GL_LINES);
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(Ow[0], Ow[1], Ow[2]);
            glVertex3f(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0f, 1.0f, 0.0f);
            glVertex3f(Ow[0], Ow[1], Ow[2]);
            glVertex3f(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3f(Ow[0], Ow[1], Ow[2]);
            glVertex3f(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }

        // draw a connection
        for (size_t i = 0; i < poses.size() -1; i++){
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        pangolin::FinishFrame();
        usleep(5000);
    }
}