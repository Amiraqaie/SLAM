#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main() {

    // q1, q2 are rotation world to robot coordinate
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();

    // t1, t2 are vector from robot to world system
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);

    // Translation Isometry Matrixes
    Isometry3d T1w(q1), T2w(q2);
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);

    // move a point from robot 1 coordinate to robot 2 coordinate
    Vector3d p1(0.5, 0, 0.2);
    Vector3d p2 = T2w * T1w.inverse() * p1;
    cout << endl << p2.transpose() << endl;

    return 0;
}

