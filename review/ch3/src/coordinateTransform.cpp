#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

int main()
{
    // robot1 transform matrix T_r1_w
    Isometry3d T1 = Isometry3d::Identity();
    T1.pretranslate(Vector3d(0.3, 0.1, 0.1));
    Quaterniond q1(0.35, 0.2, 0.3, 0.1);
    q1.normalize();
    T1.rotate(q1);

    // robot2 transform matrix T_r2_w
    Isometry3d T2 = Isometry3d::Identity();
    T2.pretranslate(Vector3d(-0.1, 0.5, 0.3));
    Quaterniond q2(-0.5, 0.4, -0.1, 0.2);
    q2.normalize();
    T2.rotate(q2);

    // a point in r1 coordinate
    Vector3d p_r1(0.5, 0.0, 0.2);

    // trasform p_r1 to r2 coordinate
    Vector3d p_r2 = T2 * T1.inverse() * p_r1;

    cout << p_r2 << endl;
    
    return 0;
}