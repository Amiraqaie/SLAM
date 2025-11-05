#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;
using namespace std;

int main() {
    
    // SO3
    Matrix3d rotationMatrix = Matrix3d::Identity();

    AngleAxis rotationVector(M_PI/4, Vector3d(0,0,1));

    cout << rotationVector.toRotationMatrix() << endl;
    cout << rotationVector.angle() << endl;
    cout << rotationVector.axis() << endl;

    rotationMatrix = rotationVector.toRotationMatrix();
    cout << rotationMatrix << endl;
    
    Vector3d v(1,0,0);
    Vector3d vRotated = rotationMatrix * v;
    cout << vRotated << endl;
    Vector3d vRotated2 = rotationVector * v;
    cout << vRotated2 << endl;

    Vector3d eulerAngle = rotationMatrix.eulerAngles(2,1,0);
    cout << eulerAngle << endl;

    Quaterniond q = Quaterniond(rotationVector);
    cout << q.coeffs() << endl;
    Vector3d vRotated3 = q * v;
    cout << vRotated3 << endl;

    // SE3
    Isometry3d T = Isometry3d::Identity();
    T.rotate(rotationVector);
    T.pretranslate(Vector3d(1,2,3));
    cout << "Transform matrix : " << T.matrix() << endl;

    Vector3d p(1,2,3);
    Vector3d pTransformed = T * p;
    cout << pTransformed << endl;

    return 0;
}