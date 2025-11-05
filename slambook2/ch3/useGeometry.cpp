#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main() {

    // 3d rotation matrix
    Matrix3d rotation_matrix = Matrix3d::Identity();

    // rotaion vector 
    AngleAxis rotation_vector(M_PI/4, Vector3d(0,0,1));
    rotation_matrix = rotation_vector.toRotationMatrix();

    // rotate vector
    Vector3d v(1, 0, 0);
    Vector3d v_rotated;
    v_rotated = rotation_vector * v;
    cout << "(1, 0, 0) after rotation by rotation vector = \n " << v_rotated << endl;
    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation by rotation matrix = \n " << v_rotated << endl;

    // euler angles
    Vector3d euler_engles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order
    
    // cout precision
    cout.precision(3);
    cout << "rotation matrix = \n " << rotation_vector.toRotationMatrix() << endl; //convert to rotation matrix
    cout << "rotation vector = \n " << rotation_vector.angle() << " " << rotation_vector.axis() << endl; //convert to angle and axis
    cout << "yaw pitch roll = \n " << euler_engles << endl;

    // Euclidean transformation matrix using Eigen::Isometry
    Isometry3d T = Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Vector3d(1, 3, 4));
    cout << "Transform Matrix : \n : " << T.matrix() << endl;  

    // transform a vector by T
    Vector3d v_transformed;
    v_transformed = T * v;
    cout << "v transformed : \n" << v_transformed.transpose() << endl;

    // Quaternion
    Quaternion q = Quaterniond(rotation_vector);
    cout << "Quaternion : \n" << q.coeffs().transpose() << endl;
    q = Quaterniond(rotation_matrix);
    cout << "Quaternion : \n" << q.coeffs().transpose() << endl;
    v_rotated = q * v;
    cout << "(1, 0, 0) after rotation by quaternion = \n " << v_rotated << endl;

    return 0;
}