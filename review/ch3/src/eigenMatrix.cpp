#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;
using namespace std;

#define MATRIX_SIZE 50

int main()
{
    Matrix<float, 2, 3> matrix23;
    matrix23 << 1, 2, 3, 4, 5, 6;
    cout << "matrix23 has : " << matrix23 << endl;

    Vector3d v_3d;
    Matrix<double, 3, 1> vd_3d;
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    Matrix3d matrix_33 = Matrix3d::Random();
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;
    MatrixXd matrix_x;

    matrix_x = matrix23.cast<double>() * v_3d;

    // eigen values and vectors
    SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(matrix_33);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    auto eigenvectors = eigensolver.eigenvectors();

    // solving equations
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();
    Matrix<double, MATRIX_SIZE, 1> v_NN = MatrixXd::Random(MATRIX_SIZE, 1);

    // solve by inversion
    auto start_time = std::chrono::high_resolution_clock::now();
    MatrixXd x = matrix_NN.inverse() * v_NN;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    cout << "direct inverse duration : " << duration << "microsecond " << endl;

    // QR decomposition
    start_time = std::chrono::high_resolution_clock::now();
    x = matrix_NN.colPivHouseholderQr().solve(v_NN);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    cout << "QR decomposition duration : " << duration << "microsecond " << endl;

    // cholesky decomposition
    start_time = std::chrono::high_resolution_clock::now();
    x = matrix_NN.ldlt().solve(v_NN);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    cout << "cholesky decomposition duration : " << duration << "microsecond " << endl;

    return 0;
}