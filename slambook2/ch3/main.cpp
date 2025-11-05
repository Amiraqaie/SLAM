#include <Eigen/Dense>
#include <iostream>
#include <ctime>
#include <Eigen/Core>

#define MATRIX_SIZE 50

using namespace std;
using namespace Eigen;


int main() {

    Matrix<float, 2, 3> matrix_23;
    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

    cout << "matrix 2x3 from 1 to 6: \n";
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << matrix_23(i, j) << " ";
        }
        cout << endl;
    }

    Vector3d v_3d;
    v_3d << 1, 2, 3;
    cout << "v_3d: " << v_3d << endl;

    Matrix<float, 3, 1> vd_3d;
    vd_3d << 1, 2, 3;
    cout << "vd_3d: " << vd_3d << endl;

    Matrix<double, 2, 1> result;
    result = matrix_23.cast<double>() * v_3d;
    cout << "result: "<< endl << result.transpose() << endl;

    Matrix<float, 2, 1> result_2;
    result_2 = matrix_23 * vd_3d;
    cout << "result_2: "<< endl << result_2.transpose() << endl;

    Matrix<float, 2, 1> result_3;
    result_3 = result_2.cast<float>() + result.cast<float>();
    cout << "result_3: "<< endl << result_3.transpose() << endl;

    Matrix3d matrix_3d = Matrix3d::Zero();

    Matrix<float, Dynamic, Dynamic> matrix_dynamic;

    MatrixXd matrix_xd;

    Matrix3d matrix_3d_random = Matrix3d::Random();
    cout << "matrix_3d_random: " << endl << matrix_3d_random << endl;
    cout << "transpose: " << endl << matrix_3d_random.transpose() << endl;
    cout << "sum: " << endl << matrix_3d_random.sum() << endl;
    cout << "trace: " << endl << matrix_3d_random.trace() << endl;
    cout << "determinant: " << endl << matrix_3d_random.determinant() << endl;
    cout << "inverse: " << endl << matrix_3d_random.inverse() << endl;
    cout << "adjoint: " << endl << matrix_3d_random.adjoint() << endl;
    cout << "eigenvalues: " << endl << matrix_3d_random.eigenvalues() << endl;
    cout << "times 10: " << endl << matrix_3d_random * 10 << endl;
    cout << "determinant: " << endl << matrix_3d_random.determinant() << endl;

    //solving equations
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    // start clock
    clock_t time_stt = clock();

    // direct inversion
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of direct inversion: " << (clock() - time_stt) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // solving with QR wich is very much faster
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of QR: " << (clock() - time_stt) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // solving using cholesky
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of cholesky: " << (clock() - time_stt) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
    cout << "x = " << x.transpose() << endl;
    
    return 0;
}
