#include <iostream>
#include </usr/include/eigen3/Eigen/Core>
#include </usr/include/eigen3/Eigen/Dense>
#include <random>

using namespace std;

int main()
{
    double sigma = 1.0;
    int DATA_SIZE = 1000;
    int ITERATION = 20;

    double a = 1.0, b = 2.0, c = 1.0;
    double ae = 0.0, be = 4.0, ce = 2.0;

    vector<double> x_data, y_data;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, sigma);

    for (double i = 0; i < DATA_SIZE; i++)
    {
        double x = i / DATA_SIZE;
        double y = exp(a * x * x + b * x + c) + (double) dist(gen);

        x_data.push_back(x);
        y_data.push_back(y);
    }

    // Gaussian Newton optimization method
    for (size_t t = 0; t < ITERATION; t++)
    {
        Eigen::Matrix<double, 3, 3> H = Eigen::Matrix<double, 3, 3>::Zero();
        Eigen::Matrix<double, 3, 1> b = Eigen::Matrix<double, 3, 1>::Zero();
        double cost = 0;

        for (size_t m = 0; m < y_data.size(); m++)
        {
            double residual = y_data[m] - exp(ae * x_data[m] * x_data[m] + be * x_data[m] + ce);
            Eigen::Matrix<double, 3, 1> J = Eigen::Matrix<double, 3, 1>::Zero();
            J(0, 0) = -x_data[m] * x_data[m] * exp(ae * x_data[m] * x_data[m] + be * x_data[m] + ce);
            J(1, 0) = -x_data[m] * exp(ae * x_data[m] * x_data[m] + be * x_data[m] + ce);
            J(2, 0) = -exp(ae * x_data[m] * x_data[m] + be * x_data[m] + ce);

            H += J * J.transpose();
            b += -J * residual;

            cost += residual * residual;
        }

        Eigen::Vector3d dx = H.ldlt().solve(b);

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        cout << "iternation  : " << t << " cost is equal to : " << cost << endl;
    }
    cout << "estimated paramters : a = " << ae << " , b = " << be << " , c = " << ce << endl;

    return 0;
}