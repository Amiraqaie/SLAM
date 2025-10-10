#include <iostream>
#include <ceres/ceres.h>
#include <random>

using namespace std;

struct ceresCurveFitting
{
    const double _x, _y;

    ceresCurveFitting(double x, double y) : _x(x), _y(y) {}

    template <typename T>
    bool operator()(const T* const abc, T* residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        return true; 
    }
};


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

    // Add ceres Residual blocks
    double abc[3] = {0, 0, 0};
    ceres::Problem problem;
    for (size_t i = 0; i < DATA_SIZE; i++)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ceresCurveFitting, 1, 3>(new ceresCurveFitting(x_data[i], y_data[i])),
            new ceres::HuberLoss(1.0),
            abc
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;
    cout << "optimized parameters = " << abc[0] << " " << abc[1] << " " << abc[2] << endl;

    return 0;
}