#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <Eigen/Core>
#include <random>

using namespace std;

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    CurveFittingVertex() {}
    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update)
    {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream &) override { return false; }
    virtual bool write(ostream &) const override { return false; }
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : _x(x) {}

    virtual void computeError() override
    {
        CurveFittingVertex *params = static_cast<CurveFittingVertex *>(vertex(0));
        const double a = params->estimate()(0);
        const double b = params->estimate()(1);
        const double c = params->estimate()(2);

        _error[0] = _measurement - exp(a * _x * _x + b * _x + c);
    }

    virtual bool read(istream &) override { return false; }
    virtual bool write(ostream &) const override { return false; }

private:
    double _x;
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
        double y = exp(a * x * x + b * x + c) + (double)dist(gen);

        x_data.push_back(x);
        y_data.push_back(y);
    }

    // // solve via g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Add Vertex
    CurveFittingVertex *params = new CurveFittingVertex();
    params->setId(0);
    params->setEstimate(Eigen::Vector3d(1.0, 1.0, 1.0));
    optimizer.addVertex(params);

    // Add Edges
    for (size_t i = 0; i < DATA_SIZE; i++)
    {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, params);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    cout << "Optimization result : " << params->estimate().transpose() << endl;
}