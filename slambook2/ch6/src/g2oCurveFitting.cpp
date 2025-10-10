#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream &is) override { return true; }
    virtual bool write(std::ostream &os) const override { return true; }
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double _x;  // x value, given
    CurveFittingEdge(double x) : _x(x) {}

    virtual void computeError() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y_pred = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _error(0, 0) = _measurement - y_pred;
    }

    virtual bool read(std::istream& is) override { return true; }
    virtual bool write(std::ostream& os) const override { return true; }
};

int main()
{
    double a = 1.0, b = 2.0, c = 1.0;
    int N = 100;
    double w_sigma = 1.0;
    std::vector<std::pair<double, double>> data;

    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0; // x in [0,1]
        double y = exp(a * x * x + b * x + c);
        y += (rand() % 1000 / 1000.0 - 0.5) * 2 * w_sigma; // add noise
        data.emplace_back(x, y);
    }

    for (size_t i = 0; i < data.size(); i += 10)
        std::cout << "data : " << i << " has x : " << data[i].first << " and has y : " << data[i].second << std::endl;

    // Set up g2o optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  // 3 parameters, 1 measurement
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // Dense solver

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
    ));
    optimizer.setVerbose(true); // print optimization info

    // Add the vertex (parameters a, b, c)
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(0, 0, 0)); // Initial guess
    v->setId(0);
    optimizer.addVertex(v);

    // Add edges (one per data point)
    int id = 1;
    for (auto& d : data) {
        CurveFittingEdge* edge = new CurveFittingEdge(d.first); // x
        edge->setId(id++);
        edge->setVertex(0, v); // connect edge to vertex
        edge->setMeasurement(d.second); // y
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // information matrix = 1
        optimizer.addEdge(edge);
    }

    // Run the optimization
    optimizer.initializeOptimization();
    optimizer.optimize(100);  // 10 iterations

    // Output result
    Eigen::Vector3d abc_estimate = v->estimate();
    std::cout << "Estimated parameters: " << abc_estimate.transpose() << std::endl;
}
