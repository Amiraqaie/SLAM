#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o" << endl;
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // set up g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    int vertexCnt = 0, edgeCnt = 0;
    while (!fin.eof())
    {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT")
        {
            g2o::VertexSE3 *v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT")
        {
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            int index1 = 0, index2 = 0;
            fin >> index1 >> index2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertex(index1));
            e->setVertex(1, optimizer.vertex(index2));
            e->read(fin);
            optimizer.addEdge(e);
        }
        if (!fin.good())
            break;
    }

    cout << "vertex cnt: " << vertexCnt << endl;
    cout << "edge cnt: " << edgeCnt << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    optimizer.save("result.g2o");
    return 0;
    
}