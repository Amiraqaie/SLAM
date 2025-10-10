cmake_minimum_required(VERSION 3.10)
project(NonLinearOptimization)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(G2O_LIBS 
    g2o_core
    g2o_stuff
    g2o_types_sba
    g2o_types_slam3d
    g2o_solver_csparse
    g2o_csparse_extension
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(G2O REQUIRED)

add_executable(gaussNewton src/gaussNewton.cpp)
add_executable(ceresCurveFitting src/ceresCurveFitting.cpp)
add_executable(g2oCurveFitting src/g2oCurveFitting.cpp)

target_link_libraries(ceresCurveFitting ${CERES_LIBRARIES})
target_link_libraries(g2oCurveFitting ${CERES_LIBRARIES} ${G2O_LIBS} fmt)