cmake_minimum_required(VERSION 3.0)
project(pose_graph)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find all required packages
find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(CSparse REQUIRED)
find_package(fmt REQUIRED)

# Set g2o libraries - IMPORTANT: Include ALL required g2o components
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

include_directories(
    ${PROJECT_SOURCE_DIR} 
    ${EIGEN3_INCLUDE_DIR} 
    ${CSPARSE_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
)

add_executable(pose_graph_g2o_SE3 src/pose_graph_g2o_SE3.cpp)
add_executable(pose_graph_g2o_lie_algebra src/pose_graph_g2o_lie_algebra.cpp)
add_executable(pose_graph_ceres src/pose_graph_ceres.cpp)

target_link_libraries(pose_graph_g2o_SE3 
    ${G2O_LIBS}
    fmt::fmt
    # Additional system libraries that might be needed
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_link_libraries(pose_graph_g2o_lie_algebra 
    ${G2O_LIBS}
    fmt::fmt
    # Additional system libraries that might be needed
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

target_link_libraries(pose_graph_ceres
  ceres
  Eigen3::Eigen
  Sophus::Sophus
)