cmake_minimum_required(VERSION 3.0)
project(bundle_adjustment)
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

add_library(bal_common src/common.cpp)
add_executable(bundle_adjustment_g2o src/bundle_adjustment_g2o.cpp)
add_executable(bundle_adjustment_ceres src/bundle_adjustment_ceres.cpp)

target_link_libraries(bundle_adjustment_ceres 
    ${CERES_LIBRARIES} 
    bal_common 
    fmt::fmt
)

target_link_libraries(bundle_adjustment_g2o 
    ${G2O_LIBS}
    bal_common 
    fmt::fmt
    # Additional system libraries that might be needed
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)