cmake_minimum_required(VERSION 3.10.0)
project(LieAlgebra VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(useSophus src/useSophus.cpp)
add_executable(trajectoryError src/trajectoryError.cpp)

target_link_libraries(trajectoryError ${Pangolin_LIBRARIES})
