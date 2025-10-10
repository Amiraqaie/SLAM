cmake_minimum_required(VERSION 3.10)
project(eigen)

find_package(Eigen3)
find_package(Pangolin)

add_executable(run src/eigenMatrix.cpp)
add_executable(run2 src/useGeometry.cpp)
add_executable(run3 src/coordinateTransform.cpp)
add_executable(run4 src/plotTrajectory.cpp)

target_link_libraries(run4 ${Pangolin_LIBRARIES})