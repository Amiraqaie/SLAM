cmake_minimum_required(VERSION 3.5.0)
project(EigenToturial VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W") # Add this flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
find_package(Pangolin REQUIRED)

add_executable(EigenToturial main.cpp)
add_executable(useGeometry useGeometry.cpp)
add_executable(coordinateTransform coordinateTransform.cpp)
add_executable(plotTrajectory plotTrajectory.cpp)

target_link_libraries(EigenToturial Eigen3::Eigen)
target_link_libraries(useGeometry Eigen3::Eigen)
target_link_libraries(coordinateTransform Eigen3::Eigen)
target_link_libraries(plotTrajectory Eigen3::Eigen)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})