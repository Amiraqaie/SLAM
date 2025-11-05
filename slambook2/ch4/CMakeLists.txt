cmake_minimum_required(VERSION 3.5.0)
project(SophusToturial VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W") # Add this flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})
find_package(Pangolin REQUIRED)


add_executable(useSophus useSophus.cpp)
add_executable(trajectoryError trajectoryError.cpp)

target_link_libraries(useSophus Eigen3::Eigen)
target_link_libraries(trajectoryError Eigen3::Eigen)
target_link_libraries(trajectoryError ${Pangolin_LIBRARIES})
