cmake_minimum_required(VERSION 3.10)
project(BasicVision VERSION 0.1.0 LANGUAGES C CXX)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W") # Add this flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imageBasics src/imageBasics.cpp)
add_executable(undistortImage src/undistortImage.cpp)
add_executable(stereoVision src/stereoVision.cpp)
add_executable(jointMap src/jointMap.cpp)


target_link_libraries(imageBasics ${OpenCV_LIBS})
target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(stereoVision ${OpenCV_LIBS})
target_link_libraries(stereoVision Eigen3::Eigen)
target_link_libraries(stereoVision ${Pangolin_LIBRARIES})
target_link_libraries(jointMap ${OpenCV_LIBS})
target_link_libraries(jointMap Eigen3::Eigen)
target_link_libraries(jointMap ${Pangolin_LIBRARIES})