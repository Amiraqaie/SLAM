cmake_minimum_required(VERSION 3.10.0)
project(AutSlam VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(slamlib src/camera.cpp src/feature.cpp)

target_link_libraries(slamlib
                       PUBLIC ${OpenCV_LIBS})

target_include_directories(slamlib
                            PUBLIC ${PROJECT_SOURCE_DIR}/include
                            PUBLIC ${EIGEN3_INCLUDE_DIR}
                            PUBLIC ${OpenCV_INCLUDE_DIR})
