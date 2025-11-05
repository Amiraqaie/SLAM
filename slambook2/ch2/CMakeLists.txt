cmake_minimum_required(VERSION 3.5.0)
project(HelloSlam VERSION 0.1.0 LANGUAGES C CXX)

add_library(hello libHelloSLAM.cpp)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(main main.cpp)

add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello)

