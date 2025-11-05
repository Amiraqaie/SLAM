cmake_minimum_required(VERSION 3.10.0)
project(ch8 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

include_directories(src)

add_executable(optical_flow 
               src/optical_flow.cpp 
               src/implimentation.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES} ${Sophus_LIBRARIES} fmt)

add_executable(direct_method src/direct_method.cpp)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES} ${Sophus_LIBRARIES} fmt)