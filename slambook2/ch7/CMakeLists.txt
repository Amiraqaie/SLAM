cmake_minimum_required(VERSION 3.0)
project(vo1)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

find_package(Eigen3 REQUIRED)
include_directories(
    ${Eigen3_INCLUDE_DIRS}
)

find_package(Pangolin REQUIRED)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIR})

# g2o
find_package(G2O)
include_directories(${G2O_INCLUDE_DIR})
link_directories(${G2O_LIB_DIR})

add_executable(orb_cv src/orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS} ${Eigen3_LIBS})

add_executable(triangulation src/triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS} ${Eigen3_LIBS} ${Pangolin_LIBRARIES})

add_executable(pose_estimation_3d2d src/pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${Eigen3_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt) 

add_executable(pose_estimation_3d3d src/pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS} ${Eigen3_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt)