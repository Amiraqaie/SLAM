cmake_minimum_required(VERSION 3.10.0)
project(AutSlam VERSION 0.1.0 LANGUAGES C CXX)

# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "-Wall")
# set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -fopenmp -pthread")


LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include_directories("/usr/include/eigen3")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Glog REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
find_package(G2O REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})
find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension g2o_types_slam3d
        ${GTEST_BOTH_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        ${GLOG_LIBRARIES}
        glog::glog
        ${DBoW3_LIB_DIR}
        /usr/local/lib/libDBoW3.so
        fmt
        )

add_library(slamlib 
            src/camera.cpp 
            src/feature.cpp 
            src/dataset.cpp 
            src/frame.cpp 
            src/mappoint.cpp 
            src/map.cpp 
            src/frontend.cpp
            src/backend.cpp
            src/config.cpp
            src/visual_odometry.cpp
            src/loop_closing.cpp
            src/pose_graph.cpp
            src/viewer_light.cpp)


target_link_libraries(slamlib
                       ${THIRD_PARTY_LIBS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(run_kitti_stereo app/run_kitti_stereo.cpp)
target_link_libraries(run_kitti_stereo slamlib ${THIRD_PARTY_LIBS})

add_executable(orb_cv src/orb_cv.cpp)
target_link_libraries(orb_cv slamlib ${THIRD_PARTY_LIBS})