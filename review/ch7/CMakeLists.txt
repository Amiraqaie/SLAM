cmake_minimum_required(VERSION 3.10)
project(feature)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)

add_executable(orb_cv src/orb_cv.cpp)
add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)
add_executable(triangulation src/triangulation.cpp)
add_executable(pose_estimation_2d3d src/pose_estimation_2d3d.cpp)
add_executable(test src/test.cpp)

target_link_libraries(orb_cv ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d3d ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt)
target_link_libraries(test ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt)
target_link_libraries(triangulation ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Pangolin_LIBRARIES})