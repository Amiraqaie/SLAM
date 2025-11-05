find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)


add_executable(imageBasis imageBasis.cpp)
add_executable(undistoredImage undistoredImage.cpp)
add_executable(stereoVision stereoVision.cpp)

target_link_libraries(imageBasis ${OpenCV_LIBS})
target_link_libraries(undistoredImage ${OpenCV_LIBS})
target_link_libraries(stereoVision ${OpenCV_LIBS} ${EIGEN3_LIBS} ${Pangolin_LIBRARIES} ${PCL_LIBRARIES})