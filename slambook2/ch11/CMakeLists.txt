cmake_minimum_required(VERSION 3.0)
project( loop_closure )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(DBoW3_INCLUDE_DIRS "/usr/local/include" )
set(DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

add_executable( feature_training src/feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( loop_closure src/loop_closure.cpp )
target_link_libraries( loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS} )

# add_executable( gen_vocab gen_vocab_large.cpp )
# target_link_libraries( gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )