cmake_minimum_required(VERSION 3.10)
project(NonLinearOptimization VERSION 0.1.0 LANGUAGES C CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W") # Add this flag
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})  # for cmake < 3.0
message(STATUS "Found Eigen3: ${Eigen3_DIR}")

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})  # for cmake < 3.0
message(STATUS "Found Ceres: ${Ceres_DIR}")

# OpenCv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # for cmake < 3.0
message(STATUS "Found OpenCV: ${OpenCV_DIR}")

# g2o
find_package(G2O)
include_directories(${G2O_INCLUDE_DIR})
link_directories(${G2O_LIB_DIR})

add_executable(gaussNewton src/gaussNewton.cpp)
target_link_libraries(gaussNewton Eigen3::Eigen)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting src/ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2oCurveFitting src/g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} fmt)