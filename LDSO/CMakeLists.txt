cmake_minimum_required(VERSION 3.0)
project(ldso_ngd)

# check for the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
    message(STATUS "Setting the build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# check for the c++ Standard
if($(CMAKE_VERSION) VERSION_LESS "3.8")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Custom modules
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Binary and library
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# required libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Pangolin REQUIRED)

# optional libraries
find_package(LibZip QUIET)

# set additional compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-unused-variable -Wno-unused-function -march=native")
if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Decision if we have ZipLib
if(LIBZIP_LIBRARY)
    message(STATUS "LibZip found, enabling zip file support.")
    add_definitions(-DHAS_ZIPLIB=1)
    include_directories(${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF})
else()
    message(STATUS "LibZip not found, disabling zip file support.")
    set(LIBZIP_LIBRARY "")
endif()