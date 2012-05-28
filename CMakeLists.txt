cmake_minimum_required(VERSION 2.6)
project(tpofinder)

# General settings
set(BUILD_SHARED_LIBS true)
set(CMAKE_BUILD_TYPE DEBUG)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# OpenCV
find_package(OpenCV 2.4 REQUIRED)

# Boost
find_package(Boost COMPONENTS filesystem program_options system)

# Sources and headers
file(GLOB srcs src/*.cpp)
file(GLOB tsts test/test*.cpp)
include_directories("${PROJECT_SOURCE_DIR}/include")

# Library construction
add_library(${PROJECT_NAME} ${srcs})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# Executables
add_executable(model_homography apps/model_homography.cpp)
add_executable(sequence_homography apps/sequence_homography.cpp)
add_executable(invert_homography apps/invert_homography.cpp)
add_executable(webcam apps/webcam.cpp)
add_executable(tpofind apps/tpofind.cpp)
target_link_libraries(model_homography ${PROJECT_NAME})
target_link_libraries(sequence_homography ${PROJECT_NAME})
target_link_libraries(invert_homography ${PROJECT_NAME})
target_link_libraries(webcam ${PROJECT_NAME})
target_link_libraries(tpofind ${PROJECT_NAME})

# Testing
enable_testing()

# Set up gest; see also
# http://groups.google.com/group/googletestframework/browse_thread/thread/d1259032b859e5a3
# for a discussion on how to best integrate gtest with a project that uses
# cmake. Here, don't use FindGTest but rather ship gtest along with the source
# as a tar.gz which can be handled by cmake.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xvzf ${PROJECT_SOURCE_DIR}/gtest-1.6.0.tar.gz
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_subdirectory(${CMAKE_BINARY_DIR}/gtest-1.6.0)
include_directories(${CMAKE_BINARY_DIR}/gtest-1.6.0/include)
set(GTest_LIBRARIES gtest gtest_main)

add_executable(utest ${tsts})
target_link_libraries(utest ${GTest_LIBRARIES} ${PROJECT_NAME})
