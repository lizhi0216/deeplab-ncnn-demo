project(NCNN_DEMO)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)


set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/disk1/fxq/lianghua/ncnn/build/install")
find_package(ncnn   REQUIRED)
find_package(OpenCV REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(deeplab src/deeplab.cpp)
target_link_libraries(deeplab ncnn ${OpenCV_LIBS})