cmake_minimum_required(VERSION 3.4.3)

project(imgproc_opencv)

include_directories(
  "/home/dkurtaev/opencv/build/include"
)
link_directories(
  "/home/dkurtaev/opencv/build/lib"
)

file(GLOB_RECURSE sources "*.cpp")

add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME}
  PRIVATE opencv_core
  PRIVATE opencv_imgproc
)
