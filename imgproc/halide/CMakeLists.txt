cmake_minimum_required(VERSION 3.4.3)

project(imgproc_halide)

include_directories(
  "/home/dkurtaev/Halide/build/include"
)
link_directories(
  "/home/dkurtaev/Halide/build/lib"
)

file(GLOB_RECURSE sources "*.cpp")

add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME}
  PRIVATE Halide
)
