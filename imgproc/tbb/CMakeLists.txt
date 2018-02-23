cmake_minimum_required(VERSION 3.4.3)

project(imgproc_tbb)

include_directories(
  "/home/dkurtaev/tbb2018_20171205oss/include"
)
link_directories(
  "/home/dkurtaev/tbb2018_20171205oss/lib/intel64/gcc4.7"
)

file(GLOB_RECURSE sources "*.cpp")

add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME}
  PRIVATE tbb
)
