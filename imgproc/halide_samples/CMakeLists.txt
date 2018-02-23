cmake_minimum_required(VERSION 3.4.3)

project(halide_samples)

file(GLOB_RECURSE sources "*.cpp")

include_directories(
  "/home/dkurtaev/opencv/build/include"
)
link_directories(
  "/home/dkurtaev/opencv/build/lib"
)

foreach(file ${sources})
  get_filename_component(name ${file} NAME_WE)

  add_executable(sample_${name} ${file})
  target_link_libraries(sample_${name}
    imgproc_halide
    opencv_core
    opencv_highgui
    opencv_videoio
    opencv_imgproc
  )
endforeach()
