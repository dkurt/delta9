cmake_minimum_required(VERSION 3.4.3)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(
  ${CMAKE_SOURCE_DIR}
)

foreach(backend tbb halide opencv)
  add_subdirectory(${backend})

  add_executable(test_${backend} test.cpp imgproc.hpp)
  target_link_libraries(test_${backend}
    imgproc_${backend}
  )
endforeach()

add_subdirectory(halide_samples)
