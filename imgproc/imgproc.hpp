#ifndef INCLUDE_DELTA9_IMGPROC_HPP_
#define INCLUDE_DELTA9_IMGPROC_HPP_

#include <stdint.h>

void rgb2gray(const uint8_t* src, uint8_t* dst, int height, int width);

void boxFilter3x3(const uint8_t* src, uint8_t* dst, int height, int width);

#endif
