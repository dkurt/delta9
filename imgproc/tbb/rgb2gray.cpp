#include "imgproc.hpp"

#include <tbb/tbb.h>

void rgb2gray(const uint8_t* src, uint8_t* dst, int height, int width)
{
    static const uint16_t R2GRAY = 77;
    static const uint16_t G2GRAY = 150;
    static const uint16_t B2GRAY = 29;

    tbb::parallel_for(
        tbb::blocked_range<int>(0, height * width),
        [&](tbb::blocked_range<int> r) {

            uint16_t red, green, blue;
            int begin = r.begin();
            int end = r.end();
            const uint8_t* __restrict__ srcData = src;
            uint8_t* __restrict__ dstData = dst;
            for (int i = begin; i < end; ++i)
            {
                red = srcData[i * 3];
                green = srcData[i * 3 + 1];
                blue = srcData[i * 3 + 2];
                dstData[i] = (R2GRAY * red + G2GRAY * green + B2GRAY * blue) >> 8;
            }
        }
    );
}
