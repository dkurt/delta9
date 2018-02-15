#include "imgproc.hpp"

#include <tbb/tbb.h>

void boxFilter3x3(const uint8_t* src, uint8_t* dst, int height, int width)
{
    static const float ratio = 1.f / 9;
    static const int rowStep = width * 3;

    // Upper and lower rows
    {
        const uint8_t* srcRow1 = src + rowStep;
        const uint8_t* srcUpper = src + rowStep * (height - 2);
        const uint8_t* srcLower = srcUpper + rowStep;
        uint8_t* dstLower = dst + rowStep * (height - 1);
        int offset;
        for (int x = 1; x < width - 1; ++x)
        {
            offset = x * 3;
            for (int c = 0; c < 3; ++c)
            {
                uint16_t s = (uint16_t)src[offset - 3 + c] +
                             (uint16_t)src[offset + c] +
                             (uint16_t)src[offset + 3 + c] +
                             (uint16_t)srcRow1[offset - 3 + c] +
                             (uint16_t)srcRow1[offset + c] +
                             (uint16_t)srcRow1[offset + 3 + c];
                dst[offset + c] = (uint8_t)(s * ratio);

                s = (uint16_t)srcUpper[offset - 3 + c] +
                    (uint16_t)srcUpper[offset + c] +
                    (uint16_t)srcUpper[offset + 3 + c] +
                    (uint16_t)srcLower[offset - 3 + c] +
                    (uint16_t)srcLower[offset + c] +
                    (uint16_t)srcLower[offset + 3 + c];
                dstLower[offset + c] = (uint8_t)(s * ratio);
            }
        }
    }

    // Left and right columns
    {
        const uint8_t* srcRowUpper = src;
        const uint8_t* srcRowMiddle = srcRowUpper + rowStep;
        const uint8_t* srcRowLower = srcRowMiddle + rowStep;
        uint8_t* dstRowUpper = dst;
        uint8_t* dstRowMiddle = dstRowUpper + rowStep;
        uint8_t* dstRowLower = dstRowMiddle + rowStep;
        for (int y = 1; y < height - 1; ++y)
        {
            for (int c = 0; c < 3; ++c)
            {
                uint16_t s = (uint16_t)srcRowUpper[c] + (uint16_t)srcRowUpper[3 + c] +
                             (uint16_t)srcRowMiddle[c] + (uint16_t)srcRowMiddle[3 + c] +
                             (uint16_t)srcRowLower[c] + (uint16_t)srcRowLower[3 + c];
                dstRowMiddle[c] = (uint8_t)(s * ratio);

                s = (uint16_t)srcRowUpper[rowStep - 6 + c] + (uint16_t)srcRowUpper[rowStep - 3 + c] +
                    (uint16_t)srcRowMiddle[rowStep - 6 + c] + (uint16_t)srcRowMiddle[rowStep - 3 + c] +
                    (uint16_t)srcRowLower[rowStep - 6 + c] + (uint16_t)srcRowLower[rowStep - 3 + c];
                dstRowMiddle[rowStep - 3 + c] = (uint8_t)(s * ratio);
            }
            srcRowUpper = srcRowMiddle;
            srcRowMiddle = srcRowLower;
            srcRowLower += rowStep;
            dstRowUpper = dstRowMiddle;
            dstRowMiddle = dstRowLower;
            dstRowLower += rowStep;
        }
    }

    // Corners.
    {
        int lastRow = rowStep * (height - 1);
        for (int c = 0; c < 3; ++c)
        {
            dst[c] = (uint8_t)(((uint16_t)src[c] +
                                (uint16_t)src[3 + c] +
                                (uint16_t)src[rowStep + c] +
                                (uint16_t)src[rowStep + 3 + c]) * ratio);
            dst[rowStep - 3 + c] = (uint8_t)(((uint16_t)src[rowStep - 6 + c] +
                                              (uint16_t)src[rowStep - 3 + c] +
                                              (uint16_t)src[2 * rowStep - 6 + c] +
                                              (uint16_t)src[2 * rowStep - 3 + c]) * ratio);
            dst[lastRow + c] = (uint8_t)(((uint16_t)src[lastRow + c] +
                                          (uint16_t)src[lastRow + 3 + c] +
                                          (uint16_t)src[lastRow - rowStep + c] +
                                          (uint16_t)src[lastRow - rowStep + 3 + c]) * ratio);
            dst[lastRow + rowStep - 3 + c] = (uint8_t)(((uint16_t)src[lastRow + rowStep - 3 + c] +
                                                       (uint16_t)src[lastRow + rowStep - 6 + c] +
                                                       (uint16_t)src[lastRow - 3 + c] +
                                                       (uint16_t)src[lastRow - 6 + c]) * ratio);
        }
    }

    tbb::parallel_for(
        tbb::blocked_range2d<int>(1, height - 1, 1, width - 1),
        [&](tbb::blocked_range2d<int> r) {
            int y, x, offset;

            int ybegin = r.rows().begin(), yend = r.rows().end();
            int xbegin = r.cols().begin(), xend = r.cols().end();

            const uint8_t* srcRowMiddle = src + ybegin * rowStep;
            const uint8_t* srcRowUpper = srcRowMiddle - rowStep;
            const uint8_t* srcRowLower = srcRowMiddle + rowStep;

            uint8_t* dstRowMiddle = dst + ybegin * rowStep;
            uint8_t* dstRowUpper = dstRowMiddle - rowStep;
            uint8_t* dstRowLower = dstRowMiddle + rowStep;

            for (y = ybegin; y < yend; ++y)
            {
                for (x = xbegin; x < xend; ++x)
                {
                    offset = x * 3;
                    for (int c = 0; c < 3; ++c)
                    {
                        uint16_t s = (uint16_t)srcRowUpper[offset - 3 + c] +
                                     (uint16_t)srcRowUpper[offset + c] +
                                     (uint16_t)srcRowUpper[offset + 3 + c] +
                                     (uint16_t)srcRowMiddle[offset - 3 + c] +
                                     (uint16_t)srcRowMiddle[offset + c] +
                                     (uint16_t)srcRowMiddle[offset + 3 + c] +
                                     (uint16_t)srcRowLower[offset - 3 + c] +
                                     (uint16_t)srcRowLower[offset + c] +
                                     (uint16_t)srcRowLower[offset + 3 + c];
                        dstRowMiddle[offset + c] = (uint8_t)(s * ratio);
                    }
                }
                srcRowUpper = srcRowMiddle;
                srcRowMiddle = srcRowLower;
                srcRowLower += rowStep;
                dstRowUpper = dstRowMiddle;
                dstRowMiddle = dstRowLower;
                dstRowLower += rowStep;
            }
        }
    );
}
