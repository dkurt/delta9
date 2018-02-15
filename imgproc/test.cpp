#include <float.h>
#include <sys/time.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

#include "imgproc.hpp"

int width = 1920;
int height = 1280;

void test_rgb2gray()
{
    std::vector<uint8_t> src(width * height * 3);
    std::vector<uint8_t> dst(width * height);
    std::generate(src.begin(), src.end(), std::rand);

    uint8_t* srcData = &src[0];
    uint8_t* dstData = &dst[0];

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        rgb2gray(srcData, dstData, height, width);
    }

    timeval start, end;
    float bestTime = FLT_MAX;
    for (int i = 0; i < 100; ++i)
    {
        std::generate(src.begin(), src.end(), std::rand);

        gettimeofday(&start, 0);
        rgb2gray(srcData, dstData, height, width);
        gettimeofday(&end, 0);

        float t = (end.tv_sec - start.tv_sec) * 1e+3 + (end.tv_usec - start.tv_usec) * 1e-3;
        bestTime = std::min(bestTime, t);
    }
    std::cout << "Best time: " << bestTime << " ms" << std::endl;

    int maxDiff = 0;
    int diff = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int offset = (y * width + x) * 3;
            uint8_t target = (uint8_t)(0.299f * srcData[offset + 0] +
                                       0.587f * srcData[offset + 1] +
                                       0.114f * srcData[offset + 2]);
            uint8_t out = dstData[y * width + x];
            maxDiff = std::max(maxDiff, abs(target - out));
            diff += abs(target - out);
        }
    }
    std::cout << "Part of different pixels: " << (float)diff / (width * height) << std::endl;
    std::cout << "Maximal absolute difference: " << maxDiff << std::endl;
}

void test_boxFilter3x3()
{
    std::vector<uint8_t> src(width * height * 3);
    std::vector<uint8_t> dst(width * height * 3);
    std::generate(src.begin(), src.end(), std::rand);

    uint8_t* srcData = &src[0];
    uint8_t* dstData = &dst[0];

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        boxFilter3x3(srcData, dstData, height, width);
    }

    timeval start, end;
    float bestTime = FLT_MAX;
    for (int i = 0; i < 100; ++i)
    {
        std::generate(src.begin(), src.end(), std::rand);

        gettimeofday(&start, 0);
        boxFilter3x3(srcData, dstData, height, width);
        gettimeofday(&end, 0);

        float t = (end.tv_sec - start.tv_sec) * 1e+3 + (end.tv_usec - start.tv_usec) * 1e-3;
        bestTime = std::min(bestTime, t);
    }
    std::cout << "Best time: " << bestTime << " ms" << std::endl;

    int maxDiff = 0;
    int diff = 0;
    const float ratio = 1.f / 9;
    const int rowStep = width * 3;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int offset = (y * width + x) * 3;
            for (int c = 0; c < 3; ++c)
            {
                float target_fp32 = src[offset + c];
                if (x > 0)
                    target_fp32 += src[offset - 3 + c];
                if (x < width - 1)
                    target_fp32 += src[offset + 3 + c];

                if (y > 0)
                {
                    target_fp32 += src[offset - rowStep + c];
                    if (x > 0)
                        target_fp32 += src[offset - rowStep - 3 + c];
                    if (x < width - 1)
                        target_fp32 += src[offset - rowStep + 3 + c];
                }
                if (y < height - 1)
                {
                    target_fp32 += src[offset + rowStep + c];
                    if (x > 0)
                        target_fp32 += src[offset + rowStep - 3 + c];
                    if (x < width - 1)
                        target_fp32 += src[offset + rowStep + 3 + c];
                }
                uint8_t target = (uint8_t)(target_fp32 * ratio);
                uint8_t out = dstData[offset + c];
                maxDiff = std::max(maxDiff, abs(target - out));
                diff += abs(target - out);
            }
        }
    }
    std::cout << "Part of different pixels: " << (float)diff / (width * height * 3) << std::endl;
    std::cout << "Maximal absolute difference: " << maxDiff << std::endl;
}

int main(int argc, char** argv)
{
    std::srand(324);

    if (argc == 1)
    {
        std::cout << "Chose an algorithm from set [rgb2gray, box_filter]" << std::endl;
        return 0;
    }

    if (std::string(argv[1]) == "rgb2gray")
        test_rgb2gray();
    else if (std::string(argv[1]) == "box_filter")
        test_boxFilter3x3();
    else
        std::cout << "Unknown test " << argv[1] << std::endl;

    return 0;
}
