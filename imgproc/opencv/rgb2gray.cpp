#include "imgproc.hpp"

#include <opencv2/opencv.hpp>

void rgb2gray(const uint8_t* src, uint8_t* dst, int height, int width)
{
    cv::Mat input(height, width, CV_8UC3, (void*)src);
    cv::Mat output(height, width, CV_8UC1, (void*)dst);
    cv::cvtColor(input, output, cv::COLOR_RGB2GRAY);
}
