#include "imgproc.hpp"

#include <opencv2/opencv.hpp>

void boxFilter3x3(const uint8_t* src, uint8_t* dst, int height, int width)
{
    cv::Mat input(height, width, CV_8UC3, (void*)src);
    cv::Mat output(height, width, CV_8UC3, (void*)dst);
    cv::boxFilter(input, output, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
}
