#include <opencv2/opencv.hpp>
#include "imgproc.hpp"

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);

    cv::Mat frame;
    cv::Mat res;
    while (cv::waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
            break;

        res.create(frame.rows, frame.cols, CV_8UC3);
        boxFilter3x3(frame.data, res.data, frame.rows, frame.cols);

        cv::namedWindow("Input", CV_WINDOW_NORMAL);
        cv::namedWindow("Output", CV_WINDOW_NORMAL);
        cv::imshow("Input", frame);
        cv::imshow("Output", res);
    }

    return 0;
}
