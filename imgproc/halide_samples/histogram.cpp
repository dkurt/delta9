#include <opencv2/opencv.hpp>
#include <Halide.h>

void histogram(const uint8_t* src, int* dst, int height, int width);

void drawHist(const cv::Mat& hist, int height, int width);

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);

    cv::Mat frame;
    while (cv::waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat hist(3, 256, CV_32SC1);
        histogram(frame.data, (int*)hist.data, frame.rows, frame.cols);
        drawHist(hist, frame.rows, frame.cols);
        cv::imshow("frame", frame);
    }
}

void histogram(const uint8_t* src, int* dst, int height, int width)
{
    static Halide::Func f("hist");
    if (!f.defined())
    {
        Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved((uint8_t*)src, width, height, 3);

        Halide::Var x("x"), y("y"), c("c"), i("i");

        Halide::RDom r(0, width, 0, height);
        f(i, c) = 0;
        Halide::Expr lum = clamp(input(r.x, r.y, c), 0, 255);
        f(lum, c) += 1;

        f.estimate(f.args()[0], 0, 256).estimate(c, 0, 3);
        Halide::Pipeline(f).auto_schedule(Halide::get_host_target());

        f.print_loop_nest();
        f.compile_jit();
    }

    Halide::Buffer<int> output(dst, {256, 3});
    f.realize(output);
}

void drawHist(const cv::Mat& hist, int height, int width)
{
    float ratio = 1.0f / (height * width);
    cv::Scalar colors[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    cv::Mat color(100, 256, CV_8UC3);
    color.setTo(cv::Scalar(255, 255, 255));
    double max;
    cv::minMaxLoc(hist, 0, &max);
    for (int c = 0; c < 3; ++c)
    {
        for (int x = 0; x < 256; ++x)
        {
            int size = color.rows * (float)hist.at<int>(c, x) / max;
            color.colRange(x, x + 1).rowRange(color.rows - size, color.rows) = colors[c];
        }
    }
    cv::imshow("color", color);
}
