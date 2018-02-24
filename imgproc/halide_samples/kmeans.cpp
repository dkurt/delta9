#include <iostream>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <Halide.h>

void kmeans_halide_2d(const uint8_t* src, uint8_t* dst, int height, int width, int k, bool gpu);

void kmeans_halide(const uint8_t* src, uint8_t* dst, int height, int width, int k);

void kmeans(const uint8_t* src, uint8_t* dst, int height, int width, int k);

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);

    cv::Mat frameBGR;
    cv::Mat res;
    cv::Mat frameGray;
    timeval start, end;
    float bestTime = FLT_MAX;
    while (cv::waitKey(1) < 0)
    {
        cap >> frameBGR;
        if (frameBGR.empty())
            break;

        cvtColor(frameBGR, frameGray, CV_BGR2GRAY);

        res.create(frameGray.rows, frameGray.cols, CV_8UC1);

        gettimeofday(&start, 0);
        kmeans_halide_2d(frameGray.data, res.data, frameGray.rows, frameGray.cols, 5, false);
        gettimeofday(&end, 0);
        float t = (end.tv_sec - start.tv_sec) * 1e+3 + (end.tv_usec - start.tv_usec) * 1e-3;
        bestTime = std::min(bestTime, t);
        std::cout << bestTime << " ms" << std::endl;

        cv::namedWindow("Input", CV_WINDOW_NORMAL);
        cv::namedWindow("Output", CV_WINDOW_NORMAL);
        cv::imshow("Input", frameGray);
        cv::imshow("Output", res);
    }

    return 0;
}

void kmeans_halide(const uint8_t* src, uint8_t* dst, int height, int width, int k)
{
    static Halide::Func clustersFunc("clustersFunc");
    static Halide::Func kmeans("kmeans");
    static std::vector<uint8_t> clusters(k);
    static Halide::Buffer<uint8_t> clustersBuffer(&clusters[0], k);
    if (!kmeans.defined() || clusters.size() != k)
    {
        clusters.resize(k);
        clustersBuffer = Halide::Buffer<uint8_t>(&clusters[0], k);
        Halide::Buffer<uint8_t> input((uint8_t*)src, width * height);

        Halide::Func clustersMap("clustersMap");
        Halide::Var x("x"), i("i");
        Halide::RDom r(0, k);
        clustersMap(x) = argmin(abs(Halide::cast<int16_t>(input(x)) -
                                    Halide::cast<int16_t>(clustersBuffer(r))))[0];
        clustersMap.estimate(x, 0, width * height);

        // Update clusters.
        Halide::Func count("count");
        Halide::RDom im(input);
        count(i) = 0;
        Halide::Expr clusterId = clamp(clustersMap(im), 0, k - 1);
        count(clusterId) += 1;
        count.estimate(i, 0, k);

        Halide::Func s("s");
        s(i) = 0;
        s(clusterId) += Halide::cast<uint32_t>(input(im));
        s.estimate(i, 0, k);

        clustersFunc(i) = Halide::cast<uint8_t>(s(i) / max(count(i), 1));
        clustersFunc.estimate(i, 0, k);

        kmeans(x) = clustersFunc(clamp(clustersMap(x), 0, k - 1));
        kmeans.estimate(x, 0, width * height);

        Halide::Pipeline({clustersFunc, kmeans}).auto_schedule(Halide::get_host_target());
        kmeans.print_loop_nest();

        kmeans.compile_jit();
    }

    for (int i = 0; i < k; ++i)
    {
        clusters[i] = rand() % 256;
    }

    for (int i = 0; i < 14; ++i)
    {
        clustersFunc.realize(clustersBuffer);
    }

    Halide::Buffer<uint8_t> output(dst, width * height);
    kmeans.realize(output);
}

void kmeans_halide_2d(const uint8_t* src, uint8_t* dst, int height, int width, int k, bool gpu)
{
    static Halide::Func clustersFunc("clustersFunc");
    static Halide::Func kmeans("kmeans");
    static std::vector<uint8_t> clusters(k);
    static Halide::Buffer<uint8_t> clustersBuffer(&clusters[0], k);
    static Halide::Buffer<uint8_t> input((uint8_t*)src, width, height);
    if (!kmeans.defined() || clusters.size() != k)
    {
        clusters.resize(k);
        clustersBuffer = Halide::Buffer<uint8_t>(&clusters[0], k);

        Halide::Func clustersMap("clustersMap");
        Halide::Var x("x"), y("y"), i("i");
        Halide::RDom r(0, k);
        clustersMap(x, y) = argmin(abs(Halide::cast<int16_t>(input(x, y)) -
                                       Halide::cast<int16_t>(clustersBuffer(r))))[0];

        // Update clusters.
        Halide::RDom im(0, width, 0, height);
        Halide::RDom im_xs(0, width);
        Halide::RDom im_ys(0, height);

        Halide::Func count_ys("count_ys");
        count_ys(i, y) = 0;
        Halide::Expr clusterId = clamp(clustersMap(im_xs, y), 0, k - 1);
        count_ys(clusterId, y) += 1;

        Halide::Func count("count");
        count(i) = sum(count_ys(i, im_ys));

        Halide::Func s_ys("s_ys");
        s_ys(i, y) = 0;
        s_ys(clusterId, y) += Halide::cast<uint32_t>(input(im_xs, y));

        Halide::Func s("s");
        s(i) = sum(s_ys(i, im_ys));

        clustersFunc(i) = Halide::cast<uint8_t>(s(i) / max(count(i), 1));

        kmeans(x, y) = clustersFunc(clamp(clustersMap(x, y), 0, k - 1));

        Halide::Target t = Halide::get_host_target();
        if (gpu)
        {
            t.set_feature(Halide::Target::OpenCL);

            clustersFunc.bound(i, 0, k);
            kmeans.bound(x, 0, width).bound(y, 0, height);
            clustersMap.bound(x, 0, width).bound(y, 0, height);

            Halide::Var xo("xo"), xi("xi"), yo("yo"), yi("yi");
            clustersMap.split(x, xo, xi, 16)
                       .split(y, yo, yi, 16)
                       .reorder(xi, yi, xo, yo)
                       .gpu_blocks(xo, yo)
                       .gpu_threads(xi, yi);
            clustersMap.compute_root();

            count_ys.split(y, yo, yi, 16)
                    .reorder(yi, yo)
                    .gpu_blocks(yo)
                    .gpu_threads(yi);
            count_ys.compute_root();

            s_ys.split(y, yo, yi, 16)
                    .reorder(yi, yo)
                    .gpu_blocks(yo)
                    .gpu_threads(yi);
            s_ys.compute_root();

            clustersFunc.compute_root();
            clustersFunc.compile_jit(t);

            kmeans.split(x, xo, xi, 16)
                  .split(y, yo, yi, 16)
                  .reorder(xi, yi, xo, yo)
                  .gpu_blocks(xo, yo)
                  .gpu_threads(xi, yi);
        }
        else
        {
            clustersFunc.estimate(i, 0, k);
            kmeans.estimate(x, 0, width).estimate(y, 0, height);

            Halide::Pipeline({clustersFunc, kmeans}).auto_schedule(Halide::get_host_target());
        }
        kmeans.print_loop_nest();
        kmeans.compile_jit(t);
    }

    for (int i = 0; i < k; ++i)
    {
        clusters[i] = rand() % 256;
    }

    input.set_host_dirty();
    for (int i = 0; i < 14; ++i)
    {
        clustersFunc.realize(clustersBuffer);
    }

    Halide::Buffer<uint8_t> output(dst, width, height);
    kmeans.realize(output);
    if (gpu)
        output.copy_to_host();
}

void kmeans(const uint8_t* src, uint8_t* dst, int height, int width, int k)
{
    std::vector<uint8_t> clusters(k, 0);

    for (int i = 0; i < k; ++i)
    {
        clusters[i] = rand() % 256;
    }

    std::vector<int> means(k);
    std::vector<int> nums(k);
    for (int iter = 0; iter < 15; ++iter)
    {
        means.assign(k, 0);
        nums.assign(k, 0);

        for (int x = 0; x < height * width; ++x)
        {
            int clusterId = 0;
            uint8_t dist = abs(src[x] - clusters[0]);
            dst[x] = 0;
            for (int i = 1; i < k; ++i)
            {
                uint8_t newDist = abs(src[x] - clusters[i]);
                if (newDist < dist)
                {
                    dist = newDist;
                    clusterId = i;
                    dst[x] = i;
                }
            }
            means[clusterId] += src[x];
            nums[clusterId] += 1;
        }

        for (int i = 0; i < k; ++i)
        {
            if (nums[i])
                clusters[i] = means[i] / nums[i];
            else
                clusters[i] = 0;
        }
    }

    for (int x = 0; x < height * width; ++x)
    {
        dst[x] = clusters[dst[x]];
    }
}
