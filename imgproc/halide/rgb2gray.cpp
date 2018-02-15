#include "imgproc.hpp"

#include <Halide.h>

void rgb2gray(const uint8_t* src, uint8_t* dst, int height, int width)
{
    static const uint16_t R2GRAY = 77;
    static const uint16_t G2GRAY = 150;
    static const uint16_t B2GRAY = 29;

    static Halide::Func f("rgb2gray");
    if (!f.defined())
    {
        Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved((uint8_t*)src, width, height, 3);

        Halide::Var x("x"), y("y");
        Halide::Expr r = Halide::cast<uint16_t>(input(x, y, 0));
        Halide::Expr g = Halide::cast<uint16_t>(input(x, y, 1));
        Halide::Expr b = Halide::cast<uint16_t>(input(x, y, 2));
        f(x, y) = Halide::cast<uint8_t>((R2GRAY * r + G2GRAY * g + B2GRAY * b) >> 8);
        f.bound(x, 0, width).bound(y, 0, height);

        Halide::Var yo("yo"), yi("yi");
        f.split(y, yo, yi, 64)
         .reorder(x, yi, yo)
         .parallel(yo)
         .vectorize(x, 8);



        // f.estimate(x, 0, width).estimate(y, 0, height);
        // Halide::Pipeline(f).auto_schedule(Halide::get_host_target());

        f.print_loop_nest();

        f.compile_jit();
    }

    Halide::Buffer<uint8_t> output(dst, {width, height});
    f.realize(output);
}
