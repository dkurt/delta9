#include "imgproc.hpp"

#include <Halide.h>

void boxFilter3x3(const uint8_t* src, uint8_t* dst, int height, int width)
{
    static Halide::Func f("box_filter");
    if (!f.defined())
    {
        Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved((uint8_t*)src, width, height, 3);
        Halide::Func padded = Halide::BoundaryConditions::constant_exterior(input, 0);

        Halide::Var x("x"), y("y"), c("c");
        Halide::Func input_uint16("input_uint16");

        input_uint16(x, y, c) = Halide::cast<uint16_t>(padded(x, y, c));

        Halide::RDom r(-1, 3, -1, 3);
        Halide::Expr s = sum(input_uint16(x + r.x, y + r.y, c));

        float ratio = 1.0f / 9;
        f(x, y, c) = Halide::cast<uint8_t>(s * ratio);

        f.output_buffer().dim(0).set_stride(3).set_bounds(0, width);
        f.output_buffer().dim(1).set_stride(3 * width).set_bounds(0, height);
        f.output_buffer().dim(2).set_stride(1).set_bounds(0, 3);

        f.estimate(x, 0, width).estimate(y, 0, height).estimate(c, 0, 3);
        Halide::Pipeline(f).auto_schedule(Halide::get_host_target());

        f.print_loop_nest();

        f.compile_jit();
    }

    Halide::Buffer<uint8_t> output = Halide::Buffer<uint8_t>::make_interleaved(dst, width, height, 3);
    f.realize(output);
}
