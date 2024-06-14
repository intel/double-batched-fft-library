#include "fft1d.hpp"
#include "bbfft/cl/error.hpp"

#include <stdexcept>

using namespace bbfft;

fft1d::fft1d(bbfft::configuration const &cfg, cl_command_queue queue, cl_context context,
             cl_device_id device)
    : queue_(queue) {
    if (cfg.dim != 1 || cfg.shape[0] != 1 || cfg.dir == direction::backward || cfg.callbacks) {
        throw std::runtime_error("Unsupported configuration");
    }

    bool inplace = false;
    if (cfg.istride == default_istride(1, cfg.shape, cfg.type, true) &&
        cfg.ostride == default_ostride(1, cfg.shape, cfg.type, true)) {
        inplace = true;
    } else if (cfg.istride == default_istride(1, cfg.shape, cfg.type, false) &&
               cfg.ostride == default_ostride(1, cfg.shape, cfg.type, false)) {
        inplace = false;
    } else {
        throw std::runtime_error("Non-default strides are unsupported");
    }

    auto N = cfg.shape[1];
    bool is_r2c = false;
    if (cfg.type == transform_type::r2c) {
        N /= 2;
        is_r2c = true;
    }
    const auto factorization = factor(N, 3);

    // FFT plans

    cl_int err;
    std::array<std::size_t, max_tensor_dim> shape = {};
    std::array<std::size_t, max_tensor_dim> istride = {};
    std::array<std::size_t, max_tensor_dim> ostride = {};
    auto cfg1d = configuration{};
    char source[1024];
    std::size_t length = 0;
    char const *real_type = get_real_type(cfg.fp);

    char const store_template[] = R"OpenCL(
typedef %s real_t;
typedef %s2 complex_t;
void store(global complex_t* out, size_t offset, complex_t val, global void*) {
    int N0 = %d;
    int N1 = %d;
    int N2 = %d;
    size_t n1 = offset / N0 %% N1;
    size_t n2 = offset / (N0 * N1) %% N2;
    real_t arg = -((real_t) 6.28318530717958647693) / (N1 * N2) * n2 * n1;
    complex_t tw = (complex_t) (native_cos(arg), native_sin(arg));
    out[offset] = (complex_t) (val.x * tw.x - val.y * tw.y, val.x * tw.y + val.y * tw.x) / N2;
})OpenCL";
    length = snprintf(source, sizeof(source), store_template, real_type, real_type,
                      factorization[0], factorization[1], factorization[2]);

    shape = {factorization[0] * factorization[1], factorization[2], cfg.shape[2]};
    istride = default_istride(1, shape, transform_type::c2c, inplace);
    ostride = default_ostride(1, shape, transform_type::c2c, inplace);
    if (is_r2c && inplace) {
        ++istride[2];
    }
    const auto N_stride = istride[2];
    cfg1d = configuration{1,
                          shape,
                          cfg.fp,
                          direction::forward,
                          transform_type::c2c,
                          istride,
                          ostride,
                          {source, length, nullptr, "store"}};
    plans_.emplace_back(make_plan(cfg1d, queue_));

    char const store_template2[] = R"OpenCL(
typedef %s real_t;
typedef %s2 complex_t;
void store(global complex_t* out, size_t offset, complex_t val, global void*) {
    int N0 = %d;
    int N1 = %d;
    int N2 = %d;
    size_t n0 = offset %% N0;
    size_t n1 = offset / N0 %% N1;
    size_t n2 = offset / (N0 * N1) %% N2;
    real_t arg = -((real_t) 6.28318530717958647693) / (N0 * N1 * N2) * n0 * (n2 + n1 * N2);
    complex_t tw = (complex_t) (native_cos(arg), native_sin(arg));
    out[offset] = (complex_t) (val.x * tw.x - val.y * tw.y, val.x * tw.y + val.y * tw.x) / N1;
})OpenCL";
    length = snprintf(source, sizeof(source), store_template2, real_type, real_type,
                      factorization[0], factorization[1], factorization[2]);

    shape = {factorization[0], factorization[1], factorization[2] * cfg.shape[2]};
    istride = default_istride(1, shape, transform_type::c2c, inplace);
    ostride = default_ostride(1, shape, transform_type::c2c, inplace);
    cfg1d = configuration{1,
                          shape,
                          cfg.fp,
                          direction::forward,
                          transform_type::c2c,
                          istride,
                          ostride,
                          {source, length, nullptr, "store"}};
    plans_.emplace_back(make_plan(cfg1d, queue_));

    char const store_template3[] = R"OpenCL(
typedef %s2 complex_t;
void store(global complex_t* out, size_t offset, complex_t val, global void*) {
    out[offset] = val / %d;
})OpenCL";
    length = snprintf(source, sizeof(source), store_template3, real_type, factorization[0]);

    shape = {1, factorization[0], factorization[1] * factorization[2] * cfg.shape[2]};
    istride = default_istride(1, shape, transform_type::c2c, inplace);
    ostride = default_ostride(1, shape, transform_type::c2c, inplace);
    cfg1d = configuration{1,
                          shape,
                          cfg.fp,
                          direction::forward,
                          transform_type::c2c,
                          istride,
                          ostride,
                          {source, length, nullptr, "store"}};
    plans_.emplace_back(make_plan(cfg1d, queue_));

    const std::size_t sizeof_real = static_cast<std::size_t>(cfg.fp);
    buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                             2 * sizeof_real * N_stride * cfg.shape[2], nullptr, &err);
    CL_CHECK(err);

    // Bit reversal

    char const header_template[] = R"OpenCL(
typedef %s real_t;
typedef %s2 complex_t;
#define N0 %d
#define N1 %d
#define N2 %d
#define TILE_SIZE %d
#define COL_THREADS %d
)OpenCL";

    /*
     * Bit reversal: y_{n2,n1,n0,k} = x_{n0,n1,n2,k}
     *
     * slice = offset:size
     *
     * for b0=0:tile_size:N0, b2=0:tile_size:N2
     *   tmp = x(b0:tile_size, n1, b2:tile_size, k)
     *   y(b2:tile_size, n1, b0:tile_size, k) = tmp^T
     *
     */
    static char const kernels[] = R"OpenCL(
kernel
__attribute__((reqd_work_group_size(TILE_SIZE, COL_THREADS, 1)))
void bit_reversal(global complex_t* input, global complex_t* output) {
    const size_t N = N0 * N1 * N2;
    local complex_t tmp[TILE_SIZE * TILE_SIZE];
    size_t l0 = get_local_id(0);
    size_t l2 = get_local_id(1);
    size_t g0 = get_group_id(0) * TILE_SIZE + l0;
    size_t g2 = get_group_id(1) * TILE_SIZE + l2;
    size_t k = get_global_id(2) / N1;
    size_t n1 = get_global_id(2) % N1;

    for (int j = 0; j < TILE_SIZE; j += COL_THREADS) {
        if (g0 < N0 && g2 + j < N2) {
            tmp[l0 + (l2 + j) * TILE_SIZE] = input[g0 + n1 * N0 + (g2 + j) * N0 * N1 + N * k];
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    g2 = get_group_id(1) * TILE_SIZE + l0;
    g0 = get_group_id(0) * TILE_SIZE + l2;
    for (int j = 0; j < TILE_SIZE; j += COL_THREADS) {
        if (g2 < N2 && g0 + j < N0) {
            output[g2 + n1 * N2 + (g0 + j) * N2 * N1 + N * k] = tmp[(l2 + j) + l0 * TILE_SIZE];
        }
    }
}

kernel void r2c_post(global complex_t* input, global complex_t* output) {
    const size_t N = N0 * N1 * N2;
    const size_t N_stride = N + 1;
    size_t n = get_global_id(0);
    size_t n_other = N - n;
    size_t k = get_global_id(1);
    
    size_t n_load = n % N;
    size_t n_other_load = n_other % N;
    
    complex_t y1 = input[n_load + k * N];
    complex_t y2 = input[n_other_load + k * N];
    y2 = (complex_t) (y2.x, -y2.y);
    // We usually need to divide by 2, but so far we normalized by N/2 only
    // so we add the missing factor 2 here
    complex_t a = (y2 + y1) / 4;
    complex_t b = (y2 - y1) / 4;
    real_t arg = -(((real_t) 6.28318530717958647693) / (2 * N)) * n;
    complex_t tw_i = (complex_t) (-native_sin(arg), native_cos(arg));
    b = (complex_t) (b.x * tw_i.x - b.y * tw_i.y, b.x * tw_i.y + b.y * tw_i.x);
    output[n + k * N_stride] = a + b;
    output[n_other + k * N_stride] = (complex_t) (a.x - b.x, b.y - a.y);
    if (n == 0) { // write n == N/2 case when n ==0
        output[N / 2 + k * N_stride] = input[N / 2 + k * N];
    }
}
)OpenCL";

    constexpr int tile_size = 64;
    constexpr int col_threads = 128 / tile_size;

    length = snprintf(source, sizeof(source), header_template, real_type, real_type,
                      factorization[0], factorization[1], factorization[2], tile_size, col_threads);

    char const *code[] = {source, kernels};
    const std::size_t lengths[] = {length, sizeof(kernels)};
    program_ = clCreateProgramWithSource(context, 2, code, lengths, &err);
    CL_CHECK(err);
    err = clBuildProgram(program_, 1, &device, "-cl-std=CL2.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string log;
        std::size_t log_size;
        CL_CHECK(
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        log.resize(log_size);
        CL_CHECK(clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                       nullptr));
        throw std::runtime_error(log.c_str());
    }

    bit_reversal_ = clCreateKernel(program_, "bit_reversal", &err);
    bit_reversal_gws_ = {factorization[0], 1 + (factorization[2] - 1) / tile_size * col_threads,
                         factorization[1] * cfg.shape[2]};
    bit_reversal_lws_ = {tile_size, col_threads, 1};
    CL_CHECK(err);
    if (is_r2c) {
        r2c_post_ = clCreateKernel(program_, "r2c_post", &err);
        CL_CHECK(err);
    } else {
        r2c_post_ = nullptr;
    }
    r2c_post_gws_ = {N / 2, cfg.shape[2]};
}

fft1d::~fft1d() {
    clReleaseMemObject(buffer_);
    clReleaseKernel(bit_reversal_);
    if (r2c_post_) {
        clReleaseKernel(r2c_post_);
    }
    clReleaseProgram(program_);
}

auto fft1d::execute(cl_mem in, cl_mem out, std::vector<cl_event> const &dep_events) -> cl_event {
    const bool is_r2c = r2c_post_ != nullptr;
    auto plan2_out = is_r2c ? out : buffer_;
    auto bit_reversal_out = is_r2c ? buffer_ : out;

    cl_event e0 = plans_[0].execute(in, buffer_, dep_events);
    cl_event e1 = plans_[1].execute(buffer_, out, e0);
    cl_event e2 = plans_[2].execute(out, plan2_out, e1);

    cl_event e3;
    CL_CHECK(clSetKernelArg(bit_reversal_, 0, sizeof(cl_mem), &plan2_out));
    CL_CHECK(clSetKernelArg(bit_reversal_, 1, sizeof(cl_mem), &bit_reversal_out));
    CL_CHECK(clEnqueueNDRangeKernel(queue_, bit_reversal_, 3, nullptr, bit_reversal_gws_.data(),
                                    bit_reversal_lws_.data(), 1, &e2, &e3));

    if (is_r2c) {
        cl_event e4;
        CL_CHECK(clSetKernelArg(r2c_post_, 0, sizeof(cl_mem), &bit_reversal_out));
        CL_CHECK(clSetKernelArg(r2c_post_, 1, sizeof(cl_mem), &out));
        CL_CHECK(clEnqueueNDRangeKernel(queue_, r2c_post_, 2, nullptr, r2c_post_gws_.data(),
                                        nullptr, 1, &e3, &e4));
        CL_CHECK(clReleaseEvent(e3));
        e3 = e4;
    }
    CL_CHECK(clReleaseEvent(e2));
    CL_CHECK(clReleaseEvent(e1));
    CL_CHECK(clReleaseEvent(e0));
    return e3;
}

auto fft1d::get_real_type(precision p) -> char const * {
    char const *real_type = "";
    switch (p) {
    case precision::f32:
        real_type = "float";
        break;
    case precision::f64:
        real_type = "double";
        break;
    }
    return real_type;
}
