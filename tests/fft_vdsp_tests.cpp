#include <catch2/catch.hpp>
#include <complex>
#include <random>
#include <vector>
#include "ear/fft.hpp"

using namespace ear;

namespace {
  std::vector<float> random_signal(size_t n) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (auto &v : x) v = dist(rng);
    return x;
  }
}  // namespace

// The vDSP implementation must be a drop-in replacement for KISS: same
// n_fft/2+1 layout and same (unnormalised) scaling, since callers such as
// block_convolver normalise assuming it. Sizes include non-powers of two
// handled by vDSP (480, 960) and one that falls back to KISS (882).
TEST_CASE("vdsp_matches_kiss") {
  for (size_t n_fft : {64, 128, 480, 960, 1024, 4096, 882}) {
    CAPTURE(n_fft);
    auto plan_kiss = get_fft_kiss<float>().plan(n_fft);
    auto plan_vdsp = get_fft_vdsp<float>().plan(n_fft);
    auto wb_kiss = plan_kiss->alloc_workbuf();
    auto wb_vdsp = plan_vdsp->alloc_workbuf();

    std::vector<float> input = random_signal(n_fft);
    std::vector<float> input_copy = input;

    std::vector<std::complex<float>> spec_kiss(n_fft / 2 + 1);
    std::vector<std::complex<float>> spec_vdsp(n_fft / 2 + 1);
    plan_kiss->transform_forward(input.data(), spec_kiss.data(), *wb_kiss);
    plan_vdsp->transform_forward(input_copy.data(), spec_vdsp.data(),
                                 *wb_vdsp);

    float spec_tol = 1e-4f * static_cast<float>(n_fft);
    for (size_t i = 0; i < spec_kiss.size(); i++) {
      CAPTURE(i);
      CHECK(std::abs(spec_vdsp[i] - spec_kiss[i]) < spec_tol);
    }

    std::vector<float> out_kiss(n_fft), out_vdsp(n_fft);
    plan_kiss->transform_reverse(spec_kiss.data(), out_kiss.data(), *wb_kiss);
    plan_vdsp->transform_reverse(spec_kiss.data(), out_vdsp.data(), *wb_vdsp);

    for (size_t i = 0; i < n_fft; i++) {
      CAPTURE(i);
      // unnormalised round trip: n_fft times the original signal
      CHECK(out_kiss[i] == Approx(input[i] * n_fft).margin(1e-3 * n_fft));
      CHECK(out_vdsp[i] == Approx(out_kiss[i]).margin(1e-3 * n_fft));
    }
  }
}
