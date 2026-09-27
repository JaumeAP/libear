#include <catch2/catch.hpp>
#include <random>
#include <vector>
#include "ear/dsp/gain_interpolator.hpp"

using namespace ear::dsp;

namespace {
  // Independent scalar reference, not shared with gain_interpolator.hpp, so
  // this test is meaningful whichever branch (vDSP or scalar) that header
  // compiles to.
  void reference_apply_interp(const std::vector<std::vector<float>> &in,
                              std::vector<std::vector<float>> &out,
                              SampleIndex range_start, SampleIndex range_end,
                              SampleIndex block_start, SampleIndex start,
                              SampleIndex end,
                              const std::vector<std::vector<float>> &s,
                              const std::vector<std::vector<float>> &e) {
    float scale = 1.0f / (end - start);
    size_t n_out = s.empty() ? 0 : s[0].size();
    for (size_t oc = 0; oc < n_out; oc++)
      for (SampleIndex i = range_start; i < range_end; i++) out[oc][i] = 0.0f;

    for (size_t ic = 0; ic < s.size(); ic++)
      for (size_t oc = 0; oc < s[ic].size(); oc++)
        for (SampleIndex i = range_start; i < range_end; i++) {
          float p = (float)((block_start + i) - start) * scale;
          float gain = (1.0f - p) * s[ic][oc] + p * e[ic][oc];
          out[oc][i] += in[ic][i] * gain;
        }
  }

  std::vector<std::vector<float>> random_matrix(size_t rows, size_t cols,
                                                std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> m(rows, std::vector<float>(cols));
    for (auto &row : m)
      for (auto &v : row) v = dist(rng);
    return m;
  }
}  // namespace

TEST_CASE("LinearInterpMatrix apply_interp matches reference") {
  std::mt19937 rng(42);
  size_t n_in = 3, n_out = 5;
  size_t block_size = 517;  // deliberately not a power of two

  auto in = random_matrix(n_in, block_size, rng);
  std::vector<const float *> in_p;
  for (auto &row : in) in_p.push_back(row.data());

  auto s = random_matrix(n_in, n_out, rng);
  auto e = random_matrix(n_in, n_out, rng);

  std::vector<std::vector<float>> out_actual(n_out,
                                             std::vector<float>(block_size));
  std::vector<float *> out_actual_p;
  for (auto &row : out_actual) out_actual_p.push_back(row.data());

  std::vector<std::vector<float>> out_expected(n_out,
                                               std::vector<float>(block_size));

  LinearInterpMatrix::apply_interp(in_p.data(), out_actual_p.data(), 0,
                                   block_size, 0, 0, block_size, s, e);
  reference_apply_interp(in, out_expected, 0, block_size, 0, 0, block_size, s,
                         e);

  for (size_t oc = 0; oc < n_out; oc++)
    for (size_t i = 0; i < block_size; i++)
      CHECK(out_actual[oc][i] ==
            Approx(out_expected[oc][i]).margin(1e-5));
}

TEST_CASE("LinearInterpMatrix apply_constant matches reference") {
  std::mt19937 rng(43);
  size_t n_in = 4, n_out = 2;
  size_t block_size = 256;

  auto in = random_matrix(n_in, block_size, rng);
  std::vector<const float *> in_p;
  for (auto &row : in) in_p.push_back(row.data());

  auto gains = random_matrix(n_in, n_out, rng);

  std::vector<std::vector<float>> out(n_out, std::vector<float>(block_size));
  std::vector<float *> out_p;
  for (auto &row : out) out_p.push_back(row.data());

  LinearInterpMatrix::apply_constant(in_p.data(), out_p.data(), 0, block_size,
                                     gains);

  for (size_t oc = 0; oc < n_out; oc++)
    for (size_t i = 0; i < block_size; i++) {
      float expected = 0.0f;
      for (size_t ic = 0; ic < n_in; ic++)
        expected += in[ic][i] * gains[ic][oc];
      CHECK(out[oc][i] == Approx(expected).margin(1e-5));
    }
}
