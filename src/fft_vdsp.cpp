#include <Accelerate/Accelerate.h>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>
#include "ear/export.hpp"
#include "ear/fft.hpp"
#include "ear/helpers/assert.hpp"

namespace ear {

  // Accelerate/vDSP implementation of the FFT interface.
  //
  // - Uses the vDSP DFT API (vDSP_DFT_zrop), which supports n_fft = f * 2^k
  //   for f in {1, 3, 5, 15}. This covers the usual host block sizes (powers
  //   of two, 240, 480, 960...). For any other size, plan() falls back to the
  //   KISS implementation, so every even n_fft keeps working.
  //
  // - vDSP packs the real N/2 term into the imaginary part of the first
  //   output; this is unpacked into the common n_fft/2+1 format.
  //
  // - vDSP's forward real transform is scaled by 2 relative to the plain DFT
  //   computed by the KISS implementation; the forward output is halved so
  //   that both implementations are interchangeable (callers such as
  //   block_convolver normalise by 1/n_fft assuming an unscaled round trip).

  class WorkBufVdsp : public FFTWorkBuf {
   public:
    explicit WorkBufVdsp(size_t n_half)
        : in_real(n_half), in_imag(n_half), out_real(n_half), out_imag(n_half) {}

    std::vector<float> in_real;
    std::vector<float> in_imag;
    std::vector<float> out_real;
    std::vector<float> out_imag;
  };

  class FFTPlanVdsp : public FFTPlan<float> {
   public:
    using Complex = typename FFTPlan<float>::Complex;

    FFTPlanVdsp(size_t n_fft, vDSP_DFT_Setup setup_forward,
                vDSP_DFT_Setup setup_reverse)
        : n_fft(n_fft),
          setup_forward(setup_forward),
          setup_reverse(setup_reverse) {}

    ~FFTPlanVdsp() override {
      vDSP_DFT_DestroySetup(setup_forward);
      vDSP_DFT_DestroySetup(setup_reverse);
    }

    void transform_forward(float *input, Complex *output,
                           FFTWorkBuf &workbuf) const override {
      WorkBufVdsp &wb = dynamic_cast<WorkBufVdsp &>(workbuf);
      size_t n_half = n_fft / 2;

      DSPSplitComplex split_in = {wb.in_real.data(), wb.in_imag.data()};
      vDSP_ctoz(reinterpret_cast<const DSPComplex *>(input), 2, &split_in, 1,
                n_half);
      vDSP_DFT_Execute(setup_forward, wb.in_real.data(), wb.in_imag.data(),
                       wb.out_real.data(), wb.out_imag.data());

      output[0] = Complex(0.5f * wb.out_real[0], 0.0f);
      output[n_half] = Complex(0.5f * wb.out_imag[0], 0.0f);
      for (size_t i = 1; i < n_half; i++)
        output[i] = Complex(0.5f * wb.out_real[i], 0.5f * wb.out_imag[i]);
    }

    void transform_reverse(Complex *input, float *output,
                           FFTWorkBuf &workbuf) const override {
      WorkBufVdsp &wb = dynamic_cast<WorkBufVdsp &>(workbuf);
      size_t n_half = n_fft / 2;

      wb.in_real[0] = input[0].real();
      wb.in_imag[0] = input[n_half].real();
      for (size_t i = 1; i < n_half; i++) {
        wb.in_real[i] = input[i].real();
        wb.in_imag[i] = input[i].imag();
      }
      vDSP_DFT_Execute(setup_reverse, wb.in_real.data(), wb.in_imag.data(),
                       wb.out_real.data(), wb.out_imag.data());

      DSPSplitComplex split_out = {wb.out_real.data(), wb.out_imag.data()};
      vDSP_ztoc(&split_out, 1, reinterpret_cast<DSPComplex *>(output), 2,
                n_half);
    }

    std::unique_ptr<FFTWorkBuf> alloc_workbuf() const override {
      return std::unique_ptr<FFTWorkBuf>(new WorkBufVdsp(n_fft / 2));
    }

   private:
    size_t n_fft;
    vDSP_DFT_Setup setup_forward;
    vDSP_DFT_Setup setup_reverse;
  };

  class FFTVdsp : public FFTImpl<float> {
   public:
    std::shared_ptr<FFTPlan<float>> plan(size_t n_fft) const override {
      ear_assert(n_fft % 2 == 0, "n_fft must be even");

      vDSP_DFT_Setup setup_forward = vDSP_DFT_zrop_CreateSetup(
          nullptr, static_cast<vDSP_Length>(n_fft), vDSP_DFT_FORWARD);
      vDSP_DFT_Setup setup_reverse =
          setup_forward ? vDSP_DFT_zrop_CreateSetup(
                              setup_forward, static_cast<vDSP_Length>(n_fft),
                              vDSP_DFT_INVERSE)
                        : nullptr;

      if (!setup_forward || !setup_reverse) {
        if (setup_forward) vDSP_DFT_DestroySetup(setup_forward);
        // size not supported by vDSP; KISS handles every even size
        return get_fft_kiss<float>().plan(n_fft);
      }
      return std::make_shared<FFTPlanVdsp>(n_fft, setup_forward, setup_reverse);
    }
  };

  template <>
  FFTImpl<float> EAR_EXPORT &get_fft_vdsp<float>() {
    static FFTVdsp fft;
    return fft;
  }

}  // namespace ear
