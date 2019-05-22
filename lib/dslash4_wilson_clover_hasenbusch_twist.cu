#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson_clover.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda {

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct WilsonCloverHasenbuschTwistLaunch {
	  static constexpr const char *kernel = "quda::wilsonCloverHasenbuschTwistGPU"; // Kernel name for jit
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream) {
      static_assert(xpay == true, "wilsonCloverHasenbuschTwist operator only defined for xpay");
      dslash.launch(wilsonCloverHasenbuschTwistGPU<Float, nDim, nColor, nParity, dagger, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class WilsonCloverHasenbuschTwist : public Dslash<Float> {

  protected:
    Arg &arg;
    const ColorSpinorField &in;

  public:

    WilsonCloverHasenbuschTwist(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in)
      : Dslash<Float>(arg, out, in,"kernels/dslash_wilson_clover.cuh"),
		arg(arg),
		in(in) { }

    virtual ~WilsonCloverHasenbuschTwist() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay) Dslash<Float>::template instantiate<WilsonCloverHasenbuschTwistLaunch,nDim,nColor,true>(tp, arg, stream);
      else errorQuda("Wilson-clover - Hasenbusch Twist operator only defined for xpay=true");
    }

    long long flops() const {
      int clover_flops = 504;
      long long flops = Dslash<Float>::flops();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:
	break; // all clover flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
    	  flops += clover_flops * in.Volume();
    	  if( arg.twist ) {
    		  // -mu * (i gamma_5 A) (A x)
    		  flops += ((clover_flops+48)*in.Volume());
    	  }
	break;
      }
      return flops;
    }

    long long bytes() const {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in.Precision() + (isFixed ? 2*sizeof(float) : 0);

      long long bytes = Dslash<Float>::bytes();
      switch(arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL:	break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes*in.Volume(); break;
      }

      return bytes;
    }

    TuneKey tuneKey() const
    {
    	return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverHasenbuschTwistApply {

     inline WilsonCloverHasenbuschTwistApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
         double a, double b, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
     {
       constexpr int nDim = 4;
       using ArgType = WilsonCloverArg<Float,nColor,recon,true>;
       ArgType arg(out, in, U, A, a, b, x, parity, dagger, comm_override);
       WilsonCloverHasenbuschTwist<Float,nDim,nColor,ArgType > wilson(arg, out, in);

       dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
		   in.VolumeCB(), in.GhostFaceCB(), profile);
       policy.apply(0);

       checkCudaError();
     }

  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwist(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
					const CloverField &A, double a, double b, const ColorSpinorField &x, int parity, bool dagger,
			 const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_CLOVER_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, A);

    // check all locations match
    checkLocation(out, in, U, A);

    instantiate<WilsonCloverHasenbuschTwistApply>(out, in, U, A, a, b, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Clover Hasensbuch Twist dslash has not been built");
#endif

  }





} // namespace quda

#endif