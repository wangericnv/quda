#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracDwfPauliDagger::DiracDwfPauliDagger(const DiracParam &param) :
      DiracWilson(param, 5),
      m5(param.m5),
      kappa5(0.5 / (5.0 + m5)),
      Ls(param.Ls)
  {
  }

  DiracDwfPauliDagger::DiracDwfPauliDagger(const DiracDwfPauliDagger &dirac) :
      DiracWilson(dirac),
      m5(dirac.m5),
      kappa5(0.5 / (5.0 + m5)),
      Ls(dirac.Ls)
  {
  }

  DiracDwfPauliDagger::~DiracDwfPauliDagger() { }

  DiracDwfPauliDagger& DiracDwfPauliDagger::operator=(const DiracDwfPauliDagger &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      m5 = dirac.m5;
      kappa5 = dirac.kappa5;
    }
    return *this;
  }

  void DiracDwfPauliDagger::checkDWF(const ColorSpinorField &out, const ColorSpinorField &in) const
  {
    if (in.Ndim() != 5 || out.Ndim() != 5) errorQuda("Wrong number of dimensions\n");
    if (in.X(4) != Ls) errorQuda("Expected Ls = %d, not %d\n", Ls, in.X(4));
    if (out.X(4) != Ls) errorQuda("Expected Ls = %d, not %d\n", Ls, out.X(4));
  }

  void DiracDwfPauliDagger::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
			       const QudaParity parity) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    bool reset = newTmp(&tmp1, in);

    ApplyDomainWall5D(*tmp, in, *gauge, 0.0, mass, in, parity, dagger, commDim, profile);

    ApplyDomainWall5D(out, *tmp, *gauge, 0.0, 1.0, in, parity, !dagger, commDim, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += 1320LL*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDwfPauliDagger::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
				   const QudaParity parity, const ColorSpinorField &x,
				   const double &k) const
  {
    checkDWF(out, in);
    checkParitySpinor(in, out);
    checkSpinorAlias(in, out);

    Dslash(out, in, parity);
    blas::caxpy(k, x, out);

    long long Ls = in.X(4);
    long long bulk = (Ls-2)*(in.Volume()/Ls);
    long long wall = 2*in.Volume()/Ls;
    flops += (1320LL+48LL)*(long long)in.Volume() + 96LL*bulk + 120LL*wall;
  }

  void DiracDwfPauliDagger::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    Dslash(out, in, parity);

    //ApplyDomainWall5D(out, in, *gauge, 0.0, mass, in, QUDA_INVALID_PARITY, dagger, commDim, profile);

    long long Ls = in.X(4);
    long long bulk = (Ls - 2) * (in.Volume() / Ls);
    long long wall = 2 * in.Volume() / Ls;
    flops += (1320LL + 48LL) * (long long)in.Volume() + 96LL * bulk + 120LL * wall;
  }

  void DiracDwfPauliDagger::MdagM(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    M(*tmp1, in);
    Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
  }

  void DiracDwfPauliDagger::prepare(ColorSpinorField* &src, ColorSpinorField* &sol,
				ColorSpinorField &x, ColorSpinorField &b, 
				const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    ApplyDomainWall5D(out, in, *gauge, 0.0, 1.0, in, !dagger, commDim, profile);


    src = &b;
    sol = &x;
  }

  void DiracDwfPauliDagger::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				    const QudaSolutionType solType) const
  {
    // do nothing
  }


} // namespace quda
