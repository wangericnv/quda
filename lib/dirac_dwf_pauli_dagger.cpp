#include <iostream>
#include <dirac_quda.h>
#include <blas_quda.h>

namespace quda {

  DiracDwfPauliDagger::DiracDwfPauliDagger(const DiracParam &param) : 
    DiracDomainWall(param),
    dwf_op(param),
    pv_dag_op(param)
  {
    pv_dag_op.setMass(1.0);
    pv_dag_op.flipDagger();
  }

  DiracDwfPauliDagger::DiracDwfPauliDagger(const DiracDwfPauliDagger &dirac) :
    DiracDomainWall(dirac),
    dwf_op(dirac.dwf_op),
    pv_dag_op(dirac.pv_dag_op)
  {
  }

  DiracDwfPauliDagger::~DiracDwfPauliDagger() { }

  DiracDwfPauliDagger& DiracDwfPauliDagger::operator=(const DiracDwfPauliDagger &dirac)
  {
    if (&dirac != this) {
      DiracDomainWall::operator=(dirac);
      dwf_op = dirac.dwf_op;
      pv_dag_op = dirac.pv_dag_op;
    }
    return *this;
  }

  void DiracDwfPauliDagger::Dslash(ColorSpinorField &out, const ColorSpinorField &in, 
                                   const QudaParity parity) const
  {
    bool reset = newTmp(&tmp1, in);

    dwf_op.Dslash(*tmp1, in, parity);
    pv_dag_op.Dslash(out, *tmp1, parity);

    deleteTmp(&tmp1, reset);
  }

  void DiracDwfPauliDagger::DslashXpay(ColorSpinorField &out, const ColorSpinorField &in, 
                                       const QudaParity parity, const ColorSpinorField &x,
                                       const double &k) const
  {
    bool reset = newTmp(&tmp1, in);

    dwf_op.Dslash(*tmp1, in, parity);
    pv_dag_op.DslashXpay(out, *tmp1, parity, in, k);

    deleteTmp(&tmp1, reset);
  }

  void DiracDwfPauliDagger::M(ColorSpinorField &out, const ColorSpinorField &in) const
  {
    checkFullSpinor(out, in);

    bool reset = newTmp(&tmp1, in);

    dwf_op.M(*tmp1, in);
    pv_dag_op.Mdag(out, *tmp1);

    deleteTmp(&tmp1, reset);
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

    pv_dag_op.Dslash(*src, b, QUDA_INVALID_PARITY);
    sol = &x;
  }

  void DiracDwfPauliDagger::reconstruct(ColorSpinorField &x, const ColorSpinorField &b,
				    const QudaSolutionType solType) const
  {
    // do nothing
  }

} // namespace quda
