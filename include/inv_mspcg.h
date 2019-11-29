#ifndef _MSPCG_H
#define _MSPCG_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
#include <vector>

#include <invert_quda.h>

#include <madwf_ml.h>

namespace quda
{
  using Tp = madwf_ml::TrainingParameter<float>;

  class MSPCG : public Solver
  { // Multisplitting Preconditioned CG

  private:
    Solver *solver_prec;
    SolverParam solver_prec_param;

    DiracMobiusPC *mat;
    DiracMobiusPC *mat_sloppy;
    DiracMobiusPC *mat_precondition;
    DiracMobiusPC *mat_precondition_truncated;

    DiracMdagM *nrm_op;
    DiracMdagM *nrm_op_sloppy;
    DiracMdagMLocal *nrm_op_precondition;
    DiracMdagMLocal *nrm_op_precondition_truncated;

    DiracParam dirac_param;
    DiracParam dirac_param_sloppy;
    DiracParam dirac_param_precondition;

    cudaGaugeField *padded_gauge_field;
    cudaGaugeField *padded_gauge_field_precondition;

    std::array<int, 4> R;

    cudaColorSpinorField *vct_dr;
    cudaColorSpinorField *vct_dp;
    cudaColorSpinorField *vct_dmmp;
    cudaColorSpinorField *vct_dtmp;
    cudaColorSpinorField *vct_dtmp2;

    cudaColorSpinorField *r;
    cudaColorSpinorField *x;
    cudaColorSpinorField *p;
    cudaColorSpinorField *z;
    cudaColorSpinorField *mmp;
    cudaColorSpinorField *tmp;
    cudaColorSpinorField *tmp2;

    cudaColorSpinorField *r_old;

    cudaColorSpinorField *fr;
    cudaColorSpinorField *fz;

    cudaColorSpinorField *immp;
    cudaColorSpinorField *ip;

#ifdef PIPELINED_PRECONDITIONER
    cudaColorSpinorField *iz;
    cudaColorSpinorField *is;
    cudaColorSpinorField *iw;
    cudaColorSpinorField *iq;
#endif
    cudaColorSpinorField *ifmmp;
    cudaColorSpinorField *ifp;
    cudaColorSpinorField *iftmp;
    cudaColorSpinorField *ifset;

    Timer copier_timer;
    Timer preconditioner_timer;
    Timer sloppy_timer;
    Timer precise_timer;
    Timer linalg_timer[2];

    int sp_len2, sp_len1, sp_len0;
    int RR2[4], RR1[4], RR0[4];
    int_fastdiv Xs2[4], Xs1[4], Xs0[4];

    int shift0[4] = {0, 0, 0, 0};
    int shift1[4] = {1, 1, 1, 1};
    int shift2[4] = {2, 2, 2, 2};

    bool tc;

    void pipelined_inner_cg(ColorSpinorField &ix, ColorSpinorField &ib);
    void Minv(ColorSpinorField &out, const ColorSpinorField &in);

    void m_inv_trained(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap);

    void inner_cg(ColorSpinorField &ix, ColorSpinorField &ib);
    int outer_cg(ColorSpinorField &dx, ColorSpinorField &db, double quit);

  public:
    /* --------------------------------------------------------------------------*
     *
     * --------------------------------------------------------------------------*/
    MSPCG(QudaInvertParam *inv_param, SolverParam &param_, TimeProfile &profile);

    ~MSPCG();

    int inner_iterations;
    double Gflops;
    double fGflops;

    void allocate(ColorSpinorField &db);
    void deallocate();

    void inner_dslash(ColorSpinorField &out, const ColorSpinorField &in);

    void operator()(ColorSpinorField &out, ColorSpinorField &in) { this->mspcg_madwf_ml(out, in, false, false, 4); }
    void mspcg_madwf_ml(ColorSpinorField &out, ColorSpinorField &in, const bool use_training,
                        const bool perform_training, const int Ls_cheap);

    void train_param(const std::vector<ColorSpinorField *> &in, std::vector<float> &tp, const double mu, int Ls_cheap);

    void calculate_TdATx(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap);

    double calculate_chi(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp, double mu, int Ls_cheap);

    void ATx(ColorSpinorField &out, const ColorSpinorField &in, const Tp &tp);
  };

} // namespace quda

#endif
