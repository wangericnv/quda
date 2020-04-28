#pragma once

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

namespace quda
{
  // Forward declarations for the JD eigensolver
  class GCR;
  class Solver;

  // Local enum for the LU axpy block type
  enum blockType { PENCIL, LOWER_TRI, UPPER_TRI };

  class EigenSolver
  {
    using range = std::pair<int, int>;

protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;
    QudaEigParam *eig_param;
    TimeProfile &profile;

    // Problem parameters
    //------------------
    int nEv;             /** Size of initial factorisation */
    int nKr;             /** Size of Krylov space after extension */
    int m_min;            /** Minimim size of subspace for Jacobi-Davidson */
    int m_max;            /** Maximum size of subspace for Jacobi-Davidson */
    int k_max;            /** NUmber of requested eigenpairs for Jacobi-Davidson */
    int corr_eq_maxiter; /** Maximum number of iterations for the correction equation */
    double corr_eq_tol;  /** Tolerance for the correction equation */
    int nConv;           /** Number of converged eigenvalues requested */
    double tol;          /** Tolerance on eigenvalues */
    bool reverse;        /** True if using polynomial acceleration */
    char spectrum[3];    /** Part of the spectrum to be computed */

    // Algorithm variables
    //--------------------
    bool converged;
    int restart_iter;
    int max_restarts;
    int check_interval;
    int batched_rotate;
    int iter;
    int iter_converged;
    int iter_locked;
    int iter_keep;
    int num_converged;
    int num_locked;
    int num_keep;

    double *residua;

    // Device side vector workspace
    std::vector<ColorSpinorField *> r;
    std::vector<ColorSpinorField *> rSloppy;
    std::vector<ColorSpinorField *> rPrecon;
    std::vector<ColorSpinorField *> kSpaceSloppy;
    std::vector<ColorSpinorField *> kSpacePrecon;
    std::vector<ColorSpinorField *> d_vecs_tmp;

    ColorSpinorField *tmp1;
    ColorSpinorField *tmp2;

public:
  /**
     @brief Constructor for base Eigensolver class
     @param mat Operator of desired precision
     @param matSloppy Operator of sloppy precision
     @param matPrecon Operator of preconditioner precision
     @param eig_param MGParam struct that defines all meta data
     @param profile Timeprofile instance used to profile
  */
  EigenSolver(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
              QudaEigParam *eig_param, TimeProfile &profile);

  /**
     Destructor for EigenSolver class.
  */
  virtual ~EigenSolver();

  /**
     @return Whether the solver is only for Hermitian systems
   */
  virtual bool hermitian() = 0;

  /**
     @brief Computes the eigen decomposition for the operator passed to create.
     @param kSpace The converged eigenvectors
     @param evals The converged eigenvalues
   */
  virtual void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals) = 0;

  /**
     @brief Creates the eigensolver using the parameters given and the matrix.
     @param eig_param The eigensolver parameters
     @param mat The operator of desired precision to solve
     @param matSloppy The operator of sloppy precision
     @param matPrecon The operator of precondionter precision
     @param profile Time Profile
  */
  static EigenSolver *create(QudaEigParam *eig_param, const DiracMatrix &mat, const DiracMatrix &matSloppy,
                             const DiracMatrix &matPrecon, TimeProfile &profile);

  /**
     @brief Applies the specified matVec operation:
     M, Mdag, MMdag, MdagM
     @param[in] mat Matrix operator
     @param[in] out Output spinor
     @param[in] in Input spinor
  */
  void matVec(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in);

  /**
     @brief Promotes the specified matVec operation:
     M, Mdag, MMdag, MdagM to a Chebyshev polynomial
     @param[in] mat Matrix operator
     @param[in] out Output spinor
     @param[in] in Input spinor
  */
  void chebyOp(const DiracMatrix &mat, ColorSpinorField &out, const ColorSpinorField &in);

  /**
     @brief Estimate the spectral radius of the operator for the max value of the
     Chebyshev polynomial
     @param[in] mat Matrix operator
     @param[in] out Output spinor
     @param[in] in Input spinor
  */
  double estimateChebyOpMax(const DiracMatrix &mat, ColorSpinorField &out, ColorSpinorField &in);

  /**
     @brief Orthogonalise input vector r against
     vector space v using block-BLAS
     @param[out] Sum of inner products
     @param[in] v Vector space
     @param[in] r Vector to be orthogonalised
     @param[in] j Number of vectors in v to orthogonalise against
  */
  Complex blockOrthogonalize(std::vector<ColorSpinorField *> v, std::vector<ColorSpinorField *> r, int j);

  /**
     @brief Change the precision of the Krylov space
     @param[in] vecs The Krylov space to be copied
     @param[in] vecs_new The Krylov Space of new precision
     @param[in] prec_new The new precision
  */
  void precChangeKrylov(std::vector<ColorSpinorField *> &vecs, std::vector<ColorSpinorField *> &vecs_new,
                        QudaPrecision prec_new);

  /**
     @brief Permute the vector space using the permutation matrix.
     @param[in/out] kSpace The current Krylov space
     @param[in] mat Eigen object storing the pivots
     @param[in] size The size of the (square) permutation matrix
  */
  void permuteVecs(std::vector<ColorSpinorField *> &kSpace, int *mat, int size);

  /**
     @brief Rotate part of kSpace
     @param[in/out] kSpace The current Krylov space
     @param[in] array The rotation matrix
     @param[in] rank row rank of array
     @param[in] is Start of i index
     @param[in] ie End of i index
     @param[in] js Start of j index
     @param[in] je End of j index
     @param[in] blockType Type of caxpy(_U/L) to perform
  */
  void blockRotate(std::vector<ColorSpinorField *> &kSpace, double *array, int rank, const range &i, const range &j,
                   blockType b_type);

  /**
     @brief Copy temp part of kSpace, zero out for next use
     @param[in/out] kSpace The current Krylov space
     @param[in] js Start of j index
     @param[in] je End of j index
  */
  void blockReset(std::vector<ColorSpinorField *> &kSpace, int js, int je);

  /**
     @brief Test for an initial guess
     @param[in/out] in Initial vector
  */
  void testInitGuess(ColorSpinorField *&in);

  /**
     @brief Deflate a set of source vectors with a given eigenspace
     @param[in] sol The resulting deflated vector set
     @param[in] src The source vector set we are deflating
     @param[in] evecs The eigenvectors to use in deflation
     @param[in] evals The eigenvalues to use in deflation
     @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
  */
  void deflate(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &src,
               const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
               bool accumulate = false) const;

  /**
     @brief Deflate a given source vector with a given eigenspace.
     This is a wrapper variant for a single source vector.
     @param[in] sol The resulting deflated vector
     @param[in] src The source vector we are deflating
     @param[in] evecs The eigenvectors to use in deflation
     @param[in] evals The eigenvalues to use in deflation
     @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
  */
  void deflate(ColorSpinorField &sol, const ColorSpinorField &src, const std::vector<ColorSpinorField *> &evecs,
               const std::vector<Complex> &evals, bool accumulate = false)
  {
    // FIXME add support for mixed-precison dot product to avoid this copy
    if (src.Precision() != evecs[0]->Precision() && !tmp1) {
      ColorSpinorParam param(*evecs[0]);
      tmp1 = ColorSpinorField::Create(param);
    }
    ColorSpinorField *src_tmp = src.Precision() != evecs[0]->Precision() ? tmp1 : const_cast<ColorSpinorField *>(&src);
    blas::copy(*src_tmp, src); // no-op if these alias
    std::vector<ColorSpinorField *> src_ {src_tmp};
    std::vector<ColorSpinorField *> sol_ {&sol};
    deflate(sol_, src_, evecs, evals, accumulate);
  }

    /**
       @brief Deflate a set of source vectors with a set of left and
       right singular vectors
       @param[in] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The singular vectors to use in deflation
       @param[in] evals The singular values to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflateSVD(std::vector<ColorSpinorField *> &sol, const std::vector<ColorSpinorField *> &vec,
                    const std::vector<ColorSpinorField *> &evecs, const std::vector<Complex> &evals,
                    bool accumulate = false) const;

    /**
       @brief Deflate a a given source vector with a given with a set of left and
       right singular vectors  This is a wrapper variant for a single source vector.
       @param[in] sol The resulting deflated vector set
       @param[in] src The source vector set we are deflating
       @param[in] evecs The singular vectors to use in deflation
       @param[in] evals The singular values to use in deflation
       @param[in] accumulate Whether to preserve the sol vector content prior to accumulating
    */
    void deflateSVD(ColorSpinorField &sol, const ColorSpinorField &src, const std::vector<ColorSpinorField *> &evecs,
                    const std::vector<Complex> &evals, bool accumulate = false)
    {
      // FIXME add support for mixed-precison dot product to avoid this copy
      if (src.Precision() != evecs[0]->Precision() && !tmp1) {
        ColorSpinorParam param(*evecs[0]);
        tmp1 = ColorSpinorField::Create(param);
      }
      ColorSpinorField *src_tmp = src.Precision() != evecs[0]->Precision() ? tmp1 : const_cast<ColorSpinorField *>(&src);
      blas::copy(*src_tmp, src); // no-op if these alias
      std::vector<ColorSpinorField *> src_ {src_tmp};
      std::vector<ColorSpinorField *> sol_ {&sol};
      deflateSVD(sol_, src_, evecs, evals, accumulate);
    }

    /**
       @brief Computes Left/Right SVD from pre computed Right/Left
       @param[in] mat Matrix operator
       @param[in] evecs Computed eigenvectors of NormOp
       @param[in] evals Computed eigenvalues of NormOp
    */
    void computeSVD(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals);

    /**
       @brief Compute eigenvalues and their residiua
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
       @param[in] size The number of eigenvalues to compute
    */
    void computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals,
                      int size);

    /**
       @brief Compute eigenvalues and their residiua.  This variant compute the number of converged eigenvalues.
       @param[in] mat Matrix operator
       @param[in] evecs The eigenvectors
       @param[in] evals The eigenvalues
    */
    void computeEvals(const DiracMatrix &mat, std::vector<ColorSpinorField *> &evecs, std::vector<Complex> &evals)
    {
      computeEvals(mat, evecs, evals, nConv);
    }

    /**
       @brief Load vectors from file
       @param[in] eig_vecs The eigenvectors to load
       @param[in] file The filename to load
    */
    static void loadVectors(std::vector<ColorSpinorField *> &eig_vecs, std::string file);

    /**
       @brief Save vectors to file
       @param[in] eig_vecs The eigenvectors to save
       @param[in] file The filename to save
    */
    static void saveVectors(const std::vector<ColorSpinorField *> &eig_vecs, std::string file);

    /**
       @brief Load and check eigenpairs from file
       @param[in] mat Matrix operator
       @param[in] eig_vecs The eigenvectors to save
       @param[in] file The filename to save
    */
    void loadFromFile(const DiracMatrix &mat, std::vector<ColorSpinorField *> &eig_vecs, std::vector<Complex> &evals);
  };

  /**
     @brief Thick Restarted Lanczos Method.
  */
  class TRLM : public EigenSolver
  {

public:
  /**
     @brief Constructor for Thick Restarted Eigensolver class
     @param eig_param The eigensolver parameters
     @param mat The operator to solve
     @param matSloppy The operator of sloppy precision
     @param matPrecon The operator to precondition precision
     @param profile Time Profile
  */
  TRLM(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, QudaEigParam *eig_param,
       TimeProfile &profile);

  /**
     @brief Destructor for Thick Restarted Eigensolver class
  */
  virtual ~TRLM();

  virtual bool hermitian() { return true; } /** TRLM is only for Hermitian systems */

  // Variable size matrix
  std::vector<double> ritz_mat;

  // Tridiagonal/Arrow matrix, fixed size.
  double *alpha;
  double *beta;

  /**
     @brief Compute eigenpairs
     @param[in] kSpace Krylov vector space
     @param[in] evals Computed eigenvalues
  */
  void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

  /**
     @brief Applies the TRLM algorithm using the mat operator.
     @param[in] mat Matrix operator
     @param[in] kSpace The Krylov space
  */
  void computeLanczosSolution(const DiracMatrix &mat, std::vector<ColorSpinorField *> &kSpace);

  /**
     @brief Lanczos step: extends the Kylov space.
     @param[in] m The operator to use
     @param[in] v Vector space
     @param[in] j Index of vector being computed
  */
  void lanczosStep(const DiracMatrix &mat, std::vector<ColorSpinorField *> &v, int j);

  /**
     @brief Reorder the Krylov space by eigenvalue
     @param[in] kSpace the Krylov space
  */
  void reorder(std::vector<ColorSpinorField *> &kSpace);

  /**
     @brief Get the eigendecomposition from the arrow matrix
     @param[in] nLocked Number of locked eigenvectors
     @param[in] arrow_pos position of arrowhead
  */
  void eigensolveFromArrowMat(int nLocked, int arror_pos);

  /**
     @brief Rotate the Ritz vectors usinng the arrow matrix eigendecomposition
     @param[in] nKspace current Krylov space
  */
  void computeKeptRitz(std::vector<ColorSpinorField *> &kSpace);
  };

  /**
     arpack_solve()

     @brief The QUDA interface function. One passes two allocated arrays to
     hold the the eigenmode data, the problem matrix, the arpack
     parameters defining what problem is to be solves, and a container
     for QUDA data structure types.
     @param[out] h_evecs Host fields where the e-vectors will be copied to
     @param[out] h_evals Where the e-values will be copied to
     @param[in] mat An explicit construction of the problem matrix.
     @param[in] param Parameter container defining the how the matrix
     is to be solved.
     @param[in] eig_param Parameter structure for all QUDA eigensolvers
     @param[in,out] profile TimeProfile instance used for profiling
  */
  void arpack_solve(std::vector<ColorSpinorField *> &h_evecs, std::vector<Complex> &h_evals, const DiracMatrix &mat,
                    QudaEigParam *eig_param, TimeProfile &profile);

  /**
     @brief Jacobi-Davidson Method.
  */
  class JD : public EigenSolver
  {

  protected:
    TimeProfile *profile_corr_eq_invs;
    TimeProfile *profile_mat_corr_eq_invs;

    SolverParam *solverParam;
    SolverParam *solverParamPrec;
    SolverParam *solverParamMG;
    Solver *mg_solve;
    CG *cg;
    GCR *gcrPrec;

    // JD-specific workspace
    std::vector<ColorSpinorField *> t;
    std::vector<ColorSpinorField *> r_tilde;
    std::vector<ColorSpinorField *> Qhat;
    std::vector<ColorSpinorField *> u;
    std::vector<ColorSpinorField *> u_A;
    std::vector<ColorSpinorField *> V;
    std::vector<ColorSpinorField *> V_A;
    std::vector<ColorSpinorField *> tmpV;
    std::vector<ColorSpinorField *> tmpAV;
    std::vector<ColorSpinorField *> r_lowprec;
    std::vector<ColorSpinorField *> t_lowprec;
    std::vector<ColorSpinorField *> u_lowprec;

    DiracPrecProjCorr *mmPP;
    DiracPrecProjCorr *mmPPSloppy;
    DiracPrecProjCorr *mmPPPrecon;

    int k;
    int m;
    int loopr;
    double theta;
    double norm;

    double outer_prec;
    double inner_prec;
    QudaPrecision outer_prec_lab;
    QudaPrecision inner_prec_lab;

  public:
    /**
       @brief Constructor for JD Eigensolver class
       @param mat Operator of desired precision
       @param matSloppy Operator of sloppy precision
       @param matPrecon Operator of preconditioner precision
       @param eig_param The eigensolver parameters
       @param profile Time Profile
    */
    JD(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, QudaEigParam *eig_param,
       TimeProfile &profile);

    /**
       @brief Compute eigenpairs
       @param[in] kSpace the "acceleration" vector space
       @param[in] evals Computed eigenvalues
    */
    void operator()(std::vector<ColorSpinorField *> &kSpace, std::vector<Complex> &evals);

    /**
       @brief Invert a matrix of the form (I - QQdag)(M-theta*I)(I - QQdag)
       @param[in] mat The original matrix to be inverted after shift-and-project
       @param[in] x Ouput spinor
       @param[in] b Input spinor
       @param[in] verb Local verbosity of the JD proj-eq solve
       @param[in] kp Number of vectors to project against
       @param[in] projSpace Space to project against
    */
    //void invertProjMat(const DiracMatrix &mat, ColorSpinorField &x, ColorSpinorField &b, QudaVerbosity verb,
    //                   int kp, std::vector<ColorSpinorField *> &projSpace);
    void invertProjMat(ColorSpinorField &x, ColorSpinorField &b, QudaVerbosity verb,
                       int kp, std::vector<ColorSpinorField *> &projSpace);

    /**
       @brief Wrapper for CG to allow flexible solver params throughout the correction equation in JD
       @param[in] cg Instance of the CG solver
       @param[in] tol Tolerance of the solve
       @param[in] maxiter Maximum allowed number of iterations for the solve
       @param[in] verb Verbosity of the solve
       @param[in] x Output spinor
       @param[in] b Input spinor
    */
    void K(CG &cg, double tol, int maxiter, QudaVerbosity verb, SolverParam &slvrPar, ColorSpinorField &x,
           ColorSpinorField &b);

    /**
       @brief Some more initializations in the JD eigensolver
       @param[in] csParam Information about the spinors
       @param[in] initVec Spinor assigned to t before the main loop starts
    */
    void moreInits(ColorSpinorParam &csParam, ColorSpinorField &initVec);

    /**
       @brief Some more initializations in the JD eigensolver
       @param[in] ort_dot_prod Resulting dot products
       @param[in] vectr The spinor to be orthogonalized
       @param[in] ort_space The subspace against which vectr will be orthogonalized
       @param[in] size_os Number of dot products to perform
    */
    void orth(Complex *ort_dot_prod, std::vector<ColorSpinorField *> &vectr, std::vector<ColorSpinorField *> &ort_space,
              const int size_os);

    /**
       @brief Check if one or more eigenpairs have been found
       @param[in] eigenpairs A vector of pairs, containing eigeninfo from the subspace
       @param[in] X_tilde The converged eigenvectors
       @param[in] evals The converged eigenvalues
    */
    void checkIfConverged(std::vector<std::pair<double, Complex*>> &eigenpairs,
                          std::vector<ColorSpinorField *> &X_tilde, std::vector<Complex> &evals);

    /**
       @brief When reached the max allowed size of the subspace, resize
       @param[in] eigenpairs A vector of pairs, containing eigeninfo from the subspace
       @param[in] H_ The matrix encoding information about the subspace
    */
    void shrinkSubspace(std::vector<std::pair<double, Complex*>> &eigenpairs, void *H_);

    /**
       @brief Perform an eigendecomposition through the acceleration subspace
       @param[in] eigenpairs A vector of pairs, containing eigeninfo from the subspace
       @param[in] eigensolver_ Eigen object used for the eigendecomposition
       @param[in] H_ The matrix encoding information about the subspace
       @param[in] ort_dot_prod Buffer to store some dot product results
    */
    void eigsolveInSubspace(std::vector<std::pair<double, Complex*>> &eigenpairs, void *eigensolver_,
                            void *H_, Complex *ort_dot_prod);

    /**
       @brief Destructor for JD Eigensolver class
    */
    virtual ~JD();

    virtual bool hermitian() { return true; } /** The current implementation of JD is only for Hermitian systems */
  };

} // namespace quda
