#ifndef _INVERT_QUDA_H
#define _INVERT_QUDA_H

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>

namespace quda {

  /**
     SolverParam is the meta data used to define linear solvers.
   */
  struct SolverParam {
    /**
       Which linear solver to use 
    */
    QudaInverterType inv_type;

    /**
     * The inner Krylov solver used in the preconditioner.  Set to
     * QUDA_INVALID_INVERTER to disable the preconditioner entirely.
     */
    QudaInverterType inv_type_precondition;
    
    /**
     * Whether to use the L2 relative residual, L2 absolute residual
     * or Fermilab heavy-quark residual, or combinations therein to
     * determine convergence.  To require that multiple stopping
     * conditions are satisfied, use a bitwise OR as follows:
     *
     * p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
     *                                     | QUDA_HEAVY_QUARK_RESIDUAL);
     */
    QudaResidualType residual_type;
    
    /**< Whether to use an initial guess in the solver or not */
    QudaUseInitGuess use_init_guess;       

    /**< Reliable update tolerance */
    double delta;           

    /**< Whether to keep the partial solution accumulator in sloppy precision */
    bool use_sloppy_partial_accumulator;

    /**< Enable pipeline solver */
    int pipeline;

    /**< Solver tolerance in the L2 residual norm */
    double tol;             

    /**< Solver tolerance in the heavy quark residual norm */
    double tol_hq;          

    /**< Actual L2 residual norm achieved in solver */
    double true_res;        

    /**< Actual heavy quark residual norm achieved in solver */
    double true_res_hq;     
    
    /**< Maximum number of iterations in the linear solver */
    int maxiter;            

    /**< The number of iterations performed by the solver */
    int iter;
    
    /**< The precision used by the QUDA solver */
    QudaPrecision precision;

    /**< The precision used by the QUDA sloppy operator */
    QudaPrecision precision_sloppy;

    /**< The precision used by the QUDA preconditioner */
    QudaPrecision precision_precondition;

    /**< Preserve the source or not in the linear solver (deprecated?) */    
    QudaPreserveSource preserve_source;       



    // Multi-shift solver parameters

    /**< Number of offsets in the multi-shift solver */    
    int num_offset; 

    /** Offsets for multi-shift solver */
    double offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each offset */
    double tol_offset[QUDA_MAX_MULTI_SHIFT];     

    /** Solver tolerance for each shift when refinement is applied using the heavy-quark residual */
    double tol_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual L2 residual norm achieved in solver for each offset */
    double true_res_offset[QUDA_MAX_MULTI_SHIFT]; 

    /** Actual heavy quark residual norm achieved in solver for each offset */
    double true_res_hq_offset[QUDA_MAX_MULTI_SHIFT]; 


    

    /** Maximum size of Krylov space used by solver */
    int Nkrylov;
    
    /** Number of preconditioner cycles to perform per iteration */
    int precondition_cycle;

    /** Tolerance in the inner solver */
    double tol_precondition;

    /** Maximum number of iterations allowed in the inner solver */
    int maxiter_precondition;

    /** Relaxation parameter used in GCR-DD (default = 1.0) */
    double omega;           


    
    /** Whether to use additive or multiplicative Schwarz preconditioning */
    QudaSchwarzType schwarz_type;

    /**< The time taken by the solver */
    double secs;

    /**< The Gflops rate of the solver */
    double gflops;

    // Incremental EigCG solver parameters
    /**< The precision of the Ritz vectors */
    QudaPrecision precision_ritz;//also search space precision

    int nev;//number of eigenvectors produced by EigCG
    int m;//Dimension of the search space
    int deflation_grid;
    int rhs_idx;
    
    /**
       Constructor that matches the initial values to that of the
       QudaInvertParam instance
       @param param The QudaInvertParam instance from which the values are copied
     */
    SolverParam(QudaInvertParam &param) : inv_type(param.inv_type), 
      inv_type_precondition(param.inv_type_precondition), 
      residual_type(param.residual_type), use_init_guess(param.use_init_guess),
      delta(param.reliable_delta), use_sloppy_partial_accumulator(param.use_sloppy_partial_accumulator), 
      pipeline(param.pipeline), tol(param.tol), tol_hq(param.tol_hq), 
      true_res(param.true_res), true_res_hq(param.true_res_hq),
      maxiter(param.maxiter), iter(param.iter), 
      precision(param.cuda_prec), precision_sloppy(param.cuda_prec_sloppy), 
      precision_precondition(param.cuda_prec_precondition), 
      preserve_source(param.preserve_source), num_offset(param.num_offset), 
      Nkrylov(param.gcrNkrylov), precondition_cycle(param.precondition_cycle), 
      tol_precondition(param.tol_precondition), maxiter_precondition(param.maxiter_precondition), 
      omega(param.omega), schwarz_type(param.schwarz_type), secs(param.secs), gflops(param.gflops),
      precision_ritz(param.cuda_prec_ritz), nev(param.nev), m(param.max_search_dim), deflation_grid(param.deflation_grid), rhs_idx(0) 
    { 
      for (int i=0; i<num_offset; i++) {
	offset[i] = param.offset[i];
	tol_offset[i] = param.tol_offset[i];
	tol_hq_offset[i] = param.tol_hq_offset[i];
      }

      if((param.inv_type == QUDA_INC_EIGCG_INVERTER || param.inv_type == QUDA_EIGCG_INVERTER) && m % 16){//current hack for the magma library
        m = (m / 16) * 16 + 16;
        warningQuda("\nSwitched eigenvector search dimension to %d\n", m);
      }
      if(param.rhs_idx != 0 && (param.inv_type==QUDA_INC_EIGCG_INVERTER || param.inv_type==QUDA_EIGCG_INVERTER)){
        rhs_idx = param.rhs_idx;
      }
    }
    ~SolverParam() { }

    /**
       Update the QudaInvertParam with the data from this
       @param param the QudaInvertParam to be updated
     */
    void updateInvertParam(QudaInvertParam &param, int offset=-1) {
      param.true_res = true_res;
      param.true_res_hq = true_res_hq;
      param.iter += iter;
      param.gflops = (param.gflops*param.secs + gflops*secs) / (param.secs + secs);
      param.secs += secs;
      if (offset >= 0) {
	param.true_res_offset[offset] = true_res_offset[offset];
	param.true_res_hq_offset[offset] = true_res_hq_offset[offset];
      } else {
	for (int i=0; i<num_offset; i++) {
	  param.true_res_offset[i] = true_res_offset[i];
	  param.true_res_hq_offset[i] = true_res_hq_offset[i];
	}
      }
      //for incremental eigCG:
      param.rhs_idx = rhs_idx;
    }

    void updateRhsIndex(QudaInvertParam &param) {
      //for incremental eigCG:
      rhs_idx = param.rhs_idx;
    }

  };

  class Solver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    Solver(SolverParam &param, TimeProfile &profile) : param(param), profile(profile) { ; }
    virtual ~Solver() { ; }

    virtual void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;

    // solver factory
    static Solver* create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy,
			  DiracMatrix &matPrecon, TimeProfile &profile);

    /**
       Set the solver stopping condition
       @param b2 L2 norm squared of the source vector
     */
    static double stopping(const double &tol, const double &b2, QudaResidualType residual_type);

    /**
       Test for solver convergence
       @param r2 L2 norm squared of the residual 
       @param hq2 Heavy quark residual
       @param r2_tol Solver L2 tolerance
       @param hq_tol Solver heavy-quark tolerance
     */
    bool convergence(const double &r2, const double &hq2, const double &r2_tol, 
		     const double &hq_tol);
 
    /**
       Prints out the running statistics of the solver (requires a verbosity of QUDA_VERBOSE)
     */
    void PrintStats(const char*, int k, const double &r2, const double &b2, const double &hq2);

    /** 
	Prints out the summary of the solver convergence (requires a
	versbosity of QUDA_SUMMARIZE).  Assumes
	SolverParam.true_res and SolverParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

  };

  class CG : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~CG();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };


  class PreconCG : public Solver {
    private: 
      const DiracMatrix &mat;
      const DiracMatrix &matSloppy;
      const DiracMatrix &matPrecon;

      Solver *K;
      SolverParam Kparam; // parameters for preconditioner solve

    public:
      PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
               SolverParam &param, TimeProfile &profile);
      virtual ~PreconCG();

      void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };


  class BiCGstab : public Solver {

  private:
    DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    // pointers to fields to avoid multiple creation overhead
    cudaColorSpinorField *yp, *rp, *pp, *vp, *tmpp, *tp;
    bool init;

  public:
    BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	     SolverParam &param, TimeProfile &profile);
    virtual ~BiCGstab();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class GCR : public Solver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matPrecon;

    Solver *K;
    SolverParam Kparam; // parameters for preconditioner solve

  public:
    GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
	SolverParam &param, TimeProfile &profile);
    virtual ~GCR();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  class MR : public Solver {

  private:
    const DiracMatrix &mat;
    cudaColorSpinorField *rp;
    cudaColorSpinorField *Arp;
    cudaColorSpinorField *tmpp;
    bool init;
    bool allocate_r;

  public:
    MR(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~MR();

    void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  // Steepest descent solver used as a preconditioner 
  class SD : public Solver {
    private:
      const DiracMatrix &mat;
      cudaColorSpinorField *Ar;
      cudaColorSpinorField *r;
      cudaColorSpinorField *y;
      bool init;
    
    public: 
      SD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
      virtual ~SD();


      void operator()(cudaColorSpinorField &out, cudaColorSpinorField &in);
  };

  // multigrid solver
  class alphaSA : public Solver {

  protected:
    const DiracMatrix &mat;

  public:
    alphaSA(DiracMatrix &mat, SolverParam &param, TimeProfile &profile);
    virtual ~alphaSA() { ; }

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
  };

  class MultiShiftSolver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    MultiShiftSolver(SolverParam &param, TimeProfile &profile) : 
    param(param), profile(profile) { ; }
    virtual ~MultiShiftSolver() { ; }

    virtual void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in) = 0;
  };

  class MultiShiftCG : public MultiShiftSolver {

  protected:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;

  public:
    MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile);
    virtual ~MultiShiftCG();

    void operator()(cudaColorSpinorField **out, cudaColorSpinorField &in);
  };

  /**
     This computes the optimum guess for the system Ax=b in the L2
     residual norm.  For use in the HMD force calculations using a
     minimal residual chronological method This computes the guess
     solution as a linear combination of a given number of previous
     solutions.  Following Brower et al, only the orthogonalised vector
     basis is stored to conserve memory.
  */
  class MinResExt {

  protected:
    const DiracMatrix &mat;
    TimeProfile &profile;

  public:
    MinResExt(DiracMatrix &mat, TimeProfile &profile);
    virtual ~MinResExt();

    /**
       param x The optimum for the solution vector.
       param b The source vector in the equation to be solved. This is not preserved.
       param p The basis vectors in which we are building the guess
       param q The basis vectors multipled by A
       param N The number of basis vectors
       return The residue of this guess.
    */  
    void operator()(cudaColorSpinorField &x, cudaColorSpinorField &b, cudaColorSpinorField **p,
		    cudaColorSpinorField **q, int N);
  };

  struct DeflationParam {
    //host   projection matrix:
    Complex *proj_matrix; //VH A V

//currently this is just host buffer, in the future it'd be more consistent/convenient to create cpuColorSpinorField. 
    QudaPrecision           ritz_prec;      //keep it right now.    
    void                    *cpuRitzVectors;       //host buffer for Ritz vectors

    cudaColorSpinorField    *cudaRitzVectors;      //device buffer for Ritz vectors

    int  cpu_ritz_dim; //number of Ritz vectors allocated on the host
    int  cuda_ritz_dim; //number of Ritz vectors allocated on the device

    int ld;                 //projection matrix leading dimension
    int tot_dim;            //projection matrix full (maximum) dimension (nev*deflation_grid)
    int cur_dim;            //current dimension (must match rhs_idx: if(rhs_idx < deflation_grid) curr_nevs <= nev * rhs_idx) 
    int added_nevs;
    
    size_t ritz_bytes;

    bool cuda_ritz_alloc;
    bool cpu_ritz_alloc;

    DeflationParam(SolverParam &param, const int eigv_volume) : proj_matrix(0),  cpuRitzVectors(0),  cudaRitzVectors(0), cpu_ritz_dim(0), cuda_ritz_dim(0), cur_dim(0), added_nevs(0), cuda_ritz_alloc(false), cpu_ritz_alloc(false){

       const int spinorSiteSize = 24;

       if(param.nev == 0 || param.deflation_grid == 0) errorQuda("\nIncorrect deflation space parameters...\n");
       
       tot_dim      = param.deflation_grid*param.nev;

       ld           = ((tot_dim+15) / 16) * tot_dim;

       //allocate deflation resources:
       proj_matrix  = new Complex[ld*tot_dim];
       
       ritz_prec = param.precision_ritz;

       ritz_bytes      = eigv_volume*spinorSiteSize*ritz_prec;

       cpu_ritz_dim    = tot_dim; 

       cpuRitzVectors  = malloc(cpu_ritz_dim * ritz_bytes);//mm_malloc(cpu_ritz_dim * ritz_bytes, 32);

printfQuda("\nAllocating %u bytes\n", cpu_ritz_dim * ritz_bytes);

       cpu_ritz_alloc  = true;
    }

    ~DeflationParam(){

       if(proj_matrix)        delete[] proj_matrix;

       if(cuda_ritz_alloc)    delete cudaRitzVectors;

       if(cpu_ritz_alloc)     free(cpuRitzVectors);

    }

    //reset current dimension:
    void ResetDeflationCurrentDim(const int addedvecs){

      if(addedvecs == 0) return; //nothing to do

      if((cur_dim+addedvecs) > tot_dim) errorQuda("\nCannot reset projection matrix dimension.\n");

      added_nevs = addedvecs;
      cur_dim   += added_nevs;

      return;
    }   

    void AllocateRitzCuda(ColorSpinorParam &eigv_param)
    {
      if(eigv_param.siteSubset == QUDA_FULL_SITE_SUBSET) errorQuda("\nError: Ritz vectors must be parity spinors.\n");
      
      if(cuda_ritz_alloc == false)
      {
        eigv_param.setPrecision(ritz_prec);

        eigv_param.eigv_dim  = cuda_ritz_dim;
        eigv_param.eigv_id   = -1;

        eigv_param.create  = QUDA_ZERO_FIELD_CREATE; 
      
        cudaRitzVectors = new cudaColorSpinorField(eigv_param);

        cuda_ritz_alloc = true;
      }
      else
      {
        errorQuda("\nError: CUDA Ritz vectors were already allocated.\n");//or just warning?
      }

      return;
    }

    //print information about the deflation space:
    void PrintInfo(){

       printfQuda("\nProjection matrix information:\n");
       printfQuda("Leading dimension %d\n", ld);
       printfQuda("Total dimension %d\n", tot_dim);
       printfQuda("Current dimension %d\n", cur_dim);
       printfQuda("Host pointer: %p\n", proj_matrix);

       printfQuda("\nRitz vector set information:\n");
       printfQuda("Number of CUDA-allocated Ritz vectors %d\n", cuda_ritz_dim);
       printfQuda("Number of CPU-allocated Ritz vectors %d\n", cpu_ritz_dim);
    }

    void CleanHostRitzVectors()  
    {
       if( cpu_ritz_alloc ){
         free(cpuRitzVectors);
         cpu_ritz_alloc = false;
       }
    }
    void CleanDeviceRitzVectors()
    {
       if( cuda_ritz_alloc ){
         delete cudaRitzVectors;
         cuda_ritz_alloc = false;
       }
    }

  };

  class DeflatedSolver {

  protected:
    SolverParam &param;
    TimeProfile &profile;

  public:
    DeflatedSolver(SolverParam &param, TimeProfile &profile) : 
    param(param), profile(profile) { ; }
    virtual ~DeflatedSolver() { ; }

    virtual void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in) = 0;

    virtual void SaveRitzVecs(cudaColorSpinorField *out, const int nevs, bool cleanResources = false) = 0;

    virtual void Deflate(cudaColorSpinorField &out, cudaColorSpinorField &in) = 0;

    virtual void CleanResources() = 0;

    // solver factory
    static DeflatedSolver* create(SolverParam &param, DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDeflate, TimeProfile &profile);

    bool convergence(const double &r2, const double &hq2, const double &r2_tol, 
		     const double &hq_tol);
 
    /**
       Prints out the running statistics of the solver (requires a verbosity of QUDA_VERBOSE)
     */
    void PrintStats(const char*, int k, const double &r2, const double &b2, const double &hq2);

    /** 
	Prints out the summary of the solver convergence (requires a
	versbosity of QUDA_SUMMARIZE).  Assumes
	SolverParam.true_res and SolverParam.true_res_hq has
	been set
    */
    void PrintSummary(const char *name, int k, const double &r2, const double &b2);

  };

  class IncEigCG : public DeflatedSolver {

  private:
    const DiracMatrix &mat;
    const DiracMatrix &matSloppy;
    const DiracMatrix &matDefl;

    QudaPrecision search_space_prec;
    cudaColorSpinorField *Vm;  //search vectors  (spinor matrix of size eigen_vector_length x m)

    Solver      *initCG;//initCG solver for deflated inversions
    SolverParam initCGparam; // parameters for initCG solve

    bool eigcg_alloc;
    bool use_eigcg;

  public:

    IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matDefl, SolverParam &param, TimeProfile &profile);

    virtual ~IncEigCG();

    //EigCG solver
    void EigCG(cudaColorSpinorField &out, cudaColorSpinorField &in);

    //Incremental eigCG solver (for eigcg and initcg calls)
    void operator()(cudaColorSpinorField *out, cudaColorSpinorField *in);

    //Compute  u dH^{-1} u^{dagger}b: 
    void DeflateSpinor(cudaColorSpinorField &out, cudaColorSpinorField &in, DeflationParam *param);
    //
    void RelocateRitzVectors(cudaColorSpinorField &eigcgSpinor, DeflationParam *param);//move host Ritz vectors to the device...

    //Deflation space management
    void CreateDeflationSpace(cudaColorSpinorField &eigcgSpinor, DeflationParam *&param);
    //extend projection matrix:
    //compute Q' = DiracM Q, (here U = [V, Q] - total Ritz set)
    //construct H-matrix components with Q'^{dag} Q', V^{dag} Q' and Q'^{dag} V
    //extend H-matrix with the components
    void ExpandDeflationSpace(DeflationParam *param, const int new_nevs);
    //
    void DeleteDeflationSpace(DeflationParam *&param);

    //External methods to manage/use deflation space resources:
    void Deflate(cudaColorSpinorField &out, cudaColorSpinorField &in) {  };//not implemented.
    //
    void SaveRitzVecs(cudaColorSpinorField *out, const int nevs, bool cleanResources = false);
    //
    void CleanResources(); 

  };

} // namespace quda

#endif // _INVERT_QUDA_H
