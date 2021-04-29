#include <reactor.h>

#define SUN_CUSP_CONTENT(S)        ( (SUNLinearSolverContent_Dense_custom)(S->content) )
#define SUN_CUSP_SUBSYS_SIZE(S)    ( SUN_CUSP_CONTENT(S)->subsys_size )
#define SUN_CUSP_NUM_SUBSYS(S)     ( SUN_CUSP_CONTENT(S)->nsubsys )
#define SUN_CUSP_SUBSYS_NNZ(S)     ( SUN_CUSP_CONTENT(S)->subsys_nnz )

#define SUN_CUSP_LASTFLAG(S)       ( SUN_CUSP_CONTENT(S)->last_flag )
#define SUN_CUSP_STREAM(S)         ( SUN_CUSP_CONTENT(S)->stream )
#define SUN_CUSP_NBLOCK(S)         ( SUN_CUSP_CONTENT(S)->nbBlocks )
#define SUN_CUSP_NTHREAD(S)        ( SUN_CUSP_CONTENT(S)->nbThreads )

#ifdef AMREX_USE_CUDA
/**********************************/
/*
 * CUSTOM SOLVER STUFF
 */
SUNLinearSolver SUNLinSol_dense_custom(N_Vector y, SUNMatrix A, cudaStream_t stream)
{
  /* Check that required arguments are not NULL */
  if (y == NULL || A == NULL) return(NULL);

  /* Check compatibility with supplied SUNMatrix and N_Vector */
  if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA) return(NULL);
  if (SUNMatGetID(A) != SUNMATRIX_CUSPARSE) return(NULL);

  /* Matrix and vector dimensions must agree */
  if (N_VGetLength(y) != SUNMatrix_cuSparse_Columns(A)) return(NULL);

  /* Check that the vector is using managed memory */
  if (!N_VIsManagedMemory_Cuda(y)) return(NULL);

  /* Create an empty linear solver */
  SUNLinearSolver S;
  S = NULL;
  S = SUNLinSolNewEmpty();
  if (S == NULL) {
     return(NULL);
  }

  /* Attach operations */
  S->ops->gettype    = SUNLinSolGetType_Dense_custom;
  S->ops->setup      = SUNLinSolSetup_Dense_custom;
  S->ops->solve      = SUNLinSolSolve_Dense_custom;
  S->ops->free       = SUNLinSolFree_Dense_custom;

  /* Create content */
  SUNLinearSolverContent_Dense_custom content;
  content = NULL;
  content = (SUNLinearSolverContent_Dense_custom) malloc(sizeof *content);
  if (content == NULL) {
      SUNLinSolFree(S);
      return(NULL);
  }

  /* Attach content */
  S->content = content;

  /* Fill content */
  content->last_flag   = 0;
  content->nsubsys     = SUNMatrix_cuSparse_NumBlocks(A);
  content->subsys_size = SUNMatrix_cuSparse_BlockRows(A);
  content->subsys_nnz  = SUNMatrix_cuSparse_BlockNNZ(A);
  content->nbBlocks    = std::max(1,content->nsubsys/32);
  content->nbThreads   = 32;
  content->stream      = stream;

  return(S);
}
#endif


#ifdef AMREX_USE_CUDA
SUNLinearSolver_Type SUNLinSolGetType_Dense_custom(SUNLinearSolver S)
{
  return(SUNLINEARSOLVER_DIRECT);
}
#endif

#ifdef AMREX_USE_CUDA
int SUNLinSolSetup_Dense_custom(SUNLinearSolver S, SUNMatrix A)
{
  return(SUNLS_SUCCESS);
}
#endif

#ifdef AMREX_USE_CUDA
int SUNLinSolSolve_Dense_custom(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                                N_Vector b, realtype tol)
{
  cudaError_t cuda_status = cudaSuccess;

  /* Get Device pointers for Kernel call */
  realtype *x_d      = N_VGetDeviceArrayPointer_Cuda(x);
  realtype *b_d      = N_VGetDeviceArrayPointer_Cuda(b);

  realtype *d_data    = SUNMatrix_cuSparse_Data(A);

  BL_PROFILE_VAR("fKernelDenseSolve()", fKernelDenseSolve);
  const auto ec = Gpu::ExecutionConfig(SUN_CUSP_NUM_SUBSYS(S));
  // TODO: why is this AMREX version NOT working ?
  //launch_global<<<ec.numBlocks, ec.numThreads, ec.sharedMem, SUN_CUSP_STREAM(S)>>>(
  //    [=] AMREX_GPU_DEVICE () noexcept {
  //        for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
  //            icell < SUN_CUSP_NUM_SUBSYS(S); icell += stride) {
  //            fKernelDenseSolve(icell, x_d, b_d,
  //                  SUN_CUSP_SUBSYS_SIZE(S), SUN_CUSP_SUBSYS_NNZ(S), data_d);
  //        }
  //    });
  fKernelDenseSolve<<<SUN_CUSP_NBLOCK(S), SUN_CUSP_NTHREAD(S), ec.sharedMem, SUN_CUSP_STREAM(S)>>>
                   (SUN_CUSP_NUM_SUBSYS(S), x_d, b_d, SUN_CUSP_SUBSYS_SIZE(S), SUN_CUSP_SUBSYS_NNZ(S), d_data);

  cuda_status = cudaStreamSynchronize(SUN_CUSP_STREAM(S));
  assert(cuda_status == cudaSuccess);

  BL_PROFILE_VAR_STOP(fKernelDenseSolve);

  return(SUNLS_SUCCESS);
}
#endif

#ifdef AMREX_USE_CUDA
int SUNLinSolFree_Dense_custom(SUNLinearSolver S)
{
  /* return with success if already freed */
  if (S == NULL) return(SUNLS_SUCCESS);

  /* free content structure */
  if (S->content) {
      free(S->content);
      S->content = NULL;
  }

  /* free ops structure */
  if (S->ops) {
      free(S->ops);
      S->ops = NULL;
  }

  /* free the actual SUNLinSol */
  free(S);
  S = NULL;

  return(SUNLS_SUCCESS);
}
#endif

/**********************************/


enum class LinearSolverMethod {

  SPARSE_DIRECT_START = 0,            // sparse direct methods -- start
  sparse_custom = 1,
  sparse_batched_cusolver = 2,

  DENSE_DIRECT_START = 20,            // dense direct methods -- start
  dense_batched_magma = 21,

  ITERATIVE_MATRIX_FREE_START = 40,         // iterative matrix free methods -- start
  iterative_mf_gmres = 41

};

class SundialsLinearSystem {

public:
    SundialsLinearSystem(CVodeUserData* user_data, LinearSolverMethod method = LinearSolverMethod::iterative_gmres, int precondition = 0);

    ~SundialsLinearSystem();

    void InitializeSparsityPattern();
    void CreateSolverObject();

    bool IsUserSuppliedJacobian();
    bool IsPreconditioned();

    SUNLinearSolver Solver();
    SUNMatrix Matrix();

private:
    SUNLinearSolver LS;
    SUNMatrix A;

    LinearSolverMethod method;
    CVodeUserData* user_data;
};


SundialsLinearSystem::SundialsLinearSystem(CVodeUserData* user_data, LinearSolverMethod method, int precondition)
  : user_data(user_data), method(method), precondition(precondition), LS(nullptr), A(nullptr)
{

}

SundialsLinearSystem::~SundialsLinearSystem()
{
    if (LS) SUNLinSolFree(LS);
    if (A) SUNMatDestroy(A);

    if (user_data->ianalytical_jacobian == 1) {
#ifdef AMREX_USE_CUDA
        The_Arena()->free(user_data->csr_row_count_h);
        The_Arena()->free(user_data->csr_col_index_h);
        if (user_data->isolve_type == iterative_gmres_solve) {
            The_Arena()->free(user_data->csr_val_h);
            The_Arena()->free(user_data->csr_jac_h);
            The_Device_Arena()->free(user_data->csr_val_d);
            The_Device_Arena()->free(user_data->csr_jac_d);

            cusolverStatus_t cusolver_status = cusolverSpDestroy(user_data->cusolverHandle);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusolver_status = cusolverSpDestroyCsrqrInfo(user_data->info);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cudaFree(user_data->buffer_qr);
        } else if (user_data->isolve_type == sparse_cusolver_solve) {
            cusolverStatus_t cusolver_status = cusolverSpDestroy(user_data->cusolverHandle);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusparseStatus_t cusparse_status = cusparseDestroy(user_data->cuSPHandle);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
        } else {
            cusparseStatus_t cusparse_status = cusparseDestroy(user_data->cuSPHandle);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);
        }
#else
	     Abort("Shoudn't be there. Analytical Jacobian only available with CUDA !");
#endif
    }
}

void SundialsLinearSystem::InitializeSparsityPattern()
{
#ifdef AMREX_USE_CUDA
    if (user_data->ianalytical_jacobian == 1) {
        int HP;
        if (user_data->ireactor_type == 1) {
            HP = 0;
        } else {
            HP = 1;
        }

        /* Find sparsity pattern to fill structure of sparse matrix */
        BL_PROFILE_VAR("SparsityFuegoStuff", SparsityStuff);
        BL_PROFILE_VAR_STOP(SparsityStuff);
        if (user_data->isolve_type == iterative_gmres_solve) {
	          // Create batch QR solver for preconditioning GMRES
            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_INFO_SYST_SIMPLIFIED(&(user_data->NNZ),&HP);
            BL_PROFILE_VAR_STOP(SparsityStuff);

            BL_PROFILE_VAR_START(AllocsCVODE);
            user_data->csr_row_count_h = (int*) The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
            user_data->csr_col_index_h = (int*) The_Arena()->alloc(user_data->NNZ * sizeof(int));
            user_data->csr_jac_h       = (double*) The_Arena()->alloc(user_data->NNZ * NCELLS * sizeof(double));
            user_data->csr_val_h       = (double*) The_Arena()->alloc(user_data->NNZ * NCELLS * sizeof(double));

            user_data->csr_row_count_d = (int*) The_Device_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
            user_data->csr_col_index_d = (int*) The_Device_Arena()->alloc(user_data->NNZ * sizeof(int));
            user_data->csr_jac_d       = (double*) The_Device_Arena()->alloc(user_data->NNZ * NCELLS * sizeof(double));
            user_data->csr_val_d       = (double*) The_Device_Arena()->alloc(user_data->NNZ * NCELLS * sizeof(double));
            BL_PROFILE_VAR_STOP(AllocsCVODE);

            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(user_data->csr_col_index_h, user_data->csr_row_count_h, &HP,1);
            BL_PROFILE_VAR_STOP(SparsityStuff);

            amrex::Gpu::htod_memcpy(&user_data->csr_col_index_d,&user_data->csr_col_index_h,
                                    sizeof(user_data->NNZ * sizeof(int)));
            amrex::Gpu::htod_memcpy(&user_data->csr_row_count_d,&user_data->csr_row_count_h,
                                    sizeof((NUM_SPECIES+2) * sizeof(int)));

            // Create Sparse batch QR solver
            // qr info and matrix descriptor
            BL_PROFILE_VAR("CuSolverInit", CuSolverInit);
            size_t workspaceInBytes, internalDataInBytes;
            cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
            cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

            workspaceInBytes = 0;
            internalDataInBytes = 0;

            cusolver_status = cusolverSpCreate(&(user_data->cusolverHandle));
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusolver_status = cusolverSpSetStream(user_data->cusolverHandle, stream);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusparse_status = cusparseCreateMatDescr(&(user_data->descrA));
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            cusparse_status = cusparseSetMatType(user_data->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            cusparse_status = cusparseSetMatIndexBase(user_data->descrA, CUSPARSE_INDEX_BASE_ONE);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            cusolver_status = cusolverSpCreateCsrqrInfo(&(user_data->info));
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            // symbolic analysis
            cusolver_status = cusolverSpXcsrqrAnalysisBatched(user_data->cusolverHandle,
                                       NUM_SPECIES+1, // size per subsystem
                                       NUM_SPECIES+1, // size per subsystem
                                       user_data->NNZ,
                                       user_data->descrA,
                                       user_data->csr_row_count_h,
                                       user_data->csr_col_index_h,
                                       user_data->info);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            /*
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaStat1 = cudaMemGetInfo( &free_mem, &total_mem );
            assert( cudaSuccess == cudaStat1 );
            std::cout<<"(AFTER SA) Free: "<< free_mem<< " Tot: "<<total_mem<<std::endl;
            */

            // allocate working space
            cusolver_status = cusolverSpDcsrqrBufferInfoBatched(user_data->cusolverHandle,
                                                      NUM_SPECIES+1, // size per subsystem
                                                      NUM_SPECIES+1, // size per subsystem
                                                      user_data->NNZ,
                                                      user_data->descrA,
                                                      user_data->csr_val_h,
                                                      user_data->csr_row_count_h,
                                                      user_data->csr_col_index_h,
                                                      NCELLS,
                                                      user_data->info,
                                                      &internalDataInBytes,
                                                      &workspaceInBytes);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cudaError_t cudaStat1            = cudaSuccess;
            cudaStat1 = cudaMalloc((void**)&(user_data->buffer_qr), workspaceInBytes);
            assert(cudaStat1 == cudaSuccess);
            BL_PROFILE_VAR_STOP(CuSolverInit);

        } else if (isolve_type == sparse_cusolver_solve) {
	          // Create batch QR solver to invert lienar systems directly
            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_INFO_SYST(&(user_data->NNZ),&HP,1);
            BL_PROFILE_VAR_STOP(SparsityStuff);

            BL_PROFILE_VAR_START(AllocsCVODE);
            user_data->csr_row_count_h = (int*) The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
            user_data->csr_col_index_h = (int*) The_Arena()->alloc(user_data->NNZ * sizeof(int));

            BL_PROFILE_VAR_STOP(AllocsCVODE);

            cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
            cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
            cudaError_t      cuda_status     = cudaSuccess;

            cusolver_status = cusolverSpCreate(&(user_data->cusolverHandle));
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusolver_status = cusolverSpSetStream(user_data->cusolverHandle, stream);
            assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

            cusparse_status = cusparseCreate(&(user_data->cuSPHandle));
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            cusparse_status = cusparseSetStream(user_data->cuSPHandle, stream);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            A = SUNMatrix_cuSparse_NewBlockCSR(NCELLS, (NUM_SPECIES + 1), (NUM_SPECIES + 1),
                                               user_data->NNZ, user_data->cuSPHandle);
            if (check_flag((void *)A, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);

            int retval = SUNMatrix_cuSparse_SetFixedPattern(A, 1);
            if(check_flag(&retval, "SUNMatrix_cuSparse_SetFixedPattern", 1)) return(1);

            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_PREPROC_SYST_CSR(user_data->csr_col_index_h, user_data->csr_row_count_h, &HP, 1, 0);
            amrex::Gpu::htod_memcpy(&user_data->csr_col_index_d,&user_data->csr_col_index_h,
                                    sizeof(user_data->csr_col_index_h));
            amrex::Gpu::htod_memcpy(&user_data->csr_row_count_d,&user_data->csr_row_count_h,
                                    sizeof(user_data->csr_row_count_h));
            SUNMatrix_cuSparse_CopyToDevice(A, NULL, user_data->csr_row_count_h, user_data->csr_col_index_h);
            BL_PROFILE_VAR_STOP(SparsityStuff);
        } else {
            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_INFO_SYST(&(user_data->NNZ),&HP,1);
            BL_PROFILE_VAR_STOP(SparsityStuff);

            BL_PROFILE_VAR_START(AllocsCVODE);
            user_data->csr_row_count_h = (int*) The_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
            user_data->csr_col_index_h = (int*) The_Arena()->alloc(user_data->NNZ * sizeof(int));

            user_data->csr_row_count_d = (int*) The_Device_Arena()->alloc((NUM_SPECIES+2) * sizeof(int));
            user_data->csr_col_index_d = (int*) The_Device_Arena()->alloc(user_data->NNZ * sizeof(int));
            BL_PROFILE_VAR_STOP(AllocsCVODE);

            cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

            cusparse_status = cusparseCreate(&(user_data->cuSPHandle));
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            cusparse_status = cusparseSetStream(user_data->cuSPHandle, stream);
            assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

            A = SUNMatrix_cuSparse_NewBlockCSR(NCELLS, (NUM_SPECIES + 1), (NUM_SPECIES + 1),
                                               user_data->NNZ, user_data->cuSPHandle);
            if (check_flag((void *)A, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);

            int retval = SUNMatrix_cuSparse_SetFixedPattern(A, 1);
            if(check_flag(&retval, "SUNMatrix_cuSparse_SetFixedPattern", 1)) return(1);

            BL_PROFILE_VAR_START(SparsityStuff);
            SPARSITY_PREPROC_SYST_CSR(user_data->csr_col_index_h, user_data->csr_row_count_h, &HP, 1, 0);
            amrex::Gpu::htod_memcpy(&user_data->csr_col_index_d,&user_data->csr_col_index_h,
                                    sizeof(user_data->csr_col_index_h));
            amrex::Gpu::htod_memcpy(&user_data->csr_row_count_d,&user_data->csr_row_count_h,
                                    sizeof(user_data->csr_row_count_h));
            SUNMatrix_cuSparse_CopyToDevice(A, NULL, user_data->csr_row_count_h, user_data->csr_col_index_h);
            BL_PROFILE_VAR_STOP(SparsityStuff);
        }
    }
#endif
}

void SundialsLinearSystem::CreateSolverObject()
{
    if (user_data->isolve_type == iterative_gmres_solve) {
        if (user_data->ianalytical_jacobian == 0) {
            LS = SUNSPGMR(y, PREC_NONE, 0);
            if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
        } else {
            LS = SUNSPGMR(y, PREC_LEFT, 0);
            if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
        }
#if defined(AMREX_USE_CUDA)
    } else if (user_data->isolve_type == sparse_cusolver_solve) {
        LS = SUNLinSol_cuSolverSp_batchQR(y, A, user_data->cusolverHandle);
        if(check_flag((void *)LS, "SUNLinSol_cuSolverSp_batchQR", 0)) return(1);

    } else if (user_data->isolve_type == dense_magma_solve) {


    } else {
        /* Create dense SUNLinearSolver object for use by CVode */
        LS = SUNLinSol_dense_custom(y, A, stream); //NCELLS, (NUM_SPECIES+1), user_data->NNZ, stream);
        if(check_flag((void *)LS, "SUNDenseLinearSolver", 0)) return(1);
#endif
    }
}

bool IsUserSuppliedJacobian()
{
    return method < ITERATIVE_MATRIX_FREE_START;
}

bool IsPreconditioned()
{
    return precondition;
}

SUNMatrix SundialsLinearSystem::Solver()
{
    return LS;
}

SUNMatrix SundialsLinearSystem::Matrix()
{
    return A;
}
