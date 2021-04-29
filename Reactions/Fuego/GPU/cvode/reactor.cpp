#include <reactor.h>
#include <reactor_linsys.h>
#include <AMReX_ParmParse.H>
#include <chemistry_file.H>
#include <PelePhysics.H>
#include "mechanism.h"
#include <AMReX_Gpu.H>
#include <AMReX_SUNMemory.H>
#include <AMREX_misc.H>

using namespace amrex;
using namespace amrex::sundials;

/**********************************/
/* Global Variables */

// int sparse_solve          = 1;
// int sparse_cusolver_solve = 5;
// int iterative_gmres_solve = 99;
int eint_rho = 1; // in/out = rhoE/rhoY
int enth_rho = 2; // in/out = rhoH/rhoY

Array<Real,NUM_SPECIES+1> typVals = {-1};
Real relTol    = 1.0e-10;
Real absTol    = 1.0e-10;
/**********************************/

/**********************************/
/* Set or update typVals */
void SetTypValsODE(const std::vector<double>& ExtTypVals) {
    Vector<std::string> kname;
    pele::physics::eos::speciesNames(kname);

    Print() << "Set the typVals in PelePhysics: \n  ";
    int size_ETV = ExtTypVals.size();
    AMREX_ASSERT(size_ETV == typVals.size());
    for (int i=0; i<size_ETV-1; i++) {
        typVals[i] = ExtTypVals[i];
        Print() << kname[i] << ":" << typVals[i] << "  ";
    }
    typVals[size_ETV-1] = ExtTypVals[size_ETV-1];
    Print() << "Temp:"<< typVals[size_ETV-1] <<  " \n";
}

/* Set or update the rel/abs tolerances  */
void SetTolFactODE(double relative_tol,double absolute_tol) {
    relTol = relative_tol;
    absTol = absolute_tol;
    Print() << "Set RTOL, ATOL = "<<relTol<< " "<<absTol<<  " in PelePhysics\n";
}

/* Infos to print once */
int reactor_info(int reactor_type, int Ncells){

    /* ParmParse from the inputs file */
    ParmParse pp("ode");
    int ianalytical_jacobian = 0;
    pp.query("analytical_jacobian",ianalytical_jacobian);
    int iverbose = 1;
    pp.query("verbose",iverbose);

    if (iverbose > 0) {
        Print() << "Number of species in mech is " << NUM_SPECIES << "\n";
        Print() << "Number of cells in one solve is " << Ncells << "\n";
    }

    std::string  solve_type_str = "none";
    ParmParse ppcv("cvode");
    ppcv.query("solve_type", solve_type_str);
    int isolve_type = -1;
    if (solve_type_str == "sparse_custom") {
        isolve_type = LinearSolverMethod::sparse_custom;
    } else if (solve_type_str == "sparse") {
        isolve_type = LienarSolverMethod::sparse_batched_cusolver;
    } else if (solve_type_str == "GMRES") {
        isolve_type = LienarSolverMethod::iterative_mf_gmres;
    } else {
        Abort("Wrong solve_type. Options are: sparse, sparse_custom, GMRES");
    }

    /* Checks */
    if (isolve_type == LienarSolverMethod::iterative_mf_gmres) {
        if (ianalytical_jacobian == 1) {
#if defined(AMREX_USE_CUDA)
            if (iverbose > 0) {
                Print() <<"Using an Iterative GMRES Solver with sparse simplified preconditioning \n";
            }
            int nJdata;
            int HP;
            if (reactor_type == eint_rho) {
                HP = 0;
            } else {
                HP = 1;
            }
            /* Precond data */
            SPARSITY_INFO_SYST_SIMPLIFIED(&nJdata,&HP);
            if (iverbose > 0) {
                Print() << "--> SPARSE Preconditioner -- non zero entries: " << nJdata << ", which represents "<< nJdata/float((NUM_SPECIES+1) * (NUM_SPECIES+1)) *100.0 <<" % fill-in pattern\n";
            }
#elif defined(AMREX_USE_HIP)
            Abort("\n--> With HIP the only option is NP GMRES \n");
#else
            Abort("\n--> Not sure what do with analytic Jacobian in this case \n");
#endif
        } else {
            if (iverbose > 0) {
                Print() <<"Using an Iterative GMRES Solver without preconditionning \n";
            }
        }

    } else if (isolve_type == LinearSolverMethod::sparse_custom) {
#if defined(AMREX_USE_CUDA)
        if (ianalytical_jacobian == 1) {
            if (iverbose > 0) {
                Print() <<"Using a custom sparse direct solver (with an analytical Jacobian) \n";
            }
            int nJdata;
            int HP;
            if (reactor_type == eint_rho) {
                HP = 0;
            } else {
                HP = 1;
            }
            /* Jac data */
            SPARSITY_INFO_SYST(&nJdata,&HP,Ncells);
            if (iverbose > 0) {
                Print() << "--> SPARSE Solver -- non zero entries: " << nJdata << ", which represents "<< nJdata/float(Ncells * (NUM_SPECIES+1) * (NUM_SPECIES+1)) * 100.0 <<" % fill-in pattern\n";
            }
        } else {
            Abort("\n--> Custom direct sparse solver requires an analytic Jacobian \n");
        }
#elif defined(AMREX_USE_HIP)
        Abort("\n--> With HIP the only option is NP GMRES \n");
#else
        Abort("\n--> No sparse solve for this case\n");
#endif

    } else if (isolve_type == LienarSolverMethod::sparse_batched_cusolver) {
#if defined(AMREX_USE_CUDA)
        if (ianalytical_jacobian == 1) {
            Print() <<"Using a Sparse Direct Solver based on cuSolver \n";
            int nJdata;
            int HP;
            if (reactor_type == eint_rho) {
                HP = 0;
            } else {
                HP = 1;
            }
            /* Jac data */
            SPARSITY_INFO_SYST(&nJdata,&HP,Ncells);
            if (iverbose > 0) {
               Print() << "--> SPARSE Solver -- non zero entries: " << nJdata << ", which represents "<< nJdata/float(Ncells * (NUM_SPECIES+1) * (NUM_SPECIES+1)) * 100.0 <<" % fill-in pattern\n";
            }
        } else {
            Abort("\n--> Sparse direct solvers requires an analytic Jacobian \n");
        }
#elif defined(AMREX_USE_HIP)
        Abort("\n--> No HIP equivalent to cuSolver implemented yet \n");
#else
        Abort("\n--> No batch solver for this case\n");
#endif

    } else {
        Abort("\n--> Bad linear solver choice for CVODE \n");
    }

    Print() << "\n--> DONE WITH INITIALIZATION (GPU)" << reactor_type << "\n";

    return(0);
}

/* Main routine for external looping */
int react(const amrex::Box& box,
          amrex::Array4<amrex::Real> const& rY_in,
          amrex::Array4<amrex::Real> const& rY_src_in,
          amrex::Array4<amrex::Real> const& T_in,
          amrex::Array4<amrex::Real> const& rEner_in,
          amrex::Array4<amrex::Real> const& rEner_src_in,
          amrex::Array4<amrex::Real> const& FC_in,
          amrex::Array4<int> const& mask,
          amrex::Real &dt_react,
          amrex::Real &time,
          const int &reactor_type,
          amrex::gpuStream_t stream) {

    /* CVODE */
    N_Vector y         = NULL;
    // SUNLinearSolver LS = NULL;
    // SUNMatrix A        = NULL;
    void *cvode_mem    = NULL;
    /* Misc */
    int flag;
    int NCELLS, neq_tot;
    N_Vector atol;
    realtype *ratol;

    /* ParmParse from the inputs file */
    ParmParse pp("ode");
    int ianalytical_jacobian = 0;
    pp.query("analytical_jacobian",ianalytical_jacobian);
    int iverbose = 1;
    pp.query("verbose",iverbose);

    std::string solve_type_str = "none";
    ParmParse ppcv("cvode");
    ppcv.query("solve_type", solve_type_str);
    int isolve_type = -1;
    int precondition = 0;
    if (solve_type_str == "sparse_custom") {
        isolve_type = LinearSolverMethod::sparse_custom;
    } else if (solve_type_str == "sparse") {
        isolve_type = LinearSolverMethod::sparse_batched_cusolver;
    } else if (solve_type_str == "GMRES") {
        precondition = ianalytical_jacobian;
        isolve_type = LinearSolverMethod::iterative_mf_gmres;
    }

    /* Linear solver */
    SundialsLinearSystem linear_system(user_data, isolve_type, precondition);

    /* Args */
    NCELLS         = box.numPts();
    neq_tot        = (NUM_SPECIES + 1) * NCELLS;

    /* User data -- host and device needed */
    UserData user_data;

    BL_PROFILE_VAR("AllocsInCVODE", AllocsCVODE);
    user_data = (CVodeUserData *) The_Arena()->alloc(sizeof(struct CVodeUserData));
    BL_PROFILE_VAR_STOP(AllocsCVODE);

    user_data->ncells               = NCELLS;
    user_data->neqs_per_cell        = NUM_SPECIES;
    user_data->ireactor_type        = reactor_type;
    user_data->ianalytical_jacobian = ianalytical_jacobian;
    user_data->isolve_type          = isolve_type;
    user_data->iverbose             = iverbose;
    user_data->nbBlocks             = std::max(1,NCELLS/32);
    user_data->stream               = stream;
    user_data->nbThreads            = 32;

    // Will initialize sparsity pattern if required.
    linear_system.InitializeSparsityPattern();

    /* Definition of main vector */
#if defined(AMREX_USE_CUDA)
    y = N_VNewWithMemHelp_Cuda(neq_tot, /*use_managed_mem=*/true, *The_SUNMemory_Helper());
    if(check_flag((void*)y, "N_VNewWithMemHelp_Cuda", 0)) return(1);
#elif defined(AMREX_USE_HIP)
    y = N_VNewWithMemHelp_Hip(neq_tot, /*use_managed_mem=*/true, *The_SUNMemory_Helper());
    if(check_flag((void*)y, "N_VNewWithMemHelp_Hip", 0)) return(1);
#endif

    /* Use a non-default cuda stream for kernel execution */
#if defined(AMREX_USE_CUDA)
    SUNCudaExecPolicy* stream_exec_policy = new SUNCudaThreadDirectExecPolicy(256, stream);
    SUNCudaExecPolicy* reduce_exec_policy = new SUNCudaBlockReduceExecPolicy(256, 0, stream);
    N_VSetKernelExecPolicy_Cuda(y, stream_exec_policy, reduce_exec_policy);
#elif defined(AMREX_USE_HIP)
    SUNHipExecPolicy* stream_exec_policy = new SUNHipThreadDirectExecPolicy(256, stream);
    SUNHipExecPolicy* reduce_exec_policy = new SUNHipBlockReduceExecPolicy(256, 0, stream);
    N_VSetKernelExecPolicy_Hip(y, stream_exec_policy, reduce_exec_policy);
#endif

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula and the use of a Newton iteration */
    cvode_mem = CVodeCreate(CV_BDF);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    flag = CVodeSetUserData(cvode_mem, static_cast<void*>(user_data));

    BL_PROFILE_VAR_START(AllocsCVODE);
    /* Define vectors to be used later in creact */
    user_data->rhoe_init_d   = (double*) The_Device_Arena()->alloc(NCELLS* sizeof(double));
    user_data->rhoesrc_ext_d = (double*) The_Device_Arena()->alloc(NCELLS* sizeof(double));
    user_data->rYsrc_d       = (double*) The_Device_Arena()->alloc(NCELLS*NUM_SPECIES*sizeof(double));
    BL_PROFILE_VAR_STOP(AllocsCVODE);

    /* Get Device pointer of solution vector */
#if defined(AMREX_USE_CUDA)
    realtype *yvec_d      = N_VGetDeviceArrayPointer_Cuda(y);
#elif defined(AMREX_USE_HIP)
    realtype *yvec_d      = N_VGetDeviceArrayPointer_Hip(y);
#else
    Abort("No device arrary pointer");
#endif

    BL_PROFILE_VAR("reactor::FlatStuff", FlatStuff);
    /* Fill the full box_ncells length vectors from input Array4*/
    const auto len        = amrex::length(box);
    const auto lo         = amrex::lbound(box);
    amrex::ParallelFor(box,
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        int icell = (k-lo.z)*len.x*len.y + (j-lo.y)*len.x + (i-lo.x);

        box_flatten(icell, i, j, k, user_data->ireactor_type,
                    rY_in, rY_src_in, T_in,
                    rEner_in, rEner_src_in,
                    yvec_d,
                    user_data->rYsrc_d,
                    user_data->rhoe_init_d,
                    user_data->rhoesrc_ext_d);
    });
    BL_PROFILE_VAR_STOP(FlatStuff);

    amrex::Real time_init = time;
    amrex::Real time_out  = time + dt_react;

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function, the inital time, and
     * initial dependent variable vector y. */
    flag = CVodeInit(cvode_mem, cF_RHS, time_init, y);
    if (check_flag(&flag, "CVodeInit", 1)) return(1);

    /* Definition of tolerances: one for each species */
    atol  = N_VClone(y);
#if defined(AMREX_USE_CUDA)
    ratol = N_VGetHostArrayPointer_Cuda(atol);
#elif defined(AMREX_USE_HIP)
    ratol = N_VGetHostArrayPointer_Hip(atol);
#endif
    if (typVals[0]>0) {
        printf("Setting CVODE tolerances rtol = %14.8e atolfact = %14.8e in PelePhysics \n",relTol, absTol);
        for (int i = 0; i < NCELLS; i++) {
            int offset = i * (NUM_SPECIES + 1);
            for  (int k = 0; k < NUM_SPECIES + 1; k++) {
                ratol[offset + k] = typVals[k]*absTol;
            }
        }
    } else {
        for (int i=0; i<neq_tot; i++) {
            ratol[i] = absTol;
        }
    }
#if defined(AMREX_USE_CUDA)
    N_VCopyToDevice_Cuda(atol);
#elif defined(AMREX_USE_HIP)
    N_VCopyToDevice_Hip(atol);
#endif
    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    flag = CVodeSVtolerances(cvode_mem, relTol, atol);
    if (check_flag(&flag, "CVodeSVtolerances", 1)) return(1);

    /* Create the linear solver object */
    linear_system.CreateSolverObject();

    /* Set matrix and linear solver to Cvode */
    flag = CVodeSetLinearSolver(cvode_mem, linear_system.Solver(), linear_system.Matrix());
    if(check_flag(&flag, "CVodeSetLinearSolver", 1)) return(1);

    /* Set the Jacobian-times-vector function */
    flag = CVodeSetJacTimes(cvode_mem, NULL, NULL);
    if(check_flag(&flag, "CVodeSetJacTimes", 1)) return(1);

    /* Set the user-supplied Jacobian routine Jac */
    if (linear_system.IsUserSuppliedJacobian()) {
        flag = CVodeSetJacFn(cvode_mem, cJac);
        if(check_flag(&flag, "CVodeSetJacFn", 1)) return(1);
    }

    /* Set the preconditioner solve and setup functions */
    if (linear_system.IsPreconditioned()) {
#if defined(AMREX_USE_CUDA)
        flag = CVodeSetPreconditioner(cvode_mem, Precond, PSolve);
        if(check_flag(&flag, "CVodeSetPreconditioner", 1)) return(1);
#else
        Abort("No options for preconditioning on non-CUDA GPUs");
#endif
    }

    /* Set the max number of time steps */
    flag = CVodeSetMaxNumSteps(cvode_mem, 100000);
    if(check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);

    /* Set the max order */
    flag = CVodeSetMaxOrd(cvode_mem, 2);
    if(check_flag(&flag, "CVodeSetMaxOrd", 1)) return(1);

    BL_PROFILE_VAR("AroundCVODE", AroundCVODE);
    flag = CVode(cvode_mem, time_out, y, &time_init, CV_NORMAL);
    if (check_flag(&flag, "CVode", 1)) return(1);
    BL_PROFILE_VAR_STOP(AroundCVODE);

#ifdef MOD_REACTOR
    /* ONLY FOR PP */
    /*If reactor mode is activated, update time */
    dt_react = time_init - time;
    time = time_init;
#endif

    long int nfe;
    flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);

    BL_PROFILE_VAR_START(FlatStuff);
    /* Update the input/output Array4 rY_in and rEner_in*/
    amrex::ParallelFor(box,
    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
    {
        int icell = (k-lo.z)*len.x*len.y + (j-lo.y)*len.x + (i-lo.x);

        box_unflatten(icell, i, j, k, user_data->ireactor_type,
                      rY_in, T_in, rEner_in, rEner_src_in, FC_in,
                      yvec_d, user_data->rhoe_init_d, nfe, dt_react);
    });
    BL_PROFILE_VAR_STOP(FlatStuff);

    if (user_data->iverbose > 1) {
        PrintFinalStats(cvode_mem);
    }

    N_VDestroy(y);          /* Free the y vector */
    CVodeFree(&cvode_mem);

    The_Device_Arena()->free(user_data->rhoe_init_d);
    The_Device_Arena()->free(user_data->rhoesrc_ext_d);
    The_Device_Arena()->free(user_data->rYsrc_d);

    The_Arena()->free(user_data);

    N_VDestroy(atol);          /* Free the atol vector */

    return nfe;
}


/*
 * CPU routines
 */
/* RHS routine used in CVODE */
static int cF_RHS(realtype t, N_Vector y_in, N_Vector ydot_in,
                  void *user_data){

    BL_PROFILE_VAR("fKernelSpec()", fKernelSpec);

    /* Get Device pointers for Kernel call */
#if defined(AMREX_USE_CUDA)
    realtype *yvec_d      = N_VGetDeviceArrayPointer_Cuda(y_in);
    realtype *ydot_d      = N_VGetDeviceArrayPointer_Cuda(ydot_in);
#elif defined(AMREX_USE_HIP)
    realtype *yvec_d      = N_VGetDeviceArrayPointer_Hip(y_in);
    realtype *ydot_d      = N_VGetDeviceArrayPointer_Hip(ydot_in);
#endif

    // allocate working space
    UserData udata = static_cast<CVodeUserData*>(user_data);
    udata->dt_save = t;

    const auto ec = Gpu::ExecutionConfig(udata->ncells);
    //launch_global<<<ec.numBlocks, ec.numThreads, ec.sharedMem, udata->stream>>>(
    launch_global<<<udata->nbBlocks, udata->nbThreads, ec.sharedMem, udata->stream>>>(
        [=] AMREX_GPU_DEVICE () noexcept {
            for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
                     icell < udata->ncells; icell += stride) {
                         fKernelSpec(icell, udata->dt_save,  udata->ireactor_type,
                                     yvec_d, ydot_d,
                                     udata->rhoe_init_d,
                                     udata->rhoesrc_ext_d,
                                     udata->rYsrc_d);
        }
    });

    Gpu::Device::streamSynchronize();

    BL_PROFILE_VAR_STOP(fKernelSpec);

    return(0);
}


#ifdef AMREX_USE_CUDA
static int Precond(realtype tn, N_Vector u, N_Vector fu, booleantype jok,
                   booleantype *jcurPtr, realtype gamma, void *user_data) {

    BL_PROFILE_VAR("Precond()", Precond);

    cudaError_t cuda_status = cudaSuccess;
    size_t workspaceInBytes, internalDataInBytes;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    workspaceInBytes = 0;
    internalDataInBytes = 0;

    /* Get Device pointers for Kernel call */
    realtype *u_d      = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *udot_d   = N_VGetDeviceArrayPointer_Cuda(fu);

    // allocate working space
    UserData udata = static_cast<CVodeUserData*>(user_data);
    udata->gamma = gamma;

    BL_PROFILE_VAR("fKernelComputeAJ()", fKernelComputeAJ);
    if (jok) {
        /* GPU tests */
        const auto ec = Gpu::ExecutionConfig(udata->ncells);
        launch_global<<<udata->nbBlocks, udata->nbThreads, ec.sharedMem, udata->stream>>>(
        [=] AMREX_GPU_DEVICE () noexcept {
            for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
                 icell < udata->ncells; icell += stride) {
                     fKernelComputeAJsys(icell, user_data, u_d, udata->csr_val_d);
            }
        });
        *jcurPtr = SUNFALSE;
    } else {
        const auto ec = Gpu::ExecutionConfig(udata->ncells);
        launch_global<<<udata->nbBlocks, udata->nbThreads, ec.sharedMem, udata->stream>>>(
            [=] AMREX_GPU_DEVICE () noexcept {
                for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
                     icell < udata->ncells; icell += stride) {
                         fKernelComputeallAJ(icell, user_data, u_d, udata->csr_val_d);
                }
        });
        *jcurPtr = SUNTRUE;
    }

    cuda_status = cudaStreamSynchronize(udata->stream);
    assert(cuda_status == cudaSuccess);
    BL_PROFILE_VAR_STOP(fKernelComputeAJ);

    BL_PROFILE_VAR("InfoBatched(inPrecond)", InfoBatched);
    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(udata->cusolverHandle,udata->neqs_per_cell+1,udata->neqs_per_cell+1,
                                (udata->NNZ),
                                udata->descrA,
                                udata->csr_val_d,
                                udata->csr_row_count_d,
                                udata->csr_col_index_d,
                                udata->ncells,
                                udata->info,
                                &internalDataInBytes,
                                &workspaceInBytes);

    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    BL_PROFILE_VAR_STOP(InfoBatched);

    BL_PROFILE_VAR_STOP(Precond);

    return(0);
}
#endif



#ifdef AMREX_USE_CUDA
static int PSolve(realtype tn, N_Vector u, N_Vector fu, N_Vector r, N_Vector z,
                  realtype gamma, realtype delta, int lr, void *user_data)
{
    BL_PROFILE_VAR("Psolve()", cusolverPsolve);

    cudaError_t cuda_status = cudaSuccess;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    UserData udata = static_cast<CVodeUserData*>(user_data);

    realtype *z_d      = N_VGetDeviceArrayPointer_Cuda(z);
    realtype *r_d      = N_VGetDeviceArrayPointer_Cuda(r);

    cusolver_status = cusolverSpDcsrqrsvBatched(udata->cusolverHandle,udata->neqs_per_cell+1,udata->neqs_per_cell+1,
                               (udata->NNZ),
                               udata->descrA,
                               udata->csr_val_d,
                               udata->csr_row_count_d,
                               udata->csr_col_index_d,
                               r_d,
                               z_d,
                               udata->ncells,
                               udata->info,
                               udata->buffer_qr);

    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    N_VCopyFromDevice_Cuda(z);
    N_VCopyFromDevice_Cuda(r);

    BL_PROFILE_VAR_STOP(cusolverPsolve);

    /* Checks */
    //if (udata->iverbose > 4) {
    //    for(int batchId = 0 ; batchId < udata->ncells; batchId++){
    //        // measure |bj - Aj*xj|
    //        realtype *csrValAj = (udata->csr_val_d) + batchId * (udata->NNZ);
    //        double *xj       = N_VGetHostArrayPointer_Cuda(z) + batchId * (udata->neqs_per_cell+1);
    //        double *bj       = N_VGetHostArrayPointer_Cuda(r) + batchId * (udata->neqs_per_cell+1);
    //        // sup| bj - Aj*xj|
    //        double sup_res = 0;
    //        for(int row = 0 ; row < (udata->neqs_per_cell+1) ; row++){
    //            printf("\n     row %d: ", row);
    //            const int start = udata->csr_row_count_d[row] - 1;
    //            const int end = udata->csr_row_count_d[row +1] - 1;
    //            double Ax = 0.0; // Aj(row,:)*xj
    //            for(int colidx = start ; colidx < end ; colidx++){
    //                const int col = udata->csr_col_index_d[colidx] - 1;
    //                const double Areg = csrValAj[colidx];
    //                const double xreg = xj[col];
    //                printf("  (%d, %14.8e, %14.8e, %14.8e) ", col,Areg,xreg,bj[row] );
    //                Ax = Ax + Areg * xreg;
    //            }
    //            double rresidi = bj[row] - Ax;
    //            sup_res = (sup_res > fabs(rresidi))? sup_res : fabs(rresidi);
    //        }
    //        printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
    //    }
    //}

    return(0);
}
#endif

#ifdef AMREX_USE_CUDA
/* Will not work for cuSolver_sparse_solve right now */
static int cJac(realtype t, N_Vector y_in, N_Vector fy, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    cudaError_t cuda_status = cudaSuccess;

    /* allocate working space */
    UserData udata = static_cast<CVodeUserData*>(user_data);

    /* Get Device pointers for Kernel call */
    realtype *yvec_d       = N_VGetDeviceArrayPointer_Cuda(y_in);
    realtype *ydot_d       = N_VGetDeviceArrayPointer_Cuda(fy);

    /* Jdata */
    realtype *Jdata;

    /* Fixed Indices and Pointers for Jacobian Matrix */
    BL_PROFILE_VAR("cJac::SparsityStuff",cJacSparsityStuff);
    Jdata   = SUNMatrix_cuSparse_Data(J);
    if ((SUNMatrix_cuSparse_Rows(J) != (udata->neqs_per_cell+1)*(udata->ncells)) ||
       (SUNMatrix_cuSparse_Columns(J) != (udata->neqs_per_cell+1)*(udata->ncells)) ||
       (SUNMatrix_cuSparse_NNZ(J) != udata->ncells * udata->NNZ )) {
            Print() << "Jac error: matrix is wrong size!\n";
            return 1;
    }
    BL_PROFILE_VAR_STOP(cJacSparsityStuff);

    BL_PROFILE_VAR("Jacobian()", fKernelJac );
    const auto ec = Gpu::ExecutionConfig(udata->ncells);
    launch_global<<<udata->nbBlocks, udata->nbThreads, ec.sharedMem, udata->stream>>>(
        [=] AMREX_GPU_DEVICE () noexcept {
            for (int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
                 icell < udata->ncells; icell += stride) {
                     fKernelComputeAJchem(icell, user_data, yvec_d, Jdata);
            }
    });
    cuda_status = cudaStreamSynchronize(udata->stream);
    assert(cuda_status == cudaSuccess);
    BL_PROFILE_VAR_STOP(fKernelJac);

    return(0);

}
#endif

/**********************************/
/*
 * GPU kernels
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelSpec(int icell, double dt_save, int reactor_type,
            realtype *yvec_d, realtype *ydot_d,
            double *rhoe_init, double *rhoesrc_ext, double *rYs)
{
  int offset = icell * (NUM_SPECIES + 1);

  /* MW CGS */
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  /* rho */
  Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
      rho_pt = rho_pt + yvec_d[offset + n];
  }

  /* Yks, C CGS*/
  GpuArray<Real,NUM_SPECIES> massfrac;
  for (int i = 0; i < NUM_SPECIES; i++){
      massfrac[i] = yvec_d[offset + i] / rho_pt;
  }

  /* NRG CGS */
  Real nrg_pt = (rhoe_init[icell] + rhoesrc_ext[icell] * dt_save) /rho_pt;

  /* temp */
  Real temp_pt = yvec_d[offset + NUM_SPECIES];

  /* Additional var needed */
  GpuArray<Real,NUM_SPECIES> ei_pt;
  Real Cv_pt = 0.0;
  auto eos = pele::physics::PhysicsType::eos();
  if ( reactor_type == 1){
      /* UV REACTOR */
      eos.EY2T(nrg_pt, massfrac.arr, temp_pt);
      eos.TY2Cv(temp_pt, massfrac.arr, Cv_pt);
      eos.T2Ei(temp_pt, ei_pt.arr);
  } else {
      /* HP REACTOR */
      eos.HY2T(nrg_pt, massfrac.arr, temp_pt);
      eos.TY2Cp(temp_pt, massfrac.arr, Cv_pt);
      eos.T2Hi(temp_pt, ei_pt.arr);
  }

  GpuArray<Real,NUM_SPECIES> cdots_pt;
  eos.RTY2WDOT(rho_pt, temp_pt, massfrac.arr, cdots_pt.arr);

  /* Fill ydot vect */
  ydot_d[offset + NUM_SPECIES] = rhoesrc_ext[icell];
  for (int i = 0; i < NUM_SPECIES; i++){
      ydot_d[offset + i]           = cdots_pt[i] + rYs[icell * NUM_SPECIES + i];
      ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES]  - ydot_d[offset + i] * ei_pt[i];
  }
  ydot_d[offset + NUM_SPECIES] = ydot_d[offset + NUM_SPECIES] /(rho_pt * Cv_pt);
}


#ifdef AMREX_USE_CUDA
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJchem(int ncell, void *user_data, realtype *u_d, realtype *Jdata)
{
  UserData udata = static_cast<CVodeUserData*>(user_data);


  int u_offset      = ncell * (NUM_SPECIES + 1);
  int jac_offset = ncell * (udata->NNZ);

  realtype* u_curr       = u_d + u_offset;
  realtype* csr_jac_cell = Jdata + jac_offset;

  /* MW CGS */
  Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  /* rho */
  Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
      rho_pt = rho_pt + u_curr[n];
  }

  /* Yks, C CGS*/
  GpuArray<Real,NUM_SPECIES> massfrac;
  for (int i = 0; i < NUM_SPECIES; i++){
      massfrac[i] = u_curr[i] / rho_pt;
  }

  /* temp */
  Real temp_pt = u_curr[NUM_SPECIES];

  /* Additional var needed */
  int consP;
  if (udata->ireactor_type == 1){
      consP = 0 ;
  } else {
      consP = 1;
  }
  GpuArray<Real,(NUM_SPECIES+1)*(NUM_SPECIES+1)> Jmat_pt;
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2JAC(rho_pt, temp_pt, massfrac.arr, Jmat_pt.arr, consP);

  /* renorm the DenseMat */
  for (int i = 0; i < udata->neqs_per_cell; i++){
      for (int k = 0; k < udata->neqs_per_cell; k++){
          Jmat_pt[k*(udata->neqs_per_cell+1)+i] = Jmat_pt[k*(udata->neqs_per_cell+1)+i] * mw[i] / mw[k];
      }
      Jmat_pt[i*(udata->neqs_per_cell+1)+udata->neqs_per_cell] = Jmat_pt[i*(udata->neqs_per_cell+1)+udata->neqs_per_cell] / mw[i];
      Jmat_pt[udata->neqs_per_cell*(udata->neqs_per_cell+1)+i] = Jmat_pt[udata->neqs_per_cell*(udata->neqs_per_cell+1)+i] * mw[i];
  }
  /* Fill the Sps Mat */
  int nbVals;
  for (int i = 1; i < udata->neqs_per_cell+2; i++) {
      nbVals = udata->csr_row_count_d[i]-udata->csr_row_count_d[i-1];
      for (int j = 0; j < nbVals; j++) {
          int idx_cell = udata->csr_col_index_d[ udata->csr_row_count_d[i-1] + j ];
              csr_jac_cell[ udata->csr_row_count_d[i-1] + j ] = Jmat_pt[ idx_cell * (udata->neqs_per_cell+1) + i - 1 ];
      }
  }

}
#endif


#ifdef AMREX_USE_CUDA
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeallAJ(int ncell, void *user_data, realtype *u_d, double * csr_val_arg)
{
  UserData udata = static_cast<CVodeUserData*>(user_data);

  Real mw[NUM_SPECIES];
  GpuArray<Real,NUM_SPECIES> massfrac, activity;
  GpuArray<Real,(NUM_SPECIES+1)*(NUM_SPECIES+1)> Jmat_pt;
  Real rho_pt, temp_pt;

  int u_offset   = ncell * (NUM_SPECIES + 1);
  int jac_offset = ncell * (udata->NNZ);

  realtype* u_curr = u_d + u_offset;
  realtype* csr_jac_cell = udata->csr_jac_d + jac_offset;
  realtype* csr_val_cell = csr_val_arg + jac_offset;

  /* MW CGS */
  get_mw(mw);

  /* rho */
  rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
      rho_pt = rho_pt + u_curr[n];
  }

  /* Yks, C CGS*/
  for (int i = 0; i < NUM_SPECIES; i++){
      massfrac[i] = u_curr[i] / rho_pt;
  }

  /* temp */
  temp_pt = u_curr[NUM_SPECIES];

  /* Activities */
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2C(rho_pt, temp_pt, massfrac.arr, activity.arr);

  /* Additional var needed */
  int consP;
  if (udata->ireactor_type == 1){
      consP = 0 ;
  } else {
      consP = 1;
  }
  DWDOT_SIMPLIFIED(Jmat_pt.arr, activity.arr, &temp_pt, &consP);

  /* renorm the DenseMat */
  for (int i = 0; i < udata->neqs_per_cell; i++){
      for (int k = 0; k < udata->neqs_per_cell; k++){
          Jmat_pt[k*(udata->neqs_per_cell+1)+i] = Jmat_pt[k*(udata->neqs_per_cell+1)+i] * mw[i] / mw[k];
      }
      Jmat_pt[i*(udata->neqs_per_cell+1)+udata->neqs_per_cell] = Jmat_pt[i*(udata->neqs_per_cell+1)+udata->neqs_per_cell] / mw[i];
      Jmat_pt[udata->neqs_per_cell*(udata->neqs_per_cell+1)+i] = Jmat_pt[udata->neqs_per_cell*(udata->neqs_per_cell+1)+i] * mw[i];
  }
  /* Fill the Sps Mat */
  int nbVals;
  for (int i = 1; i < udata->neqs_per_cell+2; i++) {
      nbVals = udata->csr_row_count_d[i]-udata->csr_row_count_d[i-1];
      for (int j = 0; j < nbVals; j++) {
          int idx = udata->csr_col_index_d[ udata->csr_row_count_d[i-1] + j - 1 ] - 1;
          /* Scale by -gamma */
          /* Add identity matrix */
          if (idx == (i-1)) {
              csr_val_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = 1.0 - (udata->gamma) * Jmat_pt[ idx * (udata->neqs_per_cell+1) + idx ];
              csr_jac_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = Jmat_pt[ idx * (udata->neqs_per_cell+1) + idx ];
          } else {
              csr_val_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = - (udata->gamma) * Jmat_pt[ idx * (udata->neqs_per_cell+1) + i-1 ];
              csr_jac_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = Jmat_pt[ idx * (udata->neqs_per_cell+1) + i-1 ];
          }
      }
  }

}
#endif

#ifdef AMREX_USE_CUDA
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJsys(int ncell, void *user_data, realtype *u_d, double * csr_val_arg)
{
  UserData udata = static_cast<CVodeUserData*>(user_data);

  int jac_offset = ncell * (udata->NNZ);

  realtype* csr_jac_cell = udata->csr_jac_d + jac_offset;
  realtype* csr_val_cell = csr_val_arg + jac_offset;

  int nbVals;
  for (int i = 1; i < udata->neqs_per_cell+2; i++) {
      nbVals = udata->csr_row_count_d[i]-udata->csr_row_count_d[i-1];
      for (int j = 0; j < nbVals; j++) {
          int idx = udata->csr_col_index_d[ udata->csr_row_count_d[i-1] + j - 1 ] - 1;
          /* Scale by -gamma */
          /* Add identity matrix */
          if (idx == (i-1)) {
              csr_val_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = 1.0 - (udata->gamma) * csr_jac_cell[ udata->csr_row_count_d[i-1] + j - 1 ];
          } else {
              csr_val_cell[ udata->csr_row_count_d[i-1] + j - 1 ] = - (udata->gamma) * csr_jac_cell[ udata->csr_row_count_d[i-1] + j - 1 ];
          }
      }
  }

}
#endif


#ifdef AMREX_USE_CUDA
__global__
void
fKernelDenseSolve(int ncell, realtype *x_d, realtype *b_d,
          int subsys_size, int subsys_nnz, realtype *csr_val)
{

  int stride = blockDim.x*gridDim.x;

  for (int icell = blockDim.x*blockIdx.x+threadIdx.x;
           icell < ncell; icell += stride) {
               int offset   = icell * subsys_size;
               int offset_A = icell * subsys_nnz;

               realtype* csr_val_cell = csr_val + offset_A;
               realtype* x_cell       = x_d + offset;
               realtype* b_cell       = b_d + offset;

               /* Solve the subsystem of the cell */
               sgjsolve(csr_val_cell, x_cell, b_cell);
  }
}
#endif

/*
 * OTHERS
*/
/* Get and print some final statistics */
static void PrintFinalStats(void *cvodeMem)
{
  long lenrw, leniw ;
  long lenrwLS, leniwLS;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  int flag;

  flag = CVodeGetWorkSpace(cvodeMem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1);
  flag = CVodeGetNumSteps(cvodeMem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvodeMem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvodeMem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvodeMem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvodeMem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvodeMem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVodeGetLinWorkSpace(cvodeMem, &lenrwLS, &leniwLS);
  check_flag(&flag, "CVodeGetLinWorkSpace", 1);
  flag = CVodeGetNumLinIters(cvodeMem, &nli);
  check_flag(&flag, "CVodeGetNumLinIters", 1);
  //flag = CVodeGetNumJacEvals(cvodeMem, &nje);
  //check_flag(&flag, "CVodeGetNumJacEvals", 1);
  flag = CVodeGetNumLinRhsEvals(cvodeMem, &nfeLS);
  check_flag(&flag, "CVodeGetNumLinRhsEvals", 1);

  flag = CVodeGetNumPrecEvals(cvodeMem, &npe);
  check_flag(&flag, "CVodeGetNumPrecEvals", 1);
  flag = CVodeGetNumPrecSolves(cvodeMem, &nps);
  check_flag(&flag, "CVodeGetNumPrecSolves", 1);

  flag = CVodeGetNumLinConvFails(cvodeMem, &ncfl);
  check_flag(&flag, "CVodeGetNumLinConvFails", 1);

#ifdef _OPENMP
  Print() <<"\nFinal Statistics: " << "(thread:" << omp_get_thread_num() << ", ";
  Print() << "cvodeMem:" << cvodeMem << ")\n";
#else
  Print() <<"\nFinal Statistics:\n";
#endif
  Print() <<"lenrw      = " << lenrw   <<"    leniw         = " << leniw   << "\n";
  Print() <<"lenrwLS    = " << lenrwLS <<"    leniwLS       = " << leniwLS << "\n";
  Print() <<"nSteps     = " << nst     <<"\n";
  Print() <<"nRHSeval   = " << nfe     <<"    nLinRHSeval   = " << nfeLS   << "\n";
  Print() <<"nnLinIt    = " << nni     <<"    nLinIt        = " << nli     << "\n";
  Print() <<"nLinsetups = " << nsetups <<"    nErrtf        = " << netf    << "\n";
  Print() <<"nPreceval  = " << npe     <<"    nPrecsolve    = " << nps     << "\n";
  Print() <<"nConvfail  = " << ncfn    <<"    nLinConvfail  = " << ncfl    << "\n\n";

}

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */
static int check_flag(void *flagvalue, const char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
      if (ParallelDescriptor::IOProcessor()) {
          fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                  funcname);
          Abort("abort");
      }
      return(1);
  }
  /* Check if flag < 0 */
  else if (opt == 1) {
      errflag = (int *) flagvalue;
      if (*errflag < 0) {
          if (ParallelDescriptor::IOProcessor()) {
              fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                      funcname, *errflag);
              Abort("abort");
          }
          return(1);
      }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
      if (ParallelDescriptor::IOProcessor()) {
          fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                  funcname);
          Abort("abort");
      }
      return(1);
  }

  return(0);
}
/* End of file  */
