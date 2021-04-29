
enum class LinearSolverMethod {
  sparse_direct_start = 0,            // sparse direct methods -- start
  sparse_custom = 1,
  sparse_batched_cusolver = 2,
  dense_direct_start = 20,            // dense direct methods -- start
  dense_batched_magma = 21,
  iterative_matrix_free = 40,         // iterative matrix free methods -- start 
  iterative_gmres = 41
};

class SundialsLinearSolver {

public:
    SundialsLinearSolver(CVodeUserData* user_data, LinearSolverMethod method = LinearSolverMethod::iterative_gmres);

private:
    SUNLinearSolver LS;
    SUNMatrix A;
  
    LinearSolverMethod method;
    CVodeUserData* user_data;
};


SundialsLinearSolver::SundialsLinearSolver(LinearSolverMethod method)
  : method(method), user_data(user_data)
{
  
}
