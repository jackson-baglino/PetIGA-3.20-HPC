#include "snes_convergence.h"

PetscErrorCode SNESDOFConvergence(SNES snes, PetscInt it_number, PetscReal xnorm, 
  PetscReal gnorm, PetscReal fnorm, SNESConvergedReason *reason, void *cctx)
{
 /* ***************************************************************************
 * Custom convergence check for SNES solver.
 * 
 * This function monitors convergence by computing:
 * - The L2 norm of each component of the residual vector.
 * - (Optional) The L2 norm of the solution vector and update vector.
 * - Checks convergence criteria based on relative and absolute tolerances.
 * 
 * Inputs:
 * - snes: SNES solver context.
 * - it_number: Current iteration number.
 * - xnorm: Norm of the solution vector (unused here).
 * - gnorm: Norm of the gradient (unused here).
 * - fnorm: Norm of the function (residual norm).
 * - reason: Pointer to the convergence status.
 * - cctx: Pointer to user-defined application context (AppCtx).
 *
 * Outputs:
 * - Updates `reason` if convergence is achieved.
 *************************************************************************** */

 PetscFunctionBegin;  // Marks function entry for PETSc error handling.
 PetscErrorCode ierr;
 AppCtx *user = (AppCtx *)cctx; // Cast context to user-defined struct.

 // Define vectors for residual, solution, and solution update
 Vec Res, Sol, Sol_upd;
 PetscScalar n2dof0, n2dof1, n2dof2; // Norms of the residual vector components
 PetscScalar solv, solupdv;          // Norms of the solution and update vector

 // Retrieve the residual vector from SNES and compute its component norms.
 ierr = SNESGetFunction(snes, &Res, 0, 0);CHKERRQ(ierr);
 ierr = VecStrideNorm(Res, 0, NORM_2, &n2dof0);CHKERRQ(ierr);
 ierr = VecStrideNorm(Res, 1, NORM_2, &n2dof1);CHKERRQ(ierr);
 ierr = VecStrideNorm(Res, 2, NORM_2, &n2dof2);CHKERRQ(ierr);

 // If temperature-dependent initial conditions are active, compute solution and update norms.
 if (user->flag_tIC == 1) {
   ierr = SNESGetSolution(snes, &Sol);CHKERRQ(ierr);
   ierr = VecStrideNorm(Sol, 2, NORM_2, &solv);CHKERRQ(ierr);  // Norm of DOF 2 solution
   ierr = SNESGetSolutionUpdate(snes, &Sol_upd);CHKERRQ(ierr);
   ierr = VecStrideNorm(Sol_upd, 2, NORM_2, &solupdv);CHKERRQ(ierr); // Norm of DOF 2 update
 }

 // Store initial residual norms at the first iteration for relative convergence checks.
 if (it_number == 0) {
   user->norm0_0 = n2dof0;
   user->norm0_1 = n2dof1;
   user->norm0_2 = n2dof2;
   if (user->flag_tIC == 1) solupdv = solv;  // Initialize update norm to solution norm.
 }

 // Print iteration information and norm values for debugging.
 PetscPrintf(PETSC_COMM_WORLD, "    IT_NUMBER: %d ", it_number);
 PetscPrintf(PETSC_COMM_WORLD, "    fnorm: %.4e \n", fnorm);
 PetscPrintf(PETSC_COMM_WORLD, "    n0: %.2e r %.1e ", n2dof0, n2dof0 / user->norm0_0);
 PetscPrintf(PETSC_COMM_WORLD, "  n1: %.2e r %.1e ", n2dof1, n2dof1 / user->norm0_1);
 if (user->flag_tIC == 1)
   PetscPrintf(PETSC_COMM_WORLD, "  x2: %.2e s %.1e \n", solv, solupdv / solv);
 else
   PetscPrintf(PETSC_COMM_WORLD, "  n2: %.2e r %.1e \n", n2dof2, n2dof2 / user->norm0_2);

 // Retrieve SNES solver tolerances (absolute, relative, step-size)
 PetscScalar atol, rtol, stol;
 PetscInt maxit, maxf;
 ierr = SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf);CHKERRQ(ierr);
 
 // If a previous timestep required reduction, increase relative tolerance.
 if (snes->prev_dt_red == 1) rtol *= 10.0;

 // Convergence check based on flag_it0 setting
 if (user->flag_it0 == 1) {
   atol = 1.0e-12;  // Set absolute tolerance
 } else {
   atol = 1.0e-20;  // More strict absolute tolerance
 }

 // Check for convergence using relative and absolute norms
 if ((n2dof0 <= rtol * user->norm0_0 || n2dof0 < atol) &&
     (n2dof1 <= rtol * user->norm0_1 || n2dof1 < atol) &&
     (n2dof2 <= rtol * user->norm0_2 || n2dof2 < atol)) {
   *reason = SNES_CONVERGED_FNORM_RELATIVE;
 }

 PetscFunctionReturn(0);  // Exit function safely.
}