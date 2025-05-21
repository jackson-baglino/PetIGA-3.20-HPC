#ifndef ENV_HELPER_H
#define ENV_HELPER_H

#include "NASA_main.h"

/* Parses required environment variables and updates the AppCtx and output parameters. */
PetscErrorCode ParseEnvironment(AppCtx *user,
                                PetscInt *Nx, PetscInt *Ny, PetscInt *Nz,
                                PetscReal *Lx, PetscReal *Ly, PetscReal *Lz,
                                PetscReal *delt_t, PetscReal *t_final, PetscInt *n_out,
                                PetscReal *humidity, PetscReal *temp,
                                PetscReal grad_temp0[3],
                                PetscInt *dim, PetscReal *eps);

#endif // ENV_HELPER_H