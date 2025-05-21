#include "env_helper.h"
#include <stdlib.h>

PetscErrorCode ParseEnvironment(AppCtx *user,
                                PetscInt *Nx, PetscInt *Ny, PetscInt *Nz,
                                PetscReal *Lx, PetscReal *Ly, PetscReal *Lz,
                                PetscReal *delt_t, PetscReal *t_final, PetscInt *n_out,
                                PetscReal *humidity, PetscReal *temp,
                                PetscReal grad_temp0[3],
                                PetscInt *dim, PetscReal *eps)
{
    const char *Nx_str          = getenv("Nx");
    const char *Ny_str          = getenv("Ny");
    const char *Nz_str          = getenv("Nz");
    const char *Lx_str          = getenv("Lx");
    const char *Ly_str          = getenv("Ly");
    const char *Lz_str          = getenv("Lz");
    const char *delt_t_str      = getenv("delt_t");
    const char *t_final_str     = getenv("t_final");
    const char *n_out_str       = getenv("n_out");
    const char *humidity_str    = getenv("humidity");
    const char *temp_str        = getenv("temp");
    const char *grad_temp0X_str = getenv("grad_temp0X");
    const char *grad_temp0Y_str = getenv("grad_temp0Y");
    const char *grad_temp0Z_str = getenv("grad_temp0Z");
    const char *dim_str         = getenv("dim");
    const char *eps_str         = getenv("eps");

    if (!Nx_str || !Ny_str || !Nz_str || !Lx_str || !Ly_str || !Lz_str ||
        !delt_t_str || !t_final_str || !n_out_str || !humidity_str || !temp_str ||
        !grad_temp0X_str || !grad_temp0Y_str || !grad_temp0Z_str || !dim_str || !eps_str) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: One or more environment variables are not set.\n");
        return PETSC_ERR_ARG_NULL;
    }

    char *endptr;
    *Nx = strtol(Nx_str, &endptr, 10);
    *Ny = strtol(Ny_str, &endptr, 10);
    *Nz = strtol(Nz_str, &endptr, 10);
    *Lx = strtod(Lx_str, &endptr);
    *Ly = strtod(Ly_str, &endptr);
    *Lz = strtod(Lz_str, &endptr);
    *delt_t = strtod(delt_t_str, &endptr);
    *t_final = strtod(t_final_str, &endptr);
    *n_out = strtol(n_out_str, &endptr, 10);
    *humidity = strtod(humidity_str, &endptr);
    *temp = strtod(temp_str, &endptr);
    grad_temp0[0] = strtod(grad_temp0X_str, &endptr);
    grad_temp0[1] = strtod(grad_temp0Y_str, &endptr);
    grad_temp0[2] = strtod(grad_temp0Z_str, &endptr);
    *dim = strtol(dim_str, &endptr, 10);
    *eps = strtod(eps_str, &endptr);

    // Also update the AppCtx fields for convenience.
    user->Nx = *Nx;
    user->Ny = *Ny;
    user->Nz = *Nz;
    user->Lx = *Lx;
    user->Ly = *Ly;
    user->Lz = *Lz;
    user->temp0 = *temp;
    user->hum0 = *humidity;
    user->eps = *eps;
    user->dim = *dim;
    user->grad_temp0[0] = grad_temp0[0];
    user->grad_temp0[1] = grad_temp0[1];
    user->grad_temp0[2] = grad_temp0[2];

    return 0;
}