#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "NASA_types.h"

/* Residual evaluation for the system */
PetscErrorCode Residual(IGAPoint pnt, PetscReal shift, const PetscScalar *V,
                          PetscReal t, const PetscScalar *U, PetscScalar *Re,
                          void *ctx);

/* Jacobian evaluation for the system */
PetscErrorCode Jacobian(IGAPoint pnt, PetscReal shift, const PetscScalar *V,
                          PetscReal t, const PetscScalar *U, PetscScalar *Je,
                          void *ctx);

/* Computes integrated scalar quantities over the domain */
PetscErrorCode Integration(IGAPoint pnt, const PetscScalar *U, PetscInt n,
                           PetscScalar *S, void *ctx);

#endif // ASSEMBLY_H