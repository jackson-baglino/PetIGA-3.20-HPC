#ifndef SNES_CONVERGENCE_H
#define SNES_CONVERGENCE_H

#include "NASA_main.h"

/* Custom SNES convergence test function */
PetscErrorCode SNESDOFConvergence(SNES snes, PetscInt it_number, PetscReal xnorm,
                                   PetscReal gnorm, PetscReal fnorm, SNESConvergedReason *reason, void *cctx);

#endif // SNES_CONVERGENCE_H