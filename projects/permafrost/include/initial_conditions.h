#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include "NASA_types.h"

/* Form initial soil distribution for 2D and 3D problems */
PetscErrorCode FormInitialSoil2D(IGA igaS, Vec S, AppCtx *user);
PetscErrorCode FormInitialSoil3D(IGA igaS, Vec S, AppCtx *user);

/* Form initial condition for the primary field variables in 2D and 3D */
PetscErrorCode FormInitialCondition2D(IGA iga, PetscReal t, Vec U, AppCtx *user,
                                      const char datafile[], const char dataPF[]);
PetscErrorCode FormLayeredInitialCondition2D(IGA iga, PetscReal t, Vec U, 
                                            AppCtx *user, const char datafile[],
                                            const char dataPF[]);
PetscErrorCode FormInitialCondition3D(IGA iga, PetscReal t, Vec U, AppCtx *user,
                                      const char datafile[], const char dataPF[]);

#endif // INITIAL_CONDITIONS_H