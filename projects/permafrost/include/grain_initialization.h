#ifndef GRAIN_INITIALIZATION_H
#define GRAIN_INITIALIZATION_H

#include "NASA_types.h"
#include <math.h>

/* Initialize sediment grains (general and gravity-based approaches) */
PetscErrorCode InitialSedGrains(IGA iga, AppCtx *user);
PetscErrorCode InitialSedGrainsGravity(IGA iga, AppCtx *user);

/* Initialize ice grains (either reading from file or generating randomly) */
PetscErrorCode InitialIceGrains(IGA iga, AppCtx *user);

#endif // GRAIN_INITIALIZATION_H