#ifndef MONITORING_H
#define MONITORING_H

#include "NASA_types.h"

/* Monitor function for TS solver */
PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal t, Vec U, void *mctx);

/* Output monitor function for writing solution data */
PetscErrorCode OutputMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *mctx);

#endif // MONITORING_H