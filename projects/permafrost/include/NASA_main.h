#ifndef NASA_MAIN_H
#define NASA_MAIN_H

// Standard library includes
#include <math.h>
#include <mpi.h>

// PETSc and PetIGA includes
#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"

// Project-specific includes
#include "NASA_types.h"
#include "material_properties.h"
#include "assembly.h"
#include "monitoring.h"
#include "grain_initialization.h"
#include "initial_conditions.h"
#include "snes_convergence.h"
#include "env_helper.h"

#endif // NASA_MAIN_H