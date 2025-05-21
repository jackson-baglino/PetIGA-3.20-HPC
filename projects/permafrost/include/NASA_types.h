#ifndef NASA_TYPES_H
#define NASA_TYPES_H

#include <petsc.h>
#include "petiga.h"

#define SQ(x) ((x)*(x))
#define CU(x) ((x)*(x)*(x))

/* Application context structure */
typedef struct {
  IGA       iga;  // Isogeometric analysis (IGA) structure for managing geometry and basis functions

  // Physical parameters related to phase field and thermodynamics
  PetscReal eps;  // Interface width parameter for phase field method
  PetscReal mob_sub, mav;  // Mobility parameters for phase evolution
  PetscReal Etai, Etam, Etaa;  // Activation energy terms for different phases (ice, metal, air)
  PetscReal alph_sub;  // Substrate interaction coefficient
  PetscReal Lambd;  // Parameter related to thermal conductivity or latent heat (context-dependent)
  PetscReal beta_sub0, d0_sub0;  // Parameters related to phase change at the substrate

  // Thermophysical properties of different phases
  PetscReal thcond_ice, thcond_met, thcond_air;  // Thermal conductivities of ice, metal, and air
  PetscReal cp_ice, cp_met, cp_air;  // Specific heat capacities of ice, metal, and air
  PetscReal rho_ice, rho_met, rho_air;  // Densities of ice, metal, and air
  PetscReal dif_vap;  // Vapor diffusivity in air
  PetscReal lat_sub;  // Latent heat of sublimation
  PetscReal diff_sub;  // Diffusivity related to sublimation

  // Environmental conditions and threshold parameters
  PetscReal air_lim;  // Air phase fraction limit (threshold to distinguish between ice/air)
  PetscReal xi_v, xi_T;  // Characteristic non-dimensional parameters for vapor and temperature

  // Initial and boundary condition parameters
  PetscReal T_melt;  // Melting temperature of ice
  PetscReal temp0, hum0;  // Initial temperature and humidity
  PetscReal grad_temp0[3];  // Initial temperature gradient in x, y, and z directions

  // Domain size and resolution
  PetscReal Lx, Ly, Lz;  // Physical domain dimensions in x, y, and z
  PetscReal Nx, Ny, Nz;  // Number of grid points in x, y, and z directions

  // Radius of curvature parameters (possibly for computing capillary effects)
  PetscReal RCice, RCsed;  // Mean radius of curvature for ice and sediment grains
  PetscReal RCice_dev, RCsed_dev;  // Standard deviation of radius of curvature for ice and sediment

  // Arrays storing geometry information for ice and sediment grains
  PetscReal cent[3][200];  // Coordinates of ice grain centers (3D array for x, y, z positions)
  PetscReal radius[200];  // Radii of individual ice grains
  PetscReal centsed[3][200];  // Coordinates of sediment grain centers (3D array for x, y, z positions)
  PetscReal radiussed[200];  // Radii of individual sediment grains

  // Initial normal vector components (possibly for a structured interface)
  PetscReal norm0_0, norm0_1, norm0_2;  // Normal vector components (x, y, z)

  // Flags for controlling different simulation options
  PetscInt flag_it0;  // Flag for iteration control at initialization
  PetscInt flag_tIC;  // Flag for setting initial conditions
  PetscInt outp;  // Output control flag (defines what to output)
  PetscInt nsteps_IC;  // Number of initial condition timesteps
  PetscInt flag_xiT;  // Flag for including temperature-dependent terms in the model
  PetscInt flag_Tdep;  // Flag for temperature dependence of specific properties
  PetscInt flag_BC_Tfix;
  PetscInt flag_BC_rhovfix;

  // Numerical method and discretization parameters
  PetscInt p;  // Polynomial degree of basis functions (for IGA)
  PetscInt C;  // Continuity of basis functions
  PetscInt dim;  // Spatial dimension of the problem (2D or 3D)
  PetscInt periodic;  // Periodicity flag (0 = non-periodic, 1 = periodic boundaries)

  // Time stepping parameters
  PetscReal t_out;  // Output time interval
  PetscReal t_interv;  // Intermediate time step interval
  PetscReal t_IC;  // Total duration for initial condition phase

  // Counters for active ice and sediment grains
  PetscInt NCice, NCsed;  // Number of ice and sediment grains
  PetscInt n_act, n_actsed;  // Number of currently active grains (ice and sediment)

  // Arrays for field variables
  PetscReal *Phi_sed;  // Phase field for sediment grains
  PetscReal *alph;  // Alpha field, possibly phase fraction or related property
  PetscReal *mob;  // Mobility field, spatially varying

  // Flag for reading input files
  PetscInt readFlag;  // Flag to indicate whether initial data should be read from a file

} AppCtx;

/* Field definitions for node data */
typedef struct {
  PetscScalar soil;
} FieldS;

typedef struct {
  PetscScalar ice,tem,rhov;
} Field;

#endif // NASA_TYPES_H