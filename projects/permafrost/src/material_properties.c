#include "material_properties.h"
#include "NASA_types.h"

/**
 * @brief Computes the effective thermal conductivity based on phase fractions.
 * 
 * This function calculates the thermal conductivity of a material as a weighted 
 * sum of the thermal conductivities of ice, metal, and air phases. It also computes 
 * the derivative of conductivity with respect to the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase (0 to 1).
 * @param met Fraction of the metal phase (0 to 1).
 * @param cond Pointer to store the computed thermal conductivity.
 * @param dcond_ice Pointer to store the derivative of conductivity with respect to ice fraction.
 */
void ThermalCond(AppCtx *user, PetscScalar ice, PetscScalar met, 
                 PetscScalar *cond, PetscScalar *dcond_ice)
{
    // Define derivatives (initialized to 1.0, modified if phase fractions are negative)
    PetscReal dice = 1.0, dair = 1.0;

    // Compute air fraction (air = 1 - ice - metal)
    PetscReal air = 1.0 - met - ice;

    // Ensure phase fractions are non-negative (corrects numerical issues)
    if (met < 0.0) met = 0.0;
    if (ice < 0.0) { 
        ice = 0.0; 
        dice = 0.0; // If ice fraction is zero, its derivative should also be zero
    }
    if (air < 0.0) { 
        air = 0.0; 
        dair = 0.0; // If air fraction is zero, its derivative should also be zero
    }

    // Retrieve material thermal conductivities from the user context
    PetscReal cond_ice = user->thcond_ice;
    PetscReal cond_met = user->thcond_met;
    PetscReal cond_air = user->thcond_air;

    // Compute the effective thermal conductivity as a weighted sum
    if (cond) 
        (*cond) = ice * cond_ice + met * cond_met + air * cond_air;

    // Compute the derivative of thermal conductivity with respect to ice fraction
    if (dcond_ice) 
        (*dcond_ice) = cond_ice * dice - cond_air * dair;

    return; // Explicit return statement for clarity
}

/**
 * @brief Computes the effective heat capacity and its derivative with respect to ice.
 * 
 * This function calculates the heat capacity as a weighted sum of contributions 
 * from ice, metal, and air. It also computes the derivative of heat capacity 
 * with respect to the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase.
 * @param met Fraction of the metal phase.
 * @param cp Pointer to store computed heat capacity.
 * @param dcp_ice Pointer to store derivative of heat capacity with respect to ice.
 */
void HeatCap(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *cp, 
  PetscScalar *dcp_ice)
{
PetscReal dice = 1.0, dair = 1.0;
PetscReal air = 1.0 - met - ice;

// Ensure phase fractions are non-negative
if (met < 0.0) met = 0.0;
if (ice < 0.0) { ice = 0.0; dice = 0.0; }
if (air < 0.0) { air = 0.0; dair = 0.0; }

// Retrieve heat capacities
PetscReal cp_ice = user->cp_ice;
PetscReal cp_met = user->cp_met;
PetscReal cp_air = user->cp_air;

// Compute effective heat capacity
if (cp) 
(*cp) = ice * cp_ice + met * cp_met + air * cp_air;

// Compute derivative with respect to ice
if (dcp_ice) 
(*dcp_ice) = cp_ice * dice - cp_air * dair;

return;
}

/**
 * @brief Computes the effective density and its derivative with respect to ice.
 *
 * This function calculates the density as a weighted sum of contributions 
 * from ice, metal, and air. It also computes the derivative of density 
 * with respect to the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase.
 * @param met Fraction of the metal phase.
 * @param rho Pointer to store computed density.
 * @param drho_ice Pointer to store derivative of density with respect to ice.
 */
void Density(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *rho, 
  PetscScalar *drho_ice)
{
PetscReal dice = 1.0, dair = 1.0;
PetscReal air = 1.0 - met - ice;

// Ensure phase fractions are non-negative
if (met < 0.0) met = 0.0;
if (ice < 0.0) { ice = 0.0; dice = 0.0; }
if (air < 0.0) { air = 0.0; dair = 0.0; }

// Retrieve densities
PetscReal rho_ice = user->rho_ice;
PetscReal rho_met = user->rho_met;
PetscReal rho_air = user->rho_air;

// Compute effective density
if (rho) 
(*rho) = ice * rho_ice + met * rho_met + air * rho_air;

// Compute derivative with respect to ice
if (drho_ice) 
(*drho_ice) = rho_ice * dice - rho_air * dair;

return;
}

/**
 * @brief Computes vapor diffusivity and its derivative with respect to temperature.
 * 
 * @param user Pointer to the application context containing material properties.
 * @param tem Temperature in Celsius.
 * @param difvap Pointer to store computed vapor diffusivity.
 * @param d_difvap Pointer to store derivative of vapor diffusivity with respect to ice.
 */
void VaporDiffus(AppCtx *user, PetscScalar tem, PetscScalar *difvap, 
  PetscScalar *d_difvap)
{
PetscReal dif_vap = user->dif_vap;
PetscReal Kratio = (tem + 273.15) / 273.15; // Convert temperature to Kelvin ratio
PetscReal aa = 1.81;

// Compute vapor diffusivity
if (difvap) 
(*difvap) = dif_vap * pow(Kratio, aa);

// Compute derivative with respect to temperature
if (d_difvap) 
(*d_difvap) = dif_vap * aa * pow(Kratio, aa - 1.0) / 273.15;

return;
}

/**
 * @brief Computes the saturation vapor density and its derivative with respect to temperature.
 * 
 * @param user Pointer to the application context containing material properties.
 * @param tem Temperature in Celsius.
 * @param rho_vs Pointer to store computed saturation vapor density.
 * @param d_rhovs Pointer to store derivative of saturation vapor density with respect to temperature.
 */
void RhoVS_I(AppCtx *user, PetscScalar tem, PetscScalar *rho_vs, 
  PetscScalar *d_rhovs)
{
PetscReal rho_air = user->rho_air;
PetscReal Patm = 101325.0;
PetscReal bb = 0.62;
PetscReal temK = tem + 273.15; // Convert temperature to Kelvin

// Empirical coefficients for saturation pressure
PetscReal K0 = -0.5865e4, K1 = 0.2224e2, K2 = 0.1375e-1;
PetscReal K3 = -0.3403e-4, K4 = 0.2697e-7, K5 = 0.6918;

// Compute saturation vapor pressure
PetscReal Pvs = exp(K0 * pow(temK, -1.0) + K1 + K2 * temK + 
             K3 * pow(temK, 2.0) + K4 * pow(temK, 3.0) + 
             K5 * log(temK));

// Compute derivative of vapor pressure with respect to temperature
PetscReal Pvs_T = Pvs * (-K0 * pow(temK, -2.0) + K2 + 
                  2.0 * K3 * temK + 3.0 * K4 * pow(temK, 2.0) + 
                  K5 / temK);

// Compute saturation vapor density
if (rho_vs) 
(*rho_vs) = rho_air * bb * Pvs / (Patm - Pvs);

// Compute derivative with respect to temperature
if (d_rhovs) 
(*d_rhovs) = rho_air * bb * (Pvs_T * (Patm - Pvs) + Pvs * Pvs_T) / 
          ((Patm - Pvs) * (Patm - Pvs));

return;
}

/**
 * @brief Computes the free energy function for the ice phase.
 * 
 * This function calculates the free energy contribution of the ice phase in the 
 * phase-field model. It also computes the derivative of this energy with 
 * respect to the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase.
 * @param met Fraction of the metal phase.
 * @param fice Pointer to store the computed free energy for the ice phase.
 * @param dfice_ice Pointer to store the derivative of fice with respect to ice.
 */
void Fice(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fice, 
  PetscScalar *dfice_ice)
{
// Retrieve material parameters from user context
PetscReal Lambd = user->Lambd;
PetscReal etai  = user->Etai;

// Compute the air fraction
PetscReal air = 1.0 - met - ice;

// Compute the free energy function for the ice phase
if (fice) 
(*fice) = etai * ice * (1.0 - ice) * (1.0 - 2.0 * ice) + 
          2.0 * Lambd * ice * met * met * air * air;

// Compute the derivative with respect to the ice fraction
if (dfice_ice) 
(*dfice_ice) = etai * (1.0 - 6.0 * ice + 6.0 * ice * ice) + 
              2.0 * Lambd * met * met * air * air - 
              2.0 * Lambd * ice * met * met * 2.0 * air;

return;
}

/**
 * @brief Computes the phase evolution function for the water phase (metal fraction).
 * 
 * This function models the free energy contribution of the water (metal) phase 
 * in a phase-field simulation. It also computes the derivative with respect to 
 * the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase.
 * @param met Fraction of the metal phase.
 * @param fwat Pointer to store the computed phase function for water.
 * @param dfwat_ice Pointer to store the derivative of fwat with respect to ice.
 */
void Fwat(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fwat, 
          PetscScalar *dfwat_ice)
{
    // Retrieve material parameters from user context
    PetscReal Lambd = user->Lambd;
    PetscReal etam  = user->Etam;

    // Compute the air fraction
    PetscReal air = 1.0 - met - ice;

    // Compute the phase evolution function for the water phase
    if (fwat) 
        (*fwat) = etam * met * (1.0 - met) * (1.0 - 2.0 * met) + 
                  2.0 * Lambd * ice * ice * met * air * air;

    // Compute the derivative with respect to the ice fraction
    if (dfwat_ice) {
        (*dfwat_ice) = 2.0 * Lambd * 2.0 * ice * met * air * air 
                     - 2.0 * Lambd * ice * ice * met * 2.0 * air;
    }

    return;
}

/**
 * @brief Computes the phase evolution function for the air phase.
 * 
 * This function models the free energy contribution of the air phase in a 
 * phase-field simulation. It also computes the derivative with respect to 
 * the ice fraction.
 *
 * @param user Pointer to the application context containing material properties.
 * @param ice Fraction of the ice phase.
 * @param met Fraction of the metal phase.
 * @param fair Pointer to store the computed phase function for air.
 * @param dfair_ice Pointer to store the derivative of fair with respect to ice.
 */
void Fair(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fair, 
          PetscScalar *dfair_ice)
{
    // Retrieve material parameters from user context
    PetscReal Lambd = user->Lambd;
    PetscReal etaa  = user->Etaa;

    // Compute the air fraction
    PetscReal air = 1.0 - met - ice;

    // Compute the phase evolution function for the air phase
    if (fair) 
        (*fair) = etaa * air * (1.0 - air) * (1.0 - 2.0 * air) + 
                  2.0 * Lambd * ice * ice * met * met * air;

    // Compute the derivative with respect to the ice fraction
    if (dfair_ice) {
        (*dfair_ice)  = -etaa * (1.0 - air) * (1.0 - 2.0 * air) 
                      + etaa * air * (1.0 - 2.0 * air) 
                      + etaa * air * (1.0 - air) * 2.0;
        (*dfair_ice) += 2.0 * Lambd * 2.0 * ice * met * met * air 
                      - 2.0 * Lambd * ice * ice * met * met;
    }

    return;
}

/**
 * @brief Computes σ₀ (sigma zero) as a function of temperature using lookup tables and interpolation.
 * 
 * This function determines the value of σ₀ based on predefined temperature values 
 * using a logarithmic interpolation technique. If the temperature is out of range, 
 * a warning is printed, and the function assigns the closest available value.
 *
 * @param temp Temperature in Celsius.
 * @param sigm0 Pointer to store the computed σ₀ value.
 */
void Sigma0(PetscScalar temp, PetscScalar *sigm0)
{
    // Lookup table for σ₀ values at different temperatures
    PetscReal sig[10], tem[10];

    // Predefined values for σ₀ corresponding to specific temperatures
    sig[0] = 3.0e-3;  sig[1] = 4.1e-3;  sig[2] = 5.5e-3;  sig[3] = 8.0e-3;  sig[4] = 4.0e-3;
    sig[5] = 6.0e-3;  sig[6] = 3.5e-2;  sig[7] = 7.0e-2;  sig[8] = 1.1e-1;  sig[9] = 0.75;

    // Corresponding temperature values (in Celsius)
    tem[0] = -0.0001;  tem[1] = -2.0;   tem[2] = -4.0;   tem[3] = -6.0;   tem[4] = -7.0;
    tem[5] = -10.0;    tem[6] = -20.0;  tem[7] = -30.0;  tem[8] = -40.0;  tem[9] = -100.0;

    PetscInt ii, interv = 0;
    PetscReal t0, t1, s0, s1;

    // Warn user if temperature is outside the valid range
    if (temp > tem[0] || temp < tem[9]) 
        PetscPrintf(PETSC_COMM_WORLD, "Warning: Temperature out of range in Sigma0 function.\n");

    // Find the interval in which `temp` falls
    for (ii = 0; ii < 10; ii++) {
        if (temp <= tem[ii]) 
            interv = ii;
    }

    // If temperature is above the highest defined value, use the first table value
    if (temp > tem[0]) 
        interv = -1;

    // Assign σ₀ based on the identified interval
    if (interv == -1) {
        *sigm0 = sig[0];  // Assign σ₀ at highest temperature
    }
    else if (interv == 9) {
        *sigm0 = sig[9];  // Assign σ₀ at lowest temperature
    }
    else {
        // Get the bounding temperature and σ₀ values
        t0 = fabs(tem[interv]); 
        t1 = fabs(tem[interv + 1]);
        s0 = sig[interv]; 
        s1 = sig[interv + 1];

        // Compute logarithmic interpolation for σ₀
        *sigm0 = pow(10.0, log10(s0) + 
                          (log10(s1) - log10(s0)) / (log10(t1) - log10(t0)) * 
                          (log10(fabs(temp)) - log10(t0)));
    }

    return;
}