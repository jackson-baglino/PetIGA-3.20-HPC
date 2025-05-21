#include "assembly.h"
#include "material_properties.h"

PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift, const PetscScalar *V,
                        PetscReal t, const PetscScalar *U,
                        PetscScalar *Re, void *ctx) {
    AppCtx *user = (AppCtx*) ctx;

    // Extract user-defined parameters
    PetscInt l, dim = user->dim;
    PetscReal eps = user->eps;
    PetscReal Etai = user->Etai;
    PetscReal Etam = user->Etam;
    PetscReal Etaa = user->Etaa;
    PetscReal ETA = Etaa * Etai + Etaa * Etam + Etam * Etai;
    PetscReal rho_ice = user->rho_ice;
    PetscReal lat_sub = user->lat_sub;
    PetscReal air_lim = user->air_lim;
    PetscReal xi_v = user->xi_v;
    PetscReal xi_T = user->xi_T;
    PetscReal rhoSE = rho_ice;

    // Compute the global index of the current Gauss point
    PetscInt indGP = pnt->index + pnt->count * pnt->parent->index;
    
    // Mobility and sublimation coefficient (temperature dependent or fixed)
    PetscReal mob, alph_sub;
    if (user->flag_Tdep == 1) {
        mob = user->mob[indGP];
        alph_sub = user->alph[indGP];
    } else {
        mob = user->mob_sub;
        alph_sub = user->alph_sub;
    }

    // Sediment phase fraction
    PetscReal met = user->Phi_sed[indGP];

    // Boundary condition handling (if applicable)
    if (pnt->atboundary) {
        return 0; // No residual calculation at boundaries (modify as needed)
    }

    // Solution values at Gauss points
    PetscScalar sol_t[3], sol[3];
    PetscScalar grad_sol[3][dim];
    IGAPointFormValue(pnt, V, &sol_t[0]);
    IGAPointFormValue(pnt, U, &sol[0]);
    IGAPointFormGrad(pnt, U, &grad_sol[0][0]);

    // Extract phase field variables and their gradients
    PetscScalar ice = sol[0], ice_t = sol_t[0];
    PetscScalar grad_ice[dim];
    for (l = 0; l < dim; l++) grad_ice[l] = grad_sol[0][l];
    
    // Air phase (complementary to ice and sediment)
    PetscScalar air = 1.0 - met - ice;
    PetscScalar air_t = -ice_t;
    
    // Temperature field
    PetscScalar tem = sol[1], tem_t = sol_t[1];
    PetscScalar grad_tem[dim];
    for (l = 0; l < dim; l++) grad_tem[l] = grad_sol[1][l];
    
    // Vapor density field
    PetscScalar rhov = sol[2], rhov_t = sol_t[2];
    PetscScalar grad_rhov[dim];
    for (l = 0; l < dim; l++) grad_rhov[l] = grad_sol[2][l];
    
    // Compute material properties based on ice and sediment fractions
    PetscReal thcond, cp, rho, difvap, rhoI_vs, fice, fmet, fair;
    ThermalCond(user, ice, met, &thcond, NULL);
    HeatCap(user, ice, met, &cp, NULL);
    Density(user, ice, met, &rho, NULL);
    VaporDiffus(user, tem, &difvap, NULL);
    RhoVS_I(user, tem, &rhoI_vs, NULL);
    Fice(user, ice, met, &fice, NULL);
    Fwat(user, ice, met, &fmet, NULL);
    Fair(user, ice, met, &fair, NULL);

    // Retrieve shape functions
    const PetscReal *N0, (*N1)[dim];
    IGAPointGetShapeFuns(pnt, 0, (const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt, 1, (const PetscReal**)&N1);
    
    // Residual contributions
    PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
    PetscInt a, nen = pnt->nen;
    for (a = 0; a < nen; a++) {
        PetscReal R_ice = 0.0, R_tem = 0.0, R_vap = 0.0;

        if (user->flag_tIC == 1) {
            // If initial condition flag is set, residuals are zeroed (modify if needed)
            R_ice = 0.0;
            R_tem = 0.0;
            R_vap = 0.0;
        } else {
            // Phase field residual
            R_ice = N0[a] * ice_t;
            for (l = 0; l < dim; l++) R_ice += 3.0 * mob * eps * (N1[a][l] * grad_ice[l]);
            R_ice += N0[a] * mob * 3.0 / eps / ETA * ((Etam + Etaa) * fice - Etaa * fmet - Etam * fair);
            R_ice -= N0[a] * alph_sub * ice * ice * air * air * (rhov - rhoI_vs) / rho_ice;

            // Energy equation residual (temperature)
            R_tem = rho * cp * N0[a] * tem_t;
            for (l = 0; l < dim; l++) R_tem += xi_T * thcond * (N1[a][l] * grad_tem[l]);
            R_tem += xi_T * rho * lat_sub * N0[a] * air_t;

            // Vapor transport residual
            R_vap = N0[a] * rhov * air_t;
            if (air > air_lim) {
                R_vap += N0[a] * air * rhov_t;
                for (l = 0; l < dim; l++) R_vap += xi_v * difvap * air * (N1[a][l] * grad_rhov[l]);
            } else {
                R_vap += N0[a] * air_lim * rhov_t;
                for (l = 0; l < dim; l++) R_vap += xi_v * difvap * air_lim * (N1[a][l] * grad_rhov[l]);
            }
            R_vap -= xi_v * N0[a] * rhoSE * air_t;
        }

        // Assign computed residuals
        R[a][0] = R_ice;
        R[a][1] = R_tem;
        R[a][2] = R_vap;
    }

    return 0;
}


PetscErrorCode Jacobian(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscInt l, dim=user->dim;
  PetscReal eps = user->eps;
  PetscReal Etai = user->Etai;
  PetscReal Etam = user->Etam;
  PetscReal Etaa = user->Etaa;
  PetscReal ETA = Etaa*Etai + Etaa*Etam + Etam*Etai; 
  PetscReal rho_ice = user->rho_ice;
  PetscReal lat_sub = user->lat_sub;
  PetscReal air_lim = user->air_lim;
  PetscReal xi_v = user->xi_v;
  PetscReal xi_T = user->xi_T;
  PetscReal rhoSE = rho_ice;
  PetscInt indGP = pnt->index + pnt->count *pnt->parent->index;
  PetscReal mob, alph_sub;
  if(user->flag_Tdep==1) {
    mob = user->mob[indGP];
    alph_sub = user->alph[indGP];
  } else {
    mob = user->mob_sub;
    alph_sub = user->alph_sub;
  }
  PetscReal met = user->Phi_sed[indGP];

 if(pnt->atboundary){

  } else {

    PetscScalar sol_t[3],sol[3];
    PetscScalar grad_sol[3][dim];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, ice_t, grad_ice[dim];
    ice          = sol[0]; 
    ice_t        = sol_t[0]; 
    for(l=0;l<dim;l++) grad_ice[l]  = grad_sol[0][l];

    PetscScalar air, air_t;
    air          = 1.0-met-ice;
    air_t        = -ice_t;

    PetscScalar tem, tem_t, grad_tem[dim];
    tem          = sol[1];
    tem_t        = sol_t[1];
    for(l=0;l<dim;l++) grad_tem[l]  = grad_sol[1][l];

    PetscScalar rhov, rhov_t, grad_rhov[dim];
    rhov           = sol[2];
    rhov_t         = sol_t[2];
    for(l=0;l<dim;l++) grad_rhov[l]   = grad_sol[2][l];

    PetscReal thcond,dthcond_ice;
    ThermalCond(user,ice,met,&thcond,&dthcond_ice);
    PetscReal cp,dcp_ice;
    HeatCap(user,ice,met,&cp,&dcp_ice);
    PetscReal rho,drho_ice;
    Density(user,ice,met,&rho,&drho_ice);
    PetscReal difvap,d_difvap;
    VaporDiffus(user,tem,&difvap,&d_difvap);
    PetscReal rhoI_vs,drhoI_vs;
    RhoVS_I(user,tem,&rhoI_vs,&drhoI_vs);
    PetscReal fice_ice;
    Fice(user,ice,met,NULL,&fice_ice);
    PetscReal fmet_ice;
    Fwat(user,ice,met,NULL,&fmet_ice);
    PetscReal fair_ice;
    Fair(user,ice,met,NULL,&fair_ice);

    const PetscReal *N0,(*N1)[dim]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
 
    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

        if(user->flag_tIC==1){

        } else {
        
        //ice
          J[a][0][b][0] += shift*N0[a]*N0[b];
          for(l=0;l<dim;l++) J[a][0][b][0] += 3.0*mob*eps*(N1[a][l]*N1[b][l]);

          J[a][0][b][0] += N0[a]*mob*3.0/eps/ETA*((Etam+Etaa)*fice_ice - Etaa*fmet_ice - Etam*fair_ice)*N0[b];
          J[a][0][b][0] -= N0[a]*alph_sub*2.0*ice*N0[b]*air*air*(rhov-rhoI_vs)/rho_ice;
          J[a][0][b][0] += N0[a]*alph_sub*ice*ice*2.0*air*N0[b]*(rhov-rhoI_vs)/rho_ice;
          J[a][0][b][1] += N0[a]*alph_sub*ice*ice*air*air*drhoI_vs*N0[b]/rho_ice;
          J[a][0][b][2] -= N0[a]*alph_sub*ice*ice*air*air*N0[b]/rho_ice;


        //temperature
          J[a][1][b][1] += shift*rho*cp*N0[a]*N0[b];
          J[a][1][b][0] += drho_ice*N0[b]*cp*N0[a]*tem_t;
          J[a][1][b][0] += rho*dcp_ice*N0[b]*N0[a]*tem_t;
          for(l=0;l<dim;l++) J[a][1][b][0] += xi_T*dthcond_ice*N0[b]*(N1[a][l]*grad_tem[l]);
          for(l=0;l<dim;l++) J[a][1][b][1] += xi_T*thcond*(N1[a][l]*N1[b][l]);
          J[a][1][b][0] += xi_T*drho_ice*N0[b]*lat_sub*N0[a]*air_t;
          J[a][1][b][0] -= xi_T*rho*lat_sub*N0[a]*shift*N0[b];

        //vapor density
          J[a][2][b][0] -= N0[a]*rhov*shift*N0[b];
          J[a][2][b][2] += N0[a]*N0[b]*air_t;
          if(air>air_lim){
            J[a][2][b][0] -= N0[a]*N0[b]*rhov_t;
            J[a][2][b][2] += N0[a]*air*shift*N0[b];
            for(l=0;l<dim;l++) J[a][2][b][0] -= xi_v*difvap*N0[b]*(N1[a][l]*grad_rhov[l]);
            for(l=0;l<dim;l++) J[a][2][b][1] += xi_v*d_difvap*N0[b]*air*(N1[a][l]*grad_rhov[l]);
            for(l=0;l<dim;l++) J[a][2][b][2] += xi_v*difvap*air*(N1[a][l]*N1[b][l]);        
          } else {
            J[a][2][b][2] += N0[a]*air_lim*shift*N0[b];
            for(l=0;l<dim;l++) J[a][2][b][1] += xi_v*d_difvap*N0[b]*air_lim*(N1[a][l]*grad_rhov[l] );
            for(l=0;l<dim;l++) J[a][2][b][2] += xi_v*difvap*air_lim*(N1[a][l]*N1[b][l]);
          }
          J[a][2][b][0] += xi_v*N0[a]*rhoSE*shift*N0[b];

        }
      }
    }

  }
  return 0;
}


PetscErrorCode Integration(IGAPoint pnt, const PetscScalar *U, PetscInt n, 
                            PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar sol[3];
  IGAPointFormValue(pnt,U,&sol[0]);

  PetscReal ice     = sol[0]; 
  PetscReal met     = user->Phi_sed[pnt->index + pnt->count *pnt->parent->index];
  PetscReal air     = 1.0-met-ice;
  PetscReal temp    = sol[1];
  PetscReal rhov    = sol[2];
  PetscReal triple  = SQ(air)*SQ(met)*SQ(ice);

  S[0]  = ice;
  S[1]  = triple;
  S[2]  = air;
  S[3]  = temp;
  S[4]  = rhov*air;
  S[5]  = air*air*ice*ice;

  PetscFunctionReturn(0);
}