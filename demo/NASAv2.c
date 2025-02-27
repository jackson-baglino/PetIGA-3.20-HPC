#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"
#include <math.h>
#include <mpi.h>

#define SQ(x) ((x)*(x))
#define CU(x) ((x)*(x)*(x))


typedef struct {
  IGA       iga;
  
  PetscReal eps;
  PetscReal mob_sub,mav,Etai,Etam,Etaa,alph_sub,Lambd, beta_sub0,d0_sub0;
  PetscReal thcond_ice,thcond_met,thcond_air,cp_ice,cp_met,cp_air,rho_ice,\
            rho_met,rho_air,dif_vap,lat_sub, diff_sub;
  PetscReal air_lim, xi_v, xi_T;
  PetscReal T_melt, temp0, hum0, grad_temp0[3];
  PetscReal Lx, Ly, Lz, Nx, Ny, Nz;
  PetscReal RCice, RCsed, RCice_dev, RCsed_dev;
  PetscReal cent[3][200],radius[200], centsed[3][200],radiussed[200];
  PetscReal norm0_0,norm0_1,norm0_2;
  PetscInt  flag_it0, flag_tIC, outp, nsteps_IC, flag_xiT, flag_Tdep;
  PetscInt  p, C, dim, periodic;
  PetscReal t_out, t_interv, t_IC;
  PetscInt  NCice, NCsed, n_act, n_actsed;
  PetscReal *Phi_sed, *alph, *mob;

  PetscInt  readFlag;

} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm, 
   PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
  /* ***************************************************************************
  * This function is called to check the convergence of the SNES solver at each 
  * iteration.
  * It calculates the norms of the residual vector and the solution update 
  * vector, and prints them along with the iteration number and the function 
  * norm.
  * It also checks if the convergence criteria are met and updates the 
  * convergence reason accordingly. 
  *************************************************************************** */

  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)cctx;

  Vec Res,Sol,Sol_upd;
  PetscScalar n2dof0,n2dof1,n2dof2;
  PetscScalar solv,solupdv;

  ierr = SNESGetFunction(snes,&Res,0,0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,0,NORM_2,&n2dof0);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,1,NORM_2,&n2dof1);CHKERRQ(ierr);
  ierr = VecStrideNorm(Res,2,NORM_2,&n2dof2);CHKERRQ(ierr);
  if(user->flag_tIC == 1){
   ierr = SNESGetSolution(snes,&Sol);CHKERRQ(ierr);
   ierr = VecStrideNorm(Sol,2,NORM_2,&solv);CHKERRQ(ierr);
   ierr = SNESGetSolutionUpdate(snes,&Sol_upd);CHKERRQ(ierr);
   ierr = VecStrideNorm(Sol_upd,2,NORM_2,&solupdv);CHKERRQ(ierr);
  }

  if(it_number==0) {
   user->norm0_0 = n2dof0;
   user->norm0_1 = n2dof1;
   user->norm0_2 = n2dof2;
   if(user->flag_tIC == 1) solupdv = solv;  
  }

  PetscPrintf(PETSC_COMM_WORLD,"    IT_NUMBER: %d ", it_number);
  PetscPrintf(PETSC_COMM_WORLD,"    fnorm: %.4e \n", fnorm);
  PetscPrintf(PETSC_COMM_WORLD,"    n0: %.2e r %.1e ", n2dof0, n2dof0/user->norm0_0);
  PetscPrintf(PETSC_COMM_WORLD,"  n1: %.2e r %.1e ", n2dof1, n2dof1/user->norm0_1);
  if(user->flag_tIC == 1) PetscPrintf(PETSC_COMM_WORLD,"  x2: %.2e s %.1e \n", solv, solupdv/solv);
  else PetscPrintf(PETSC_COMM_WORLD,"  n2: %.2e r %.1e \n", n2dof2, n2dof2/user->norm0_2);

  PetscScalar atol, rtol, stol;
  PetscInt maxit, maxf;
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);
  if(snes->prev_dt_red ==1) rtol *= 10.0;

  if(user->flag_it0 == 1){
   atol = 1.0e-12;
   if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
    && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
    && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) ) {

    *reason = SNES_CONVERGED_FNORM_RELATIVE;
   }    
  } else {
   atol = 1.0e-20;
   if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
    && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
    && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) ) {

    *reason = SNES_CONVERGED_FNORM_RELATIVE;
   }     
  }

  PetscFunctionReturn(0);
}


void ThermalCond(AppCtx *user, PetscScalar ice, PetscScalar met, 
                  PetscScalar *cond, PetscScalar *dcond_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-met-ice;
  if(met<0.0) {met=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal cond_ice = user->thcond_ice;
  PetscReal cond_met = user->thcond_met;
  PetscReal cond_air = user->thcond_air;
  if(cond)      (*cond)  = ice*cond_ice + met*cond_met + air*cond_air;
  if(dcond_ice)    (*dcond_ice) = cond_ice*dice-cond_air*dair;

  return;
}

void HeatCap(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *cp, 
              PetscScalar *dcp_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-met-ice;
  if(met<0.0) {met=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal cp_ice = user->cp_ice;
  PetscReal cp_met = user->cp_met;
  PetscReal cp_air = user->cp_air;
  if(cp)     (*cp)  = ice*cp_ice + met*cp_met + air*cp_air;
  if(dcp_ice)    (*dcp_ice) = cp_ice*dice-cp_air*dair;

  return;
}

void Density(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *rho, 
              PetscScalar *drho_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-met-ice;
  if(met<0.0) {met=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal rho_ice = user->rho_ice;
  PetscReal rho_met = user->rho_met;
  PetscReal rho_air = user->rho_air;
  if(rho)     (*rho)  = ice*rho_ice + met*rho_met + air*rho_air;
  if(drho_ice)    (*drho_ice) = rho_ice*dice-rho_air*dair;

  
  return;
}

void VaporDiffus(AppCtx *user, PetscScalar tem, PetscScalar *difvap, 
                  PetscScalar *d_difvap)
{

  PetscReal dif_vap = user->dif_vap;
  PetscReal Kratio = (tem+273.15)/273.15;
  PetscReal aa = 1.81;
  if(difvap)     (*difvap)  = dif_vap*pow(Kratio,aa);
  if(d_difvap)    (*d_difvap) = dif_vap*aa*pow(Kratio,aa-1.0)/273.15;
  
  return;
}

void RhoVS_I(AppCtx *user, PetscScalar tem, PetscScalar *rho_vs, 
              PetscScalar *d_rhovs)
{

  PetscReal rho_air = user->rho_air;
  PetscReal K0,K1,K2,K3,K4,K5;
  K0 = -0.5865*1.0e4;   K1 = 0.2224*1.0e2;    K2 = 0.1375*1.0e-1;
  K3 = -0.3403*1.0e-4;  K4 = 0.2697*1.0e-7;   K5 = 0.6918;
  PetscReal Patm = 101325.0;
  PetscReal bb = 0.62;
  PetscReal temK = tem+273.15;
  PetscReal Pvs = exp(K0*pow(temK,-1.0)+K1+K2*pow(temK,1.0)+K3*pow(temK,2.0)+K4*pow(temK,3.0)+K5*log(temK));
  PetscReal Pvs_T = Pvs*(-K0*pow(temK,-2.0)+K2+2.0*K3*pow(temK,1.0)+3.0*K4*pow(temK,2.0)+K5/temK);

  if(rho_vs)     (*rho_vs)  = rho_air*bb*Pvs/(Patm-Pvs);
  if(d_rhovs)  (*d_rhovs) = rho_air*bb*(Pvs_T*(Patm-Pvs)+Pvs*Pvs_T)/(Patm-Pvs)/(Patm-Pvs);
  
  return;
}

void Fice(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fice, 
          PetscScalar *dfice_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etai  = user->Etai;
  PetscReal air = 1.0-met-ice;
  if(fice)     (*fice)  = etai*ice*(1.0-ice)*(1.0-2.0*ice) + 2.0*Lambd*ice*met*met*air*air;
  if(dfice_ice)    (*dfice_ice) = etai*(1.0-6.0*ice+6.0*ice*ice) + \
                    2.0*Lambd*met*met*air*air - 2.0*Lambd*ice*met*met*2.0*air;
  
  return;
}

void Fwat(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fwat, 
          PetscScalar *dfwat_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etam  = user->Etam;
  PetscReal air = 1.0-met-ice;
  if(fwat)     (*fwat)  = etam*(met)*(1.0-met)*(1.0-2.0*met) + 2.0*Lambd*ice*ice*met*air*air;
  if(dfwat_ice)    {
    (*dfwat_ice)  = 2.0*Lambd*2.0*ice*met*air*air - 2.0*Lambd*ice*ice*met*2.0*air;

  }

  
  return;
}

void Fair(AppCtx *user, PetscScalar ice, PetscScalar met, PetscScalar *fair, 
          PetscScalar *dfair_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etaa  = user->Etaa;
  PetscReal air = 1.0-met-ice;
  if(fair)     (*fair)  = etaa*air*(1.0-air)*(1.0-2.0*air) + 2.0*Lambd*ice*ice*met*met*air;
  if(dfair_ice)    {
    (*dfair_ice)  = -etaa*(1.0-air)*(1.0-2.0*air) + etaa*air*(1.0-2.0*air) + etaa*air*(1.0-air)*2.0;
    (*dfair_ice) += 2.0*Lambd*2.0*ice*met*met*air - 2.0*Lambd*ice*ice*met*met;
  }
  
  return;
}

void Sigma0(PetscScalar temp, PetscScalar *sigm0)
{
  PetscReal sig[10], tem[10];
  sig[0] = 3.0e-3;  sig[1] = 4.1e-3;  sig[2] = 5.5e-3; sig[3] = 8.0e-3; sig[4] = 4.0e-3;
  tem[0] = -0.0001;     tem[1] = -2.0;    tem[2] = -4.0;   tem[3] = -6.0;   tem[4] = -7.0;
  sig[5] = 6.0e-3;  sig[6] = 3.5e-2;  sig[7] = 7.0e-2; sig[8] = 1.1e-1; sig[9] = 0.75; 
  tem[5] = -10.0;   tem[6] = -20.0;   tem[7] = -30.0;  tem[8] = -40.0;  tem[9] = -100.0;

  PetscInt ii, interv=0;
  PetscReal t0, t1, s0, s1;
  if(temp>tem[0] || temp<tem[9]) PetscPrintf(PETSC_COMM_WORLD,"Temperature out of range: Sigma_0 \n");

  for(ii=0;ii<10;ii++){
    if(temp<=tem[ii]) interv = ii;
  }
  if(temp>tem[0]) interv = -1;

  if(interv ==-1) (*sigm0) = sig[0];
  else if (interv==9) (*sigm0) = sig[9];
  else{
    t0=fabs(tem[interv]), t1=fabs(tem[interv+1]);
    s0=sig[interv], s1=sig[interv+1];

    (*sigm0) = pow(10.0, log10(s0) + (log10(s1)-log10(s0))/(log10(t1)-log10(t0))*(log10(fabs(temp))-log10(t0)) );
  }

  

  return;
}

PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscInt l, dim = user->dim;
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
    
  } else  {
    
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

    PetscReal thcond,cp,rho,difvap,rhoI_vs,fice,fmet,fair;
    ThermalCond(user,ice,met,&thcond,NULL);
    HeatCap(user,ice,met,&cp,NULL);
    Density(user,ice,met,&rho,NULL);
    VaporDiffus(user,tem,&difvap,NULL);
    RhoVS_I(user,tem,&rhoI_vs,NULL);
    Fice(user,ice,met,&fice,NULL);
    Fwat(user,ice,met,&fmet,NULL);
    Fair(user,ice,met,&fair,NULL);

    const PetscReal *N0,(*N1)[dim]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
    
    PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      
      PetscReal R_ice,R_tem,R_vap;

      if(user->flag_tIC==1){

        R_ice  = 0.0;//N0[a]*ice_t;

        R_tem  =  0.0;//rho*cp*N0[a]*tem_t;
        R_tem +=  0.0;//thcond*(N1[a][0]*grad_tem[0] + N1[a][1]*grad_tem[1]);

        R_vap  =  0.0;//N0[a]*rhov;
        R_vap -=  0.0;//N0[a]*rhoI_vs;

      } else {

        R_ice  = N0[a]*ice_t; 
        for(l=0;l<dim;l++) R_ice += 3.0*mob*eps*(N1[a][l]*grad_ice[l]);
        R_ice += N0[a]*mob*3.0/eps/ETA*((Etam+Etaa)*fice - Etaa*fmet - Etam*fair);
        R_ice -= N0[a]*alph_sub*ice*ice*air*air*(rhov-rhoI_vs)/rho_ice;        

        R_tem  = rho*cp*N0[a]*tem_t;
        for(l=0;l<dim;l++) R_tem += xi_T*thcond*(N1[a][l]*grad_tem[l]);
        R_tem += xi_T*rho*lat_sub*N0[a]*air_t;

        R_vap  = N0[a]*rhov*air_t;
        if(air>air_lim){
          R_vap += N0[a]*air*rhov_t;
          for(l=0;l<dim;l++) R_vap += xi_v*difvap*air*(N1[a][l]*grad_rhov[l] );
        } else {
          R_vap += N0[a]*air_lim*rhov_t;
          for(l=0;l<dim;l++) R_vap += xi_v*difvap*air_lim*(N1[a][l]*grad_rhov[l] );
        }
        R_vap -=  xi_v*N0[a]*rhoSE*air_t;        

      }

      R[a][0] = R_ice;
      R[a][1] = R_tem;
      R[a][2] = R_vap;
    }
    
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

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  //-------- compute beta_sub
  Vec localU;
  const PetscScalar *arrayU;
  IGAElement element;
  IGAPoint point;
  PetscScalar *UU;
  PetscScalar rhovs, arg_kin, sigm0, sigm_surf, v_kin, alp;
  PetscInt indd=0;
  PetscReal a1=5.0, a2=0.1581, bet_max=0.0, bet_min=1.0e30;
  PetscReal bet0, d0, rho_rhovs, d0_sub,  beta_sub, lambda_sub, tau_sub;

  if(user->flag_Tdep==1){
    ierr = IGAGetLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
    ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
    while (IGANextElement(user->iga,element)) {
        ierr = IGAElementGetValues(element,arrayU,&UU);CHKERRQ(ierr);
        ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
        while (IGAElementNextPoint(element,point)) {
            PetscScalar solS[3];
            ierr = IGAPointFormValue(point,UU,&solS[0]);CHKERRQ(ierr);
            RhoVS_I(user,solS[1],&rhovs,NULL);
            sigm_surf=fabs(solS[2]-rhovs)/rhovs;
            rho_rhovs = user->rho_ice/rhovs;

            arg_kin = 1.38e-23*(solS[1]+273.15)/(2.0*3.14159*3.0e-26);
            v_kin = pow(arg_kin,0.5)/rho_rhovs;

            Sigma0(solS[1],&sigm0);
            if(sigm0<=0.0) PetscPrintf(PETSC_COMM_SELF,"ERROR: Negative Sigma0 value %e\n",sigm0);
            if(sigm_surf < sigm0/69.0775) alp = 1.0e-30;
            else alp = exp(-sigm0/sigm_surf);

            if(alp*v_kin<1.0e-30) bet0 = 1.0e30;
            else bet0 = 1.0/(alp*v_kin);
            d0 = 2.548e-7/(solS[1]+273.15);

            if(bet0>bet_max) bet_max = bet0;
            if(bet0<bet_min) bet_min = bet0;
            d0_sub   = d0/rho_rhovs;  
            beta_sub = bet0/rho_rhovs;
            lambda_sub = a1*user->eps/d0_sub;
            tau_sub    = user->eps*lambda_sub*(beta_sub/a1 + a2*user->eps/user->diff_sub + a2*user->eps/user->dif_vap);

            user->mob[indd] = user->eps/3.0/tau_sub; 
            user->alph[indd] = lambda_sub/tau_sub; 

            indd ++;
        }
        ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
    ierr = IGARestoreLocalVecArray(user->iga,U,&localU,&arrayU);CHKERRQ(ierr);
    PetscReal B_min, B_max;
    ierr = MPI_Allreduce(&bet_max,&B_max,1,MPI_DOUBLE,MPI_MAX,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Allreduce(&bet_min,&B_min,1,MPI_DOUBLE,MPI_MIN,PETSC_COMM_WORLD);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD," b_min %.2e b_max %.2e\n",B_min,B_max);

    // After computing beta_sub, we set the flag to 0...
    user->flag_Tdep = 0;
  }


  //-------- domain integrals
  PetscScalar stats[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,6,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal tot_ice     = PetscRealPart(stats[0]);
  PetscReal tot_trip    = PetscRealPart(stats[1]);
  PetscReal tot_air     = PetscRealPart(stats[2]);
  PetscReal tot_temp    = PetscRealPart(stats[3]);
  PetscReal tot_rhov    = PetscRealPart(stats[4]);
  PetscReal sub_interf  = PetscRealPart(stats[5]); 
 
  //------------- 
  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  if(step==1) user->flag_it0 = 0;

  //------------- initial condition
  if(user->flag_tIC==1) if(step==user->nsteps_IC) {
    user->flag_tIC = 0; user->t_IC = t; //user->flag_rtol = 1;
    PetscPrintf(PETSC_COMM_WORLD,"INITIAL_CONDITION!!! \n");
  }

  //------printf information
  if(step%10==0) {
    PetscPrintf(PETSC_COMM_WORLD,"\nTIME               TIME_STEP     TOT_ICE      TOT_AIR       TEMP      TOT_RHOV     I-A interf   Tripl_junct \n");
    PetscPrintf(PETSC_COMM_WORLD,"\n(%.0f) %.3e    %.3e   %.3e   %.3e   %.3e   %.3e   %.3e   %.3e \n\n",
                t,t,dt,tot_ice,tot_air,tot_temp,tot_rhov,sub_interf,tot_trip);
  }

  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }
  
  print = 1;

  if(print==1) {
    char filedata[256];
    const char *env = "folder"; char *dir; dir = getenv(env);

    sprintf(filedata,"%s/SSA_evo.dat",dir);
    PetscViewer       view;
    PetscViewerCreate(PETSC_COMM_WORLD,&view);
    PetscViewerSetType(view,PETSCVIEWERASCII);

    if (step==0){
      PetscViewerFileSetMode(view,FILE_MODE_WRITE);
    } else {
      PetscViewerFileSetMode(view,FILE_MODE_APPEND);
    }

    PetscViewerFileSetName(view,filedata);
    PetscViewerASCIIPrintf(view,"%e %e %e %d\n",sub_interf/user->eps, tot_ice, t, step);

    PetscViewerDestroy(&view);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode OutputMonitor(TS ts, PetscInt step, PetscReal t, Vec U, 
                              void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;  

  // Check if it's the first step
  if (step == 0) {
    const char *env = "folder";
    char *dir;
    dir = getenv(env);

    char fileiga[256];
    sprintf(fileiga, "%s/igasol.dat", dir);

    ierr = IGAWrite(user->iga, fileiga);CHKERRQ(ierr);
  }

  // Check if it's time to print output
  PetscInt print = 0;
  if (user->outp > 0) {   // Print output every user->outp steps
    if (step % user->outp == 0) print = 1;
  } 
  else {                  // Print output every user->t_interv seconds
    if (t >= user->t_out) print = 1;
  }

  // If it's time to print output, do the following
  if (print == 1) {
    PetscPrintf(PETSC_COMM_WORLD, "OUTPUT print!\n");
    user->t_out += user->t_interv;

    // Get the directory path from the environment variable
    const char *env = "folder";
    char *dir;
    dir = getenv(env);

    // Create the filename for the output file
    char filename[256];
    sprintf(filename, "%s/sol_%05d.dat", dir, step);

    // Write the vector U to the output file
    ierr = IGAWriteVec(user->iga, U, filename);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


PetscErrorCode InitialSedGrains(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"--------------------- SEDIMENTS --------------------------\n");

  if(user->NCsed==0) {
    user->n_actsed= 0;
    PetscPrintf(PETSC_COMM_WORLD,"No sed grains\n\n");
    PetscFunctionReturn(0);
  }

  PetscReal rad = user->RCsed;
  PetscReal rad_dev = user->RCsed_dev;
  PetscInt  numb_clust = user->NCsed,ii,jj,tot=10000;
  PetscInt  l, n_act=0, flag, dim=user->dim, seed=13;

//----- cluster info
  PetscReal centX[3][numb_clust], radius[numb_clust];
  PetscRandom randcX,randcY,randcZ,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcY);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcY,0.0,user->Ly);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+2+8*iga->elem_start[0]+11*iga->elem_start[1]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcY,seed+numb_clust*34+5*iga->elem_start[1]+4*iga->elem_start[0]);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*numb_clust+5*iga->proc_ranks[1]+8*iga->elem_start[0]+2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcY);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);
  if(dim==3){
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcZ);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(randcZ,0.0,user->Lz);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(randcZ,seed*3+iga->elem_width[1]+6);CHKERRQ(ierr);
    ierr = PetscRandomSeed(randcZ);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randcZ);CHKERRQ(ierr);
  }
  PetscReal xc[3], rc=0.0, dist=0.0; 
  xc[0]=xc[1]=xc[2]=0.0;

  for(ii=0;ii<tot*numb_clust;ii++){
    ierr=PetscRandomGetValue(randcX,&xc[0]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcY,&xc[1]);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    if(dim==3) {ierr=PetscRandomGetValue(randcZ,&xc[2]);CHKERRQ(ierr);}
    //PetscPrintf(PETSC_COMM_WORLD,"  %.4f %.4f %.4f \n",xc,yc,rc);
    flag=1;
    
    for(jj=0;jj<n_act;jj++){
      dist = 0.0;
      for(l=0;l<dim;l++) dist += SQ(xc[l]-centX[l][jj]);
      dist = sqrt(dist);
      if(dist< (rc+radius[jj]) ) flag = 0;
    }

    if(flag==1){
      if(dim==3) PetscPrintf(PETSC_COMM_WORLD," new sed grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n",n_act,xc[0],xc[1],xc[2],rc);
      else PetscPrintf(PETSC_COMM_WORLD," new sed grain %d!!  x %.2e  y %.2e  r %.2e \n",n_act,xc[0],xc[1],rc);
      for(l=0;l<dim;l++) centX[l][n_act] = xc[l];
      radius[n_act] = rc;
      n_act++;
    }
    if(n_act==numb_clust) {
      PetscPrintf(PETSC_COMM_WORLD," %d sed grains in %d iterations \n\n", n_act, ii);
      ii=tot*numb_clust;
    }
  }
  if(n_act != numb_clust) PetscPrintf(PETSC_COMM_WORLD," %d sed grains in maximum number of iterations allowed (%d)\n \n", n_act, ii);

  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcY);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);
  if(dim==3) {ierr = PetscRandomDestroy(&randcZ);CHKERRQ(ierr);}

  //PetscPrintf(PETSC_COMM_SELF,"before  %.4f %.4f %.4f \n",centX[0],centY[0],radius[0]);

  //----- communication
  for(l=0;l<dim;l++){ierr = MPI_Bcast(centX[l],numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);}
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);


  user->n_actsed = n_act;
  for(jj=0;jj<n_act;jj++){
    for(l=0;l<dim;l++) user->centsed[l][jj] = centX[l][jj];
    user->radiussed[jj] = radius[jj];
    //PetscPrintf(PETSC_COMM_SELF,"Sed grains: points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
  }

  //-------- define the Phi_sed values

  IGAElement element;
  IGAPoint point;
  PetscReal sed=0.0;
  PetscInt  aa,ind=0;

  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
        sed=0.0;
        for(aa=0;aa<user->n_actsed;aa++){
          dist=0.0;
          for(l=0;l<dim;l++) dist += SQ(point->mapX[0][l]-user->centsed[l][aa]);
          dist = sqrt(dist);
          sed += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[aa]));
        }
        if(sed>1.0) sed=1.0;
        //PetscPrintf(PETSC_COMM_SELF," sed %.3f \n",sed);
        user->Phi_sed[ind] = sed;
        ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF," ind  %d \n",ind);

  PetscFunctionReturn(0); 
}


PetscErrorCode InitialSedGrainsGravity(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"--------------------- SEDIMENTS --------------------------\n");

  if(user->NCsed==0) {
    user->n_actsed= 0;
    PetscPrintf(PETSC_COMM_WORLD,"No sed grains\n\n");
    PetscFunctionReturn(0);
  }

  PetscReal rad = user->RCsed;
  PetscReal rad_dev = user->RCsed_dev;
  PetscInt  tot_part = user->NCsed, ii, jj, kk, ll;
  PetscInt  n_act=0, flag, flag_nextpart, seed=11, fail=0, repeat=0;

  //----- x-coordinate and radius of new particles
  PetscReal centX[tot_part],centY[tot_part], radius[tot_part];
  PetscRandom randcX,randcR;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcX,seed+21+9*iga->elem_start[0]-11*iga->elem_start[1]+fail);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(randcR,seed*tot_part+7*iga->proc_ranks[1]+6*iga->elem_start[0]+2);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);

  PetscReal xc=0.0,yc=0.0,rc=0.0, dist=0.0, xcaux, xmax,xmin;
  PetscInt lim=20, upp, n_cand=10, cand, cand_pick, init_cand, nst;
  PetscReal x1,x2, y1,y2, r1,r2, AA,BB,CC,EE,a_,b_,c_, x_can,y_can;
  PetscReal xc_cand[n_cand], yc_cand[n_cand];
  PetscReal xxst[lim],yyst[lim],rrst[lim], auxx,auxy,auxr;

  for(ii=0;ii<tot_part+1;ii++){

    flag_nextpart=0;
    if(repeat==1) {
      ii--;
      repeat=0;
    }
    if(ii==tot_part) break;

    ierr=PetscRandomGetValue(randcX,&xc);CHKERRQ(ierr);
    ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\nParticle #%d random values xc %.3e rc %.3e \n",ii,xc,rc);
    if(xc<rc) {
      PetscPrintf(PETSC_COMM_WORLD," Outside left boundary, correct!\n");
      xc = rc;
    }
    if(xc>user->Lx-rc) {
      PetscPrintf(PETSC_COMM_WORLD," Outside right boundary, correct!\n");
      xc = user->Lx-rc;
    }


    //particles in stripe xc+-3R
    PetscReal xst[n_act+1],yst[n_act+1],rst[n_act+1];
    nst=0;
    for(jj=0;jj<n_act;jj++){
      if( centX[jj]> xc-3.0*rad*(1.0+rad_dev) && centX[jj]< xc+3.0*rad*(1.0+rad_dev) ) {
        xst[nst] = centX[jj];
        yst[nst] = centY[jj];
        rst[nst] = radius[jj];
        nst++;
      }
    }
    PetscPrintf(PETSC_COMM_WORLD," There are %d existing particles in stripe around xc %.2e \n",nst,xc);
    //order according to y-coordinate and pick the upper 20?
    //PetscInt lim=20, upp;
    //PetscReal xxst[upp+1],yyst[upp+1],rrst[upp+1];
    for(jj=0;jj<nst-1;jj++){
      for(kk=jj+1;kk<nst;kk++){
        if(yst[jj]<yst[kk]){
          auxx=xst[kk];     auxy=yst[kk];     auxr=rst[kk];
          xst[kk]=xst[jj];  yst[kk]=yst[jj];  rst[kk]=rst[jj];
          xst[jj]=auxx;     yst[jj]=auxy;     rst[jj]=auxr;
        }
      }
    }
    if(nst<lim) upp = nst;
    else upp=lim;
    //pick the upper #
    for(jj=0;jj<upp;jj++){
      xxst[jj] = xst[jj];
      yyst[jj] = yst[jj];
      rrst[jj] = rst[jj];
    }
    // if upp<20, we should consider the bottom boundary!!
    // it is possible that upp=0 (no particles)

    //bottom wall
    if(upp<lim){
      PetscPrintf(PETSC_COMM_WORLD,"  BOTTOM: Low number of particles, thus, we check the bottom \n");
      cand=0;
      yc=rc;
      flag=1;
    //try particle placed on bottom, no readjustment
      for(jj=0;jj<n_act;jj++){ //
        dist = sqrt(SQ(xc-centX[jj])+SQ(yc-centY[jj]));
        if(1.01*dist < rc+radius[jj]) flag = 0;
      }
      if(flag==1){
        PetscPrintf(PETSC_COMM_WORLD,"   Particle on the bottom, no adjustment: ");
        PetscPrintf(PETSC_COMM_WORLD,"    new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc,yc,rc);
        centX[n_act] = xc;
        centY[n_act] = yc;
        radius[n_act] = rc;
        n_act++;
        flag_nextpart = 1;
      } 
      // try with readjustment //for each upp we find a potential candidate
      if(flag_nextpart==0){
        PetscPrintf(PETSC_COMM_WORLD,"   Bottom contact needs adjustment: \n");
        for(jj=0;jj<upp;jj++){ 
          flag=1;
          if(2.0*rc + rrst[jj] > yyst[jj]) { //contact: new particle in contact with floor and particle jj
            //PetscPrintf(PETSC_COMM_WORLD,"    Computing position with particle %d in the stripe \n",jj);
            //two solutions
            x1 = xxst[jj] - sqrt(SQ(rc+rrst[jj])-SQ(rc-yyst[jj]));
            x2 = xxst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-yyst[jj]));
            //pick solution closer to xc
            if(fabs(x1-xc)<fabs(x2-xc)) xcaux = x1;
            else xcaux = x2;
            //PetscPrintf(PETSC_COMM_WORLD,"     Left xc %.3e, right xc %.3e, our pick is %.3e \n",x1,x2,xcaux);
            //look for overlap
            for(kk=0;kk<n_act;kk++){
              dist = sqrt(SQ(xcaux-centX[kk]) + SQ(yc-centY[kk]));
              if(1.01*dist < rc+radius[kk]) flag = 0;
            }
            if(xcaux<rc || xcaux > user->Lx-rc) flag = 0;
            if(flag==1){ // add candidate to list
              //PetscPrintf(PETSC_COMM_WORLD,"      No overlap with previous particles \n");
              xc_cand[cand] = xcaux;
              yc_cand[cand] = yc;
              cand++;
            } //else PetscPrintf(PETSC_COMM_WORLD,"      Overlap with previous particles \n");
            if(cand == n_cand) jj=upp; //number of candidates is completed!
          } //else PetscPrintf(PETSC_COMM_WORLD,"   No possible contact with particle %d in the stripe \n",jj);
        }
      //choose a candidate (the closer one)
        if(cand>0){
          //PetscPrintf(PETSC_COMM_WORLD,"    Choosing candidates (closer to original xc) amongst the %d options \n",cand);
          dist=1.0e6;
          for(jj=0;jj<cand;jj++){
            if(fabs(xc_cand[jj]-xc)<dist ){
              dist= fabs(xc_cand[jj]-xc);
              cand_pick = jj;
            }
          }
          PetscPrintf(PETSC_COMM_WORLD,"      new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc_cand[cand_pick],yc_cand[cand_pick],rc);
          centX[n_act] = xc_cand[cand_pick];
          centY[n_act] = yc_cand[cand_pick];
          radius[n_act] = rc;
          n_act++;
          flag_nextpart = 1;
        }
      }
    }

    cand=0;

    //lateral walls (left wall)
    if(xc<3.0*rad*(1.0+rad_dev) && flag_nextpart==0 ) {
      PetscPrintf(PETSC_COMM_WORLD," LEFT WALL! \n");
      init_cand = cand;
      for(jj=0;jj<upp;jj++){ 
        flag=1;
        if(2.0*rc+rrst[jj] > xxst[jj] && rc<xxst[jj]) { //contact: new particle in contact with wall and particle jj
          //PetscPrintf(PETSC_COMM_WORLD,"  Contact with particle %d ",jj);
          //two solutions
          //PetscReal y1;
          y1 = yyst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-xxst[jj]));
          //PetscPrintf(PETSC_COMM_WORLD,"  at point y=%.3e \n",y1);
          //y2 = yyst[jj] - sqrt(SQ(rc+rrst[jj])-SQ(rc-xxst[jj]));
        //look for overlap
          for(kk=0;kk<n_act;kk++){
            dist = sqrt(SQ(xc-centX[kk])+SQ(y1-centY[kk]));
            if(1.01*dist < rc+radius[kk]) flag = 0;
          }
          if(y1+rc > user->Ly) flag=0;
          if(flag==1){ // add candidate to list
            //PetscPrintf(PETSC_COMM_WORLD,"   No overlap, new left_B candidate  %d \n",cand-init_cand);
            xc_cand[cand] = rc;
            yc_cand[cand] = y1;
            cand++;
          } //else PetscPrintf(PETSC_COMM_WORLD,"   Overlap with previous particles\n");
          if(cand == n_cand) jj=upp; //number of candidates is completed!
        }
      }
      //choose a candidate (the higher one)
      if(cand>init_cand){
        PetscPrintf(PETSC_COMM_WORLD,"   Pick amongst %d left_B candidates ",cand-init_cand);
        dist=0.0;
        for(jj=init_cand;jj<cand;jj++){
          if(yc_cand[jj]>dist ){
            dist = yc_cand[jj];
            cand_pick = jj;
          }
        }
        //store this candidate in the first available element
        PetscPrintf(PETSC_COMM_WORLD,"  the number %d \n",cand_pick);
        xc_cand[init_cand]=xc_cand[cand_pick];
        yc_cand[init_cand]=yc_cand[cand_pick];
        cand = init_cand+1;
      }

    }
    //right wall
    if(xc>user->Lx-3.0*rad*(1.0+rad_dev)  && flag_nextpart==0 ) {
      PetscPrintf(PETSC_COMM_WORLD," RIGHT WALL! \n");
      init_cand = cand;
      for(jj=0;jj<upp;jj++){ 
        flag=1;
        if(2.0*rc+rrst[jj] > user->Lx-xxst[jj] && rc<user->Lx-xxst[jj]) { //contact: new particle in contact with wall and particle jj
          //PetscPrintf(PETSC_COMM_WORLD,"  Contact with particle %d ",jj);
      //two solutions
          //PetscReal y1;
          y1 = yyst[jj] + sqrt(SQ(rc+rrst[jj])-SQ(rc-(user->Lx-xxst[jj])));
          //PetscPrintf(PETSC_COMM_WORLD,"  at point y=%.3e \n",y1);
      //look for overlap
          for(kk=0;kk<n_act;kk++){
            dist = sqrt(SQ(xc-centX[kk])+SQ(y1-centY[kk]));
            if(1.01*dist < rc+radius[kk]) flag = 0;
          }
          if(y1+rc > user->Ly) flag=0;
          if(flag==1){ // add candidate to list
            //PetscPrintf(PETSC_COMM_WORLD,"   No overlap, new right_B candidate  %d \n",cand-init_cand);
            xc_cand[cand] = user->Lx-rc;
            yc_cand[cand] = y1;
            cand++;
          } //else PetscPrintf(PETSC_COMM_WORLD,"   Overlap with previous particles\n");
          if(cand == n_cand) jj=upp; //number of candidates is completed!
        }
      }
      //choose a candidate (the higher one)
      if(cand>init_cand){
        PetscPrintf(PETSC_COMM_WORLD,"   Pick amongst %d right_B candidates ",cand-init_cand);
        dist=0.0;
        for(jj=init_cand;jj<cand;jj++){
          if(yc_cand[jj]>dist ){
            dist = yc_cand[jj];
            cand_pick = jj;
          }
        }
        //store this candidate in the first available element
        PetscPrintf(PETSC_COMM_WORLD,"  the number %d \n",cand_pick);
        xc_cand[init_cand]=xc_cand[cand_pick];
        yc_cand[init_cand]=yc_cand[cand_pick];
        cand = init_cand+1;
      }
    }
    //loop amongs the particles on the top
    if(flag_nextpart==0 && cand<n_cand) {
      PetscPrintf(PETSC_COMM_WORLD," INNER PARTICLES: Check particles far from boundaries! \n");
      for(jj=0;jj<upp-1;jj++){
        init_cand = cand;
        for(kk=jj+1;kk<upp;kk++){
          //if gap between particles large enough: no consider as candidate
          //PetscPrintf(PETSC_COMM_WORLD,"  Pair %d,%d \n",jj,kk);
          if(sqrt(SQ(xxst[jj]-xxst[kk])+SQ(yyst[jj]-yyst[kk])) < rrst[jj]+rrst[kk]+2.0*rc) if(fabs(xxst[jj]-xxst[kk]) > fabs(yyst[jj]-yyst[kk])) { //compute coordinates 
            //Define variables
            //PetscPrintf(PETSC_COMM_WORLD,"   There is possible contact with pair %d,%d \n",jj,kk);
            x2=xxst[jj];
            x1=xxst[kk];
            y2=yyst[jj];
            y1=yyst[kk];
            r2=rrst[jj];
            r1=rrst[kk];
            //PetscPrintf(PETSC_COMM_WORLD," x1 %e y1 %e r1 %e x2 %e y2 %e r2 %e \n",x1,y1,r1,x2,y2,r2);
            AA = SQ(r2+rc) -SQ(r1+rc) + SQ(x1) +SQ(y1) -SQ(x2) - SQ(y2);
            BB = SQ((y1-y2)/(x1-x2));
            CC = AA*(y1-y2)/SQ(x1-x2);
            EE = SQ(AA)/4.0/SQ(x1-x2);
            a_ = BB+1.0;
            b_ = 2.0*x1*(y1-y2)/(x1-x2)-2.0*y1-CC;
            c_ = EE-x1*AA/(x1-x2) - SQ(r1+rc) +SQ(x1) +SQ(y1);
            //PetscPrintf(PETSC_COMM_WORLD," AA %e BB %e CC %e EE %e a_ %e b_ %e c_ %e \n",AA,BB,CC,EE,a_,b_,c_);
            if (SQ(b_) < 4.0*a_*c_) PetscPrintf(PETSC_COMM_WORLD,"ERROR! no contact between three circles!\n");
            else {
              y_can = (-b_ + sqrt(SQ(b_) - 4.0*a_*c_))/2.0/a_;
              x_can = -(y1-y2)/(x1-x2)*y_can + AA/2.0/(x1-x2);
              //PetscPrintf(PETSC_COMM_WORLD,"    xX %.3e yY %.3e \n",x_can,y_can);
              dist=1.0e6;
              flag = 1;
              for(ll=0;ll<n_act;ll++){
                dist = sqrt(SQ(x_can-centX[ll])+SQ(y_can-centY[ll]));
                if(1.01*dist < rc+radius[ll]) flag = 0;
              }
              if(x_can<rc || x_can > user->Lx-rc) flag = 0;
              if(x1>x2) {xmax = x1; xmin = x2;}
              else {xmax = x2; xmin = x1;}
              if(x_can>xmax || x_can<xmin) flag = 0;
              if(y_can + rc > user->Ly) flag=0;
              if(flag==1){
                //PetscPrintf(PETSC_COMM_WORLD,"     No overlap with previous particles \n");
                xc_cand[cand] = x_can;
                yc_cand[cand] = y_can;
                cand++;
              } //else PetscPrintf(PETSC_COMM_WORLD,"     Overlap with previous particles \n");
            }
            if(cand == n_cand) {
              kk = upp;
              jj = upp-1;
            }
          }
        }
      }
      PetscPrintf(PETSC_COMM_WORLD,"     New %d candidates \n",cand-init_cand);
    }
    //pick among candidates
    if(flag_nextpart==0 && cand>0){
      PetscPrintf(PETSC_COMM_WORLD,"    Pick amongst %d candidates the lowest one \n",cand);
      dist = 1.0e6;
      for(jj=0;jj<cand;jj++){ //closer to xc, or lowest? lowest for now
        if(yc_cand[jj]<dist) {
          cand_pick = jj;
          dist = yc_cand[jj];
        }
      }
      //give values
      PetscPrintf(PETSC_COMM_WORLD,"     new sed grain %d!!  x %.3e  y %.3e  r %.3e \n",n_act,xc_cand[cand_pick],yc_cand[cand_pick],rc);
      centX[n_act] = xc_cand[cand_pick];
      centY[n_act] = yc_cand[cand_pick];
      radius[n_act] = rc;
      n_act++;
    } 
    if(flag_nextpart==0 && cand==0) {
      PetscPrintf(PETSC_COMM_WORLD,"    NO candidates for particle %d. REPEAT WITH DIFFERENT xc \n",ii);
      fail++;
      repeat=1;
    }
    
    if(n_act==tot_part) {
      PetscPrintf(PETSC_COMM_WORLD,"\n DONE: %d sed grains in %d iterations \n\n", n_act, ii+1+fail);
      ii=tot_part+1;
    }

    if(fail>200) {
      PetscPrintf(PETSC_COMM_WORLD,"Too much iterations to create grains, we consider the sample is created!\n");
      break;
    }

  }

  if(n_act != tot_part) PetscPrintf(PETSC_COMM_WORLD," NOT ENOUGH particles %d (%d) in %d iterations\n\n", n_act, tot_part, ii+fail);

  ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_SELF,"Rank %d,%d \n",user->iga->proc_ranks[0],user->iga->proc_ranks[1]);
  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Iteration amongst grains finished. Now, communication \n");

  //----- communication
  ierr = MPI_Bcast(centX,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(centY,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(radius,tot_part,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF,"Communication finished!\n");
  //ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_actsed = n_act;
  for(jj=0;jj<n_act;jj++){
    user->centsed[0][jj] = centX[jj];
    user->centsed[1][jj] = centY[jj];
    user->radiussed[jj] = radius[jj];
    //PetscPrintf(PETSC_COMM_SELF,"Sed grains: points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
  }


  //-------- define the Phi_sed values
  IGAElement element;
  IGAPoint point;
  PetscReal sed=0.0;
  PetscInt aa,ind=0;

  ierr = IGABeginElement(user->iga,&element);CHKERRQ(ierr);
  while (IGANextElement(user->iga,element)) {
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
        sed=0.0;
        for(aa=0;aa<user->n_actsed;aa++){
          dist=sqrt(SQ(point->mapX[0][0]-user->centsed[0][aa])+SQ(point->mapX[0][1]-user->centsed[1][aa]));
          sed += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
        }
        if(sed>1.0) sed=1.0;
        //PetscPrintf(PETSC_COMM_SELF," sed %.3f \n",sed);
        user->Phi_sed[ind] = sed;
        ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF," ind  %d \n",ind);

  PetscFunctionReturn(0); 
}

// // Define function to read ice grains from file
// PetscErrorCode ReadGrainData(const char *filename, AppCtx *user) {
//     PetscErrorCode ierr;
//     PetscFunctionBegin;

//       PetscPrintf(PETSC_COMM_WORLD,"--------------------- Grain Read Begin --------------------------\n");

//     FILE *file = fopen(filename, "r");
//     if (!file) {
//         SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open file: %s", filename);
//     }

//     PetscInt grainCount = 0;
//     PetscReal x, y, r;
//     while (fscanf(file, "%lf %lf %lf", &x, &y, &r) == 3) {
//         if (grainCount >= 200) {
//             fclose(file);  // Make sure to close the file before returning
//             SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exceeds maximum number of grains");
//         }
//         user->cent[0][grainCount] = x;
//         user->cent[1][grainCount] = y;
//         user->radius[grainCount] = r;
//         grainCount++;
//     }

//     fclose(file);
//     user->NCice = grainCount;  // Assuming this is for ice grains
//     PetscFunctionReturn(0);
// }



PetscErrorCode InitialIceGrains(IGA iga,AppCtx *user)
{
  // Begin function: InitialIceGrains
  PetscErrorCode ierr;
  PetscFunctionBegin;

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"--------------------- ICE GRAINS --------------------------\n");
  }

  // Define ice grain parameters
  PetscReal rad = user->RCice;
  PetscReal rad_dev = user->RCice_dev;
  PetscInt  numb_clust = user->NCice, ii,jj,tot=10000;
  PetscInt  l, dim=user->dim, n_act=0,flag,seed=14;

  // Unpack user data
  PetscInt readFlag = user->readFlag;

  if (readFlag == 1)
  { // Read ice grains from file
    FILE          *file;
    char          grainDataFile[PETSC_MAX_PATH_LEN];

    // Copy the file path to the grainDataFile variable
    const char *inputFile          = getenv("inputFile");
    PetscStrcpy(grainDataFile, inputFile);

    // PetscStrcpy(grainDataFile, "/Users/jacksonbaglino/PetIGA-3.20/demo/input/grainReadFile-10_s1-10.dat");
    PetscPrintf(PETSC_COMM_WORLD,"Reading grains from %s\n\n\n", grainDataFile);

    // Function to read ice grains from file:
    file = fopen(grainDataFile, "r");
    if (!file) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open file: %s", grainDataFile);
    }

    PetscInt grainCount = 0;
    PetscReal x, y, z, r;
    int readCount;
    while ((readCount = fscanf(file, "%lf %lf %lf %lf", &x, &y, &z, &r)) >= 3) {
        if (grainCount >= 200) {
            fclose(file);  // Make sure to close the file before returning
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Exceeds maximum number of grains");
        }
        user->cent[0][grainCount] = x;
        user->cent[1][grainCount] = y;

        if (dim == 3) {
            if (readCount == 4) {
                user->cent[2][grainCount] = z;
                user->radius[grainCount] = r;
            } else if (readCount == 3) {
                user->cent[2][grainCount] = user->Lz / 2.0;
                user->radius[grainCount] = z;  // The third value in this case is the radius
            }
        } else {
            user->radius[grainCount] = r;
        }
        
        grainCount++;

        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, " new ice grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n", grainCount, x, y, (readCount == 4) ? z : user->Lz / 2.0, r);
        }
    }

    fclose(file);
    user->NCice = grainCount;  // Assuming this is for ice grains
    user->n_act = grainCount;
    PetscFunctionReturn(0);

  } else { 
    // Generate ice grains
    PetscPrintf(PETSC_COMM_WORLD,"Generating ice grains\n\n\n");
    if(user->NCice==0) {
      user->n_act = 0;
      PetscPrintf(PETSC_COMM_WORLD,"No ice grains\n\n");
      PetscFunctionReturn(0);
    }

    //----- cluster info
    PetscReal centX[3][numb_clust], radius[numb_clust];
    PetscRandom randcX,randcY,randcZ,randcR;
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcX);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcY);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcR);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(randcX,0.0,user->Lx);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(randcY,0.0,user->Ly);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(randcR,rad*(1.0-rad_dev),rad*(1.0+rad_dev));CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(randcX,seed+24+9*iga->elem_start[0]+11*iga->elem_start[1]);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(randcY,seed+numb_clust*35+5*iga->elem_start[1]+3*iga->elem_start[0]);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(randcR,seed*numb_clust+6*iga->proc_ranks[1]+5*iga->elem_start[0]+9);CHKERRQ(ierr);
    ierr = PetscRandomSeed(randcX);CHKERRQ(ierr);
    ierr = PetscRandomSeed(randcY);CHKERRQ(ierr);
    ierr = PetscRandomSeed(randcR);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randcX);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randcY);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(randcR);CHKERRQ(ierr);

    if(dim==3)
    { // 3D
      ierr = PetscRandomCreate(PETSC_COMM_WORLD,&randcZ);CHKERRQ(ierr);
      ierr = PetscRandomSetInterval(randcZ,0.0,user->Lz);CHKERRQ(ierr);
      ierr = PetscRandomSetSeed(randcZ,seed+iga->elem_width[2]+5*iga->elem_start[0]);CHKERRQ(ierr);
      ierr = PetscRandomSeed(randcZ);CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(randcZ);CHKERRQ(ierr);
    }

    // Initialize ice grains
    PetscReal xc[3], rc=0.0, dist=0.0;
    xc[0] = xc[1] = xc[2] = 0.0;

    // Generate ice grains from random values
    for(ii=0;ii<tot*numb_clust;ii++)
    {
      ierr=PetscRandomGetValue(randcX,&xc[0]);CHKERRQ(ierr);
      ierr=PetscRandomGetValue(randcY,&xc[1]);CHKERRQ(ierr);
      ierr=PetscRandomGetValue(randcR,&rc);CHKERRQ(ierr);

      if(dim==3) 
      {
        ierr=PetscRandomGetValue(randcZ,&xc[2]);CHKERRQ(ierr);
      }

      flag=1;

      // Check if ice grain is within domain
      if(xc[0]<rc || xc[0]>user->Lx-rc) flag = 0;
      if(xc[1]<rc || xc[1]>user->Ly-rc) flag = 0;
      if(dim==3) if(xc[2]<rc || xc[2]>user->Lz-rc) flag = 0;
      
      // Check if ice grain overlaps with existing ice grains
      for(jj=0;jj<user->n_actsed;jj++){
        dist = 0.0;
        for(l=0;l<dim;l++) dist += SQ(xc[l]-user->centsed[l][jj]);
        dist = sqrt(dist);
        if(dist< (rc+user->radiussed[jj]) ) flag = 0;
      }
      if(flag==1){
        for(jj=0;jj<n_act;jj++){
          dist = 0.0;
          for(l=0;l<dim;l++) dist += SQ(xc[l]-centX[l][jj]);
          dist = sqrt(dist);
          if(dist< (rc+radius[jj]) ) flag = 0;
        }
      }
      if(flag==1){
        if(dim==3) PetscPrintf(PETSC_COMM_WORLD," new ice grain %d!!  x %.2e  y %.2e  z %.2e  r %.2e \n",n_act,xc[0],xc[1],xc[2],rc);
        else PetscPrintf(PETSC_COMM_WORLD," new ice grain %d!!  x %.2e  y %.2e  r %.2e \n",n_act,xc[0],xc[1],rc);
        for(l=0;l<dim;l++) centX[l][n_act] = xc[l];
        radius[n_act] = rc;
        n_act++;
      }
      if(n_act==numb_clust) {
        PetscPrintf(PETSC_COMM_WORLD," %d ice grains in %d iterations \n\n", n_act,ii+1);
        ii=tot*numb_clust;
      }
    }

    if(n_act != numb_clust) PetscPrintf(PETSC_COMM_WORLD," %d ice grains in maximum number of iterations allowed (%d) \n\n", n_act, ii);

    ierr = PetscRandomDestroy(&randcX);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&randcY);CHKERRQ(ierr);
    if(dim==3) {ierr = PetscRandomDestroy(&randcZ);CHKERRQ(ierr);}
    ierr = PetscRandomDestroy(&randcR);CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_SELF,"before  %.4f %.4f %.4f \n",centX[0],centY[0],radius[0]);

    //----- communication
    for(l=0;l<dim;l++) {ierr = MPI_Bcast(centX[l],numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);}
    ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

    user->n_act = n_act;
    for(jj=0;jj<n_act;jj++){
      for(l=0;l<dim;l++) user->cent[l][jj] = centX[l][jj];
      user->radius[jj] = radius[jj];
      // PetscPrintf(PETSC_COMM_SELF,"Ice grains: points %.4f %.4f %.4f \n",centX[jj],centY[jj],radius[jj]);
    }

  } // End if readFlag == 0
  PetscFunctionReturn(0); 
}


typedef struct {
  PetscScalar soil;
} FieldS;

PetscErrorCode FormInitialSoil2D(IGA igaS,Vec S,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  ierr = IGACreateNodeDM(igaS,1,&da);CHKERRQ(ierr);
  FieldS **u;
  ierr = DMDAVecGetArray(da,S,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscReal dist=0.0,value;
  PetscInt i,j,kk, l=-1;
  if(user->periodic==1) l=user->p-1;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
      PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
      value=0.0;
      for(kk=0;kk<user->n_actsed;kk++){
        dist = sqrt(SQ(x-user->centsed[0][kk])+SQ(y-user->centsed[1][kk]));
        value += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[kk]));
      }
      if(value>1.0) value=1.0;

      u[j][i].soil = value;
    }
  }
  ierr = DMDAVecRestoreArray(da,S,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSoil3D(IGA igaS,Vec S,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  ierr = IGACreateNodeDM(igaS,1,&da);CHKERRQ(ierr);
  FieldS ***u;
  ierr = DMDAVecGetArray(da,S,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscReal dist=0.0,value;
  PetscInt i,j,k, kk, l=-1;
  if(user->periodic==1) l=user->p-1;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
        PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );
        value=0.0;
        for(kk=0;kk<user->n_actsed;kk++){
          dist = sqrt(SQ(x-user->centsed[0][kk])+SQ(y-user->centsed[1][kk])+SQ(z-user->centsed[2][kk]));
          value += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[kk]));
        }
        if(value>1.0) value=1.0;
        
        u[k][j][i].soil = value;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,S,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


typedef struct {
  PetscScalar ice,tem,rhov;
} Field;

PetscErrorCode FormInitialCondition2D(IGA iga, PetscReal t, Vec U,AppCtx *user, 
                                    const char datafile[],const char dataPF[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

  } else if (dataPF[0] != 0){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,2);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);
    IGAAxis axisPF0,axisPF1;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF1,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF1,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
    ierr = IGASetFromOptions(igaPF);CHKERRQ(ierr);
    ierr = IGASetUp(igaPF);CHKERRQ(ierr);
    Vec PF;
    ierr = IGACreateVec(igaPF,&PF);CHKERRQ(ierr);
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)PF,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,dataPF,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(PF,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecStrideScatter(PF,0,U,INSERT_VALUES);
    ierr = VecDestroy(&PF);CHKERRQ(ierr);
    ierr = IGADestroy(&igaPF);CHKERRQ(ierr);

    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j, k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = user->hum0*rho_vs;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 

  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt i,j,k=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+k) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+k) );

        PetscReal dist,ice=0.0;
        PetscInt aa;
        for(aa=0;aa<user->n_act;aa++){
          dist=sqrt(SQ(x-user->cent[0][aa])+SQ(y-user->cent[1][aa]));
          ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
        }
        if(ice>1.0) ice=1.0;

        u[j][i].ice = ice;    
        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = user->hum0*rho_vs;
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

PetscErrorCode FormInitialCondition3D(IGA iga, PetscReal t, Vec U,AppCtx *user, 
                                    const char datafile[],const char dataPF[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

  } else if (dataPF[0] != 0){
    IGA igaPF;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaPF);CHKERRQ(ierr);
    ierr = IGASetDim(igaPF,3);CHKERRQ(ierr);
    ierr = IGASetDof(igaPF,1);CHKERRQ(ierr);
    IGAAxis axisPF0, axisPF1, axisPF2;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF0,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF1,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF1,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF1,user->Ny,0.0,user->Ly,user->C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaPF,2,&axisPF2);CHKERRQ(ierr);
    if(user->periodic==1) {ierr = IGAAxisSetPeriodic(axisPF2,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axisPF2,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF2,user->Nz,0.0,user->Lz,user->C);CHKERRQ(ierr);
    ierr = IGASetFromOptions(igaPF);CHKERRQ(ierr);
    ierr = IGASetUp(igaPF);CHKERRQ(ierr);
    Vec PF;
    ierr = IGACreateVec(igaPF,&PF);CHKERRQ(ierr);
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)PF,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,dataPF,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(PF,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecStrideScatter(PF,0,U,INSERT_VALUES);
    ierr = VecDestroy(&PF);CHKERRQ(ierr);
    ierr = IGADestroy(&igaPF);CHKERRQ(ierr);

    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field ***u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j,k, l=-1;
    if(user->periodic==1) l=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        for(k=info.zs;k<info.zs+info.zm;k++){
          PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
          PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
          PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );

          u[k][j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly) + user->grad_temp0[2]*(z-0.5*user->Lz);
          PetscScalar rho_vs, temp=u[k][j][i].tem;
          RhoVS_I(user,temp,&rho_vs,NULL);
          u[k][j][i].rhov = user->hum0*rho_vs;
        }
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 

  } else {
    DM da;
    ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
    Field ***u;
    ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

    PetscInt i,j,k, l=-1;
    if(user->periodic==1) k=user->p -1;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        for(k=info.zs;k<info.zs+info.zm;k++){
          PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx+l) );
          PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my+l) );
          PetscReal z = user->Lz*(PetscReal)k / ( (PetscReal)(info.mz+l) );

          PetscReal dist,ice=0.0;
          PetscInt aa;
          for(aa=0;aa<user->n_act;aa++){
            dist=sqrt(SQ(x-user->cent[0][aa])+SQ(y-user->cent[1][aa])+SQ(z-user->cent[2][aa]));
            ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
          }
          if(ice>1.0) ice=1.0;

          u[k][j][i].ice = ice;    
          u[k][j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly) + user->grad_temp0[2]*(z-0.5*user->Lz);
          PetscScalar rho_vs, temp=u[k][j][i].tem;
          RhoVS_I(user,temp,&rho_vs,NULL);
          u[k][j][i].rhov = user->hum0*rho_vs;
        }
      }
    }
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}


int main(int argc, char *argv[]) {

  // Petsc Initialization rite of passage 
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  // Define simulation specific parameters
  AppCtx user;
  PetscInt flag_sedgrav, flag_BC_Tfix, flag_BC_rhovfix;

  //----------------------model parameters
  user.xi_v       = 1.0e-5;       
  user.xi_T       = 1.0e-4;       //--- time scaling depends on the problem temperature and the length scale, 
  user.flag_xiT   = 1;            //    note kinetics change 2-3 orders of magnitude from 0 to -70 C. 
                                  //    xi_v > 1e2*Lx/beta_sub;      xi_t > 1e4*Lx/beta_sub;   xi_v>1e-5; xi_T>1e-5;

  // user.eps        = 9.1e-7;      //--- usually: eps < 1.0e-7, in some setups this limitation can be relaxed (see Manuscript-draft)
	user.Lambd      = 1.0;          //    for low temperatures (T=-70C), we might have eps < 1e-11
  user.air_lim    = 1.0e-6;
  user.nsteps_IC  = 10;

  user.lat_sub    = 2.83e6;
  user.thcond_ice = 2.29; //1.27e-6
  user.thcond_met = 36.0; //1.32e-7
  user.thcond_air = 0.02; //1.428e-5
  user.cp_ice     = 1.96e3;
  user.cp_met     = 4.86e2;
  user.cp_air     = 1.044e3;
  user.rho_ice    = 919.0;
  user.rho_met    = 7753.0;
  user.rho_air    = 1.341;
  user.dif_vap    = 2.178e-5;
  user.T_melt     = 0.0;
  user.flag_it0   = 1;
  user.flag_tIC   = 0;

  user.readFlag   = 1; // 0: generate ice grains, 1: read ice grains from file

  //---------Gibbs-Thomson parameters 
  user.flag_Tdep  = 1;        // Temperature-dependent GT parameters; 
                              // pretty unstable, need to check implementation!!!

  user.d0_sub0    = 1.0e-9; 
  user.beta_sub0  = 1.4e5;    
  PetscReal gamma_im = 0.033, gamma_iv = 0.109, gamma_mv = 0.056; //76
  PetscReal rho_rhovs = 2.0e5; // at 0C;  rho_rhovs=5e5 at -10C


  // Unpack environment variables
  PetscPrintf(PETSC_COMM_WORLD, "Unpacking environment variables...\n");

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
	
	const char *eps_str 				= getenv("eps");
	
  if (!Nx_str || !Ny_str || !Nz_str || !Lx_str || !Ly_str || !Lz_str || 
      !delt_t_str || !t_final_str || !humidity_str || !temp_str || 
      !grad_temp0X_str || !grad_temp0Y_str || !grad_temp0Z_str || !dim_str || !eps_str) {
      PetscPrintf(PETSC_COMM_WORLD, "Error: One or more environment variables are not set.\n");
      PetscFinalize();
      return EXIT_FAILURE;
  } else {
      PetscPrintf(PETSC_COMM_WORLD, "Environment variables successfully set.\n");
      PetscPrintf(PETSC_COMM_WORLD, "Nx: %s\n", Nx_str);
      PetscPrintf(PETSC_COMM_WORLD, "Ny: %s\n", Ny_str);
      PetscPrintf(PETSC_COMM_WORLD, "Nz: %s\n", Nz_str);
      PetscPrintf(PETSC_COMM_WORLD, "Lx: %s\n", Lx_str);
      PetscPrintf(PETSC_COMM_WORLD, "Ly: %s\n", Ly_str);
      PetscPrintf(PETSC_COMM_WORLD, "Lz: %s\n", Lz_str);
      PetscPrintf(PETSC_COMM_WORLD, "delt_t: %s\n", delt_t_str);
      PetscPrintf(PETSC_COMM_WORLD, "t_final: %s\n", t_final_str);
      PetscPrintf(PETSC_COMM_WORLD, "n_out: %s\n", n_out_str);
      PetscPrintf(PETSC_COMM_WORLD, "humidity: %s\n", humidity_str);
      PetscPrintf(PETSC_COMM_WORLD, "temp: %s\n", temp_str);
      PetscPrintf(PETSC_COMM_WORLD, "grad_temp0X: %s\n", grad_temp0X_str);
      PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Y: %s\n", grad_temp0Y_str);
      PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Z: %s\n", grad_temp0Z_str);
      PetscPrintf(PETSC_COMM_WORLD, "dim: %s\n", dim_str);
      PetscPrintf(PETSC_COMM_WORLD, "eps: %s\n", eps_str);
  }

  char *endptr;
  PetscInt Nx          = strtod(Nx_str, &endptr);
  PetscInt Ny          = strtod(Ny_str, &endptr);
  PetscInt Nz          = strtod(Nz_str, &endptr);

  PetscReal Lx          = strtod(Lx_str, &endptr);
  PetscReal Ly          = strtod(Ly_str, &endptr);
  PetscReal Lz          = strtod(Lz_str, &endptr);

  PetscReal delt_t      = strtod(delt_t_str, &endptr);
  PetscReal t_final     = strtod(t_final_str, &endptr);
  PetscInt n_out        = strtod(n_out_str, &endptr);

  PetscReal humidity    = strtod(humidity_str, &endptr);
  PetscReal temp        = strtod(temp_str, &endptr);

  PetscReal grad_temp0X = strtod(grad_temp0X_str, &endptr);
  PetscReal grad_temp0Y = strtod(grad_temp0Y_str, &endptr);
  PetscReal grad_temp0Z = strtod(grad_temp0Z_str, &endptr);

  PetscInt dim          = strtod(dim_str, &endptr);
  
	PetscReal eps         = strtod(eps_str, &endptr);

  // Verify that conversion was successful
  if (*endptr != '\0') {
      PetscPrintf(PETSC_COMM_WORLD, "Error: One or more environment variables contain invalid values.\n");
      PetscFinalize();
      return EXIT_FAILURE;
  }

  // Define the polynomial order of basis functions and global continuity order
  PetscInt  l,m, p=1, C=0; //dim=2;
  user.p=p; user.C=C;  user.dim=dim;
  user.Lx=Lx; user.Ly=Ly; user.Lz=Lz; 
  user.Nx=Nx; user.Ny=Ny; user.Nz=Nz;
  user.eps=eps;

  // grains!
  flag_sedgrav    = 0; 
  user.NCsed      = 0; //less than 200, otherwise update in user
  user.RCsed      = 0.8e-5;
  user.RCsed_dev  = 0.4;

  user.NCice      = 2; //less than 200, otherwise update in user
  user.RCice      = 0.2e-4;
  user.RCice_dev  = 0.5;

  //initial conditions
  user.hum0          = humidity;
  user.temp0         = temp;
  user.grad_temp0[0] = grad_temp0X;  user.grad_temp0[1] = grad_temp0Y;  user.grad_temp0[2] = grad_temp0Z;

  //boundary conditions
  user.periodic   = 0;          // periodic >> Dirichlet   
  flag_BC_Tfix    = 1;
  flag_BC_rhovfix = 0;
  if(user.periodic==1 && flag_BC_Tfix==1) flag_BC_Tfix=0;
  if(user.periodic==1 && flag_BC_rhovfix==1) flag_BC_rhovfix=0;

  //output
  user.outp = 2; // if 0 -> output according to t_interv
  user.t_out = 0;    // user.t_interv = t_final/(n_out-1); //output every t_interv
  user.t_interv =  36.0; //output every t_interv

  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 5;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.01*delt_t, dtmax = 0.5*user.t_interv;
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 10;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.2e  dt_range %.2e-%.2e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscInt size;
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  PetscPrintf(PETSC_COMM_WORLD, "Running on %d processes.\n\n\n", size);


  // G-T kinetic parameters
  user.diff_sub = 0.5*(user.thcond_air/user.rho_air/user.cp_air + user.thcond_ice/user.rho_ice/user.cp_ice);
  user.Etai       = gamma_iv + gamma_im - gamma_mv;
  user.Etam       = gamma_mv + gamma_im - gamma_iv;
  user.Etaa       = gamma_iv + gamma_mv - gamma_im;
  PetscReal a1=5.0, a2=0.1581; // at 0C;  rho_rhovs=5e5 at -10C    // Might need to change these values (depends on the temperature, rho_rhovs=10e5 at -20C)
  PetscReal d0_sub,  beta_sub, lambda_sub, tau_sub;
  d0_sub = user.d0_sub0/rho_rhovs;  beta_sub = user.beta_sub0/rho_rhovs; 
  lambda_sub    = a1*user.eps/d0_sub;
  tau_sub       = user.eps*lambda_sub*(beta_sub/a1 + a2*user.eps/user.diff_sub + a2*user.eps/user.dif_vap);

  user.mob_sub    = 1*user.eps/3.0/tau_sub; 
  user.alph_sub   = 10*lambda_sub/tau_sub;
  if(user.flag_Tdep==0) PetscPrintf(PETSC_COMM_WORLD,"FIXED PARAMETERS: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n\n",tau_sub,lambda_sub,user.mob_sub,user.alph_sub);
  else PetscPrintf(PETSC_COMM_WORLD,"TEMPERATURE DEPENDENT G-T PARAMETERS \n\n");
  

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  char PFgeom[PETSC_MAX_PATH_LEN] = {0};
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "NASA Options", "IGA");
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_cond","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_PFgeom","Load initial ice geometry from file",__FILE__,PFgeom,PFgeom,sizeof(PFgeom),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-NASA_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-NASA_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"phaseice"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"temperature"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"vap_density"); CHKERRQ(ierr);

  IGAAxis axis0, axis1, axis2;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);
  if(dim==3){
    ierr = IGAGetAxis(iga,2,&axis2);CHKERRQ(ierr);
    if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis2,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axis2,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis2,Nz,0.0,Lz,C);CHKERRQ(ierr);
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  PetscInt nmb = iga->elem_width[0]*iga->elem_width[1]*SQ(p+1);
  if(dim==3) nmb = iga->elem_width[0]*iga->elem_width[1]*iga->elem_width[2]*CU(p+1);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.Phi_sed);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Phi_sed,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.alph);CHKERRQ(ierr);
  ierr = PetscMemzero(user.alph,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.mob);CHKERRQ(ierr);
  ierr = PetscMemzero(user.mob,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  //Boundary Condition
  if(flag_BC_rhovfix==1){
    PetscReal rho0_vs;
    RhoVS_I(&user,user.temp0,&rho0_vs,NULL);
    for(l=0;l<dim;l++) for(m=0;m<2;m++) ierr = IGASetBoundaryValue(iga,l,m,2,user.hum0*rho0_vs);CHKERRQ(ierr);
  }
  if(flag_BC_Tfix==1){
    PetscReal T_BC[dim][2], LL[dim];
    LL[0] = Lx; LL[1]=Ly; LL[2]=Lz;
    for(l=0;l<dim;l++) for(m=0;m<2;m++) T_BC[l][m] = user.temp0 + (2.0*m-1)*user.grad_temp0[l]*0.5*LL[l];
    for(l=0;l<dim;l++) for(m=0;m<2;m++) ierr = IGASetBoundaryValue(iga,l,m,1,T_BC[l][m]);CHKERRQ(ierr);
  }

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,t_final);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,delt_t);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  if (monitor) {ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ts->adap = adap;
  ts->NRmin = NRmin;
  ts->NRmax = NRmax;
  ts->factor = factor;
  ts->dtmax = dtmax;
  ts->dtmin = dtmin;
  ts->max_reject = max_rej;
  ts->max_snes_failures = -1;

  SNES nonlin;
  ierr = TSGetSNES(ts,&nonlin);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(nonlin,SNESDOFConvergence,&user,NULL);CHKERRQ(ierr);

  if(user.NCsed>0){
    if(flag_sedgrav==1) {
      if(dim==2) {ierr = InitialSedGrainsGravity(iga,&user);CHKERRQ(ierr);}
      else {
        PetscPrintf(PETSC_COMM_WORLD,"Pluviation Script not prepared for 3D. Run no-gravity \n");
        ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);
      }
    } else {ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);}

    //output sediment/metal in OutputMonitor function --> single file for all variables

    IGA igaS;   IGAAxis axis0S, axis1S, axis2S;
    ierr = IGACreate(PETSC_COMM_WORLD,&igaS);CHKERRQ(ierr);
    ierr = IGASetDim(igaS,dim);CHKERRQ(ierr);
    ierr = IGASetDof(igaS,1);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaS,0,&axis0S);CHKERRQ(ierr);
    if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0S,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axis0S,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis0S,Nx,0.0,Lx,C);CHKERRQ(ierr);
    ierr = IGAGetAxis(igaS,1,&axis1S);CHKERRQ(ierr);
    if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1S,PETSC_TRUE);CHKERRQ(ierr);}
    ierr = IGAAxisSetDegree(axis1S,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis1S,Ny,0.0,Ly,C);CHKERRQ(ierr);
    if(dim==3){
      ierr = IGAGetAxis(igaS,2,&axis2S);CHKERRQ(ierr);
      if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis2S,PETSC_TRUE);CHKERRQ(ierr);}
      ierr = IGAAxisSetDegree(axis2S,p);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis2S,Nz,0.0,Lz,C);CHKERRQ(ierr);
    }
    ierr = IGASetFromOptions(igaS);CHKERRQ(ierr);
    ierr = IGASetUp(igaS);CHKERRQ(ierr);

    Vec S;
    ierr = IGACreateVec(igaS,&S);CHKERRQ(ierr);
    //ierr = IGACreateVec(igaS,&user.Sed);CHKERRQ(ierr);
    if(dim==2) {ierr = FormInitialSoil2D(igaS,S,&user);CHKERRQ(ierr);}
    else {ierr = FormInitialSoil3D(igaS,S,&user);CHKERRQ(ierr);}
    //ierr = VecCopy(S,user.Sed);CHKERRQ(ierr);

    const char *env="folder"; char *dir; dir=getenv(env);
    char filename[256],filevect[256];
    sprintf(filename, "%s/igasoil.dat", dir);
    ierr=IGAWrite(igaS,filename);CHKERRQ(ierr);
    
    sprintf(filevect, "%s/soil.dat", dir);
    ierr=IGAWriteVec(igaS,S,filevect);CHKERRQ(ierr);

    ierr = VecDestroy(&S);CHKERRQ(ierr);
    ierr = IGADestroy(&igaS);CHKERRQ(ierr);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"No sed grains\n\n");
    user.n_actsed= 0;
  }

  ierr = InitialIceGrains(iga,&user);CHKERRQ(ierr);

  // Print the variables
  PetscPrintf(PETSC_COMM_WORLD, "Nx: %f\n", user.Nx);
  PetscPrintf(PETSC_COMM_WORLD, "Ny: %f\n", user.Ny);
  PetscPrintf(PETSC_COMM_WORLD, "Nz: %f\n", user.Nz);
  PetscPrintf(PETSC_COMM_WORLD, "Lx: %f\n", user.Lx);
  PetscPrintf(PETSC_COMM_WORLD, "Ly: %f\n", user.Ly);
  PetscPrintf(PETSC_COMM_WORLD, "Lz: %f\n", user.Lz);
  PetscPrintf(PETSC_COMM_WORLD, "delt_t: %f\n", delt_t);
  PetscPrintf(PETSC_COMM_WORLD, "t_final: %f\n", t_final);
  PetscPrintf(PETSC_COMM_WORLD, "humidity: %f\n", humidity);
  PetscPrintf(PETSC_COMM_WORLD, "temp: %f\n", user.temp0);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0X: %f\n", user.grad_temp0[0]);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Y: %f\n", user.grad_temp0[1]);
  PetscPrintf(PETSC_COMM_WORLD, "grad_temp0Z: %f\n", user.grad_temp0[2]);
  PetscPrintf(PETSC_COMM_WORLD, "dim: %d\n", user.dim);
  PetscPrintf(PETSC_COMM_WORLD, "eps: %f\n", user.eps);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(U);CHKERRQ(ierr);
  if(dim==2) {ierr = FormInitialCondition2D(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);}
  else {ierr = FormInitialCondition3D(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);}

  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  //ierr = VecDestroy(&user.Sed);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFree(user.Phi_sed);CHKERRQ(ierr);
  ierr = PetscFree(user.alph);CHKERRQ(ierr);
  ierr = PetscFree(user.mob);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
