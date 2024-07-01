#include <petsc/private/tsimpl.h>
#include <petsc/private/snesimpl.h>
#include "petiga.h"
#include <petscsys.h>

#define SQ(x) ((x)*(x))
#define CU(x) ((x)*(x)*(x))


typedef struct {
  IGA       iga;

  PetscReal eps;
  PetscReal nucleat;
  PetscReal mob_sol,mob_sub,mob_eva,mav,Etai,Etaw,Etaa,alph_sol,alph_sub,\
            alph_eva,Lambd;
  PetscReal thcond_ice,thcond_wat,thcond_air,cp_ice,cp_wat,cp_air,rho_ice,\
            rho_wat,rho_air,dif_vap,lat_sol,lat_sub;
  PetscReal phi_L,air_lim,xi_v,xi_T;
  PetscReal T_melt,temp0,grad_temp0[2],heat;
  PetscReal Lx,Ly,Nx,Ny,Rx,Ry;
  PetscReal RCice,RCsed,RCice_dev,RCsed_dev,centX[2000],centY[2000],radius[2000];
  PetscReal centXsed[2000],centYsed[2000],radiussed[2000];
  PetscReal norm0_0,norm0_1,norm0_2,norm0_3;
  PetscInt  flag_it0, flag_tIC, outp, nsteps_IC,flag_xiT;
  PetscInt  xiT_count, mobch_count;
  PetscInt  p,C,periodic;
  PetscReal *Phi_sed,t_out,t_interv,t_IC;
  PetscInt  NCice,NCsed,n_act,n_actsed;

} AppCtx;

PetscErrorCode SNESDOFConvergence(SNES snes,PetscInt it_number,PetscReal xnorm, 
    PetscReal gnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)
{
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
  // if(snes->prev_dt_red ==1) rtol *= 10.0;

  if(user->flag_it0 == 1){
    atol = 1.0e-12;
    if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
      && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
      && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) ) {
      //&& (solupdv <= rtol*solv || n2dof3 < atol) ) {

    *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }    
  } else {
    atol = 1.0e-25;
    if ( (n2dof0 <= rtol*user->norm0_0 || n2dof0 < atol) 
      && (n2dof1 <= rtol*user->norm0_1 || n2dof1 < atol)  
      && (n2dof2 <= rtol*user->norm0_2 || n2dof2 < atol) ) {

    *reason = SNES_CONVERGED_FNORM_RELATIVE;
    }     
  }

  PetscFunctionReturn(0);
}


void ThermalCond(AppCtx *user, PetscScalar ice, PetscScalar wat, 
                  PetscScalar *cond, PetscScalar *dcond_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-wat-ice;
  if(wat<0.0) {wat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal cond_ice = user->thcond_ice;
  PetscReal cond_wat = user->thcond_wat;
  PetscReal cond_air = user->thcond_air;
  if(cond)      (*cond)  = ice*cond_ice + wat*cond_wat + air*cond_air;
  if(dcond_ice)    (*dcond_ice) = cond_ice*dice-cond_air*dair;

  return;
}

void HeatCap(AppCtx *user, PetscScalar ice, PetscScalar wat, PetscScalar *cp, 
              PetscScalar *dcp_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-wat-ice;
  if(wat<0.0) {wat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal cp_ice = user->cp_ice;
  PetscReal cp_wat = user->cp_wat;
  PetscReal cp_air = user->cp_air;
  if(cp)     (*cp)  = ice*cp_ice + wat*cp_wat + air*cp_air;
  if(dcp_ice)    (*dcp_ice) = cp_ice*dice-cp_air*dair;

  return;
}

void Density(AppCtx *user, PetscScalar ice, PetscScalar wat, PetscScalar *rho, 
              PetscScalar *drho_ice)
{
  PetscReal dice=1.0, dair=1.0;
  PetscReal air = 1.0-wat-ice;
  if(wat<0.0) {wat=0.0;}
  if(ice<0.0) {ice=0.0;dice=0.0;}
  if(air<0.0) {air=0.0;dair=0.0;}
  PetscReal rho_ice = user->rho_ice;
  PetscReal rho_wat = user->rho_wat;
  PetscReal rho_air = user->rho_air;
  if(rho)     (*rho)  = ice*rho_ice + wat*rho_wat + air*rho_air;
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

void Fice(AppCtx *user, PetscScalar ice, PetscScalar wat, PetscScalar *fice, 
          PetscScalar *dfice_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etai  = user->Etai;
  PetscReal air = 1.0-wat-ice;
  if(fice)     (*fice)  = etai*ice*(1.0-ice)*(1.0-2.0*ice) + 2.0*Lambd*ice*wat*wat*air*air;
  if(dfice_ice)    (*dfice_ice) = etai*(1.0-6.0*ice+6.0*ice*ice) + \
                    2.0*Lambd*wat*wat*air*air - 2.0*Lambd*ice*wat*wat*2.0*air;
  
  return;
}

void Fwat(AppCtx *user, PetscScalar ice, PetscScalar wat, PetscScalar *fwat, 
          PetscScalar *dfwat_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etaw  = user->Etaw;
  PetscReal air = 1.0-wat-ice;
  if(fwat)     (*fwat)  = etaw*(wat)*(1.0-wat)*(1.0-2.0*wat) + 2.0*Lambd*ice*ice*wat*air*air;
  if(dfwat_ice)    {
    (*dfwat_ice)  = 2.0*Lambd*2.0*ice*wat*air*air - 2.0*Lambd*ice*ice*wat*2.0*air;

  }

  
  return;
}

void Fair(AppCtx *user, PetscScalar ice, PetscScalar wat, PetscScalar *fair, 
          PetscScalar *dfair_ice)
{

  PetscReal Lambd = user->Lambd;
  PetscReal etaa  = user->Etaa;
  PetscReal air = 1.0-wat-ice;
  if(fair)     (*fair)  = etaa*air*(1.0-air)*(1.0-2.0*air) + 2.0*Lambd*ice*ice*wat*wat*air;
  if(dfair_ice)    {
    (*dfair_ice)  = -etaa*(1.0-air)*(1.0-2.0*air) + etaa*air*(1.0-2.0*air) + etaa*air*(1.0-air)*2.0;
    (*dfair_ice) += 2.0*Lambd*2.0*ice*wat*wat*air - 2.0*Lambd*ice*ice*wat*wat;
  }
  
  return;
}

/*void WATER(AppCtx *user, IGAPoint pnt,PetscScalar *wat)
{
  PetscReal x = pnt->point[0];
  PetscReal y = pnt->point[1];
  PetscReal horiz_strip = 0.5-0.5*tanh(0.5/user->eps*(x-user->Rx*user->Lx));
  PetscReal vert_strip = 0.5-0.5*tanh(0.5/user->eps*(sqrt(SQ(y-0.5*user->Ly))-0.5*user->Ry*user->Ly));
  if(wat) (*wat) = horiz_strip*vert_strip; //0.5-0.5*tanh(0.5/user->eps*(pnt->point[0]-0.1*user->Lx));
  
  return;
}

void WaterXY(AppCtx *user,PetscReal x,PetscReal y,PetscScalar *wat)
{

  PetscReal horiz_strip = 0.5-0.5*tanh(0.5/user->eps*(x-user->Rx*user->Lx));
  PetscReal vert_strip = 0.5-0.5*tanh(0.5/user->eps*(sqrt(SQ(y-0.5*user->Ly))-0.5*user->Ry*user->Ly));
  if(wat) (*wat) = horiz_strip*vert_strip; //0.5-0.5*tanh(0.5/user->eps*(pnt->point[0]-0.1*user->Lx));
  
  return;
}
*/

PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;

  PetscReal eps = user->eps;
  PetscReal Etai = user->Etai;
  PetscReal Etaw = user->Etaw;
  PetscReal Etaa = user->Etaa;
  PetscReal ETA = Etaa*Etai + Etaa*Etaw + Etaw*Etai; 
  PetscReal alph_sub = user->alph_sub;
  PetscReal rho_ice = user->rho_ice;
  PetscReal lat_sub = user->lat_sub;
  PetscReal air_lim = user->air_lim;
  PetscReal xi_v = user->xi_v;
  PetscReal xi_T = user->xi_T;
  PetscReal mob = user->mob_sub;
  PetscReal rhoSE = rho_ice;
  PetscReal heat = user->heat;
  PetscReal sed = user->Phi_sed[pnt->index + pnt->count*pnt->parent->index];

  if(pnt->atboundary){

    //PetscScalar sed;

    //WaterXY(user,pnt->point[0],pnt->point[1],&wat);

 /*   PetscScalar sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, grad_ice[2],modgradice;
    ice          = sol[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];
    modgradice = sqrt(SQ(grad_ice[0])+SQ(grad_ice[1]));

    PetscScalar air,grad_air[2],modgradair;
    //air          = sol[1]; 
    //grad_air[0]  = grad_sol[1][0];
    //grad_air[1]  = grad_sol[1][1];
    //modgradair = sqrt(SQ(grad_air[0])+SQ(grad_air[1]));

    PetscReal mob;
*/
/*    const PetscReal *N0; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    
    PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
    PetscInt a,nen=pnt->nen;
    for(a=0; a<nen; a++) {
      R[a][0] = 0.0; //-N0[a]*3.0*eps*mob*modgradice*costhet;
      R[a][1] -= N0[a]*heat; //N0[a]*3.0*eps*mob*modgradair*costhet;
      R[a][2] = 0.0;
    }
 */     
  } else  {
    
    PetscScalar sol_t[3],sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, ice_t, grad_ice[2];
    ice          = sol[0]; 
    ice_t        = sol_t[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];

 //   PetscScalar wat;
 //   WaterXY(user,pnt->point[0],pnt->point[1],&wat);

    PetscScalar air, air_t;
    air          = 1.0-sed-ice;
    air_t        = -ice_t;

    PetscScalar tem, tem_t, grad_tem[2];
    tem          = sol[1];
    tem_t        = sol_t[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar rhov, rhov_t, grad_rhov[2];
    rhov           = sol[2];
    rhov_t         = sol_t[2];
    grad_rhov[0]   = grad_sol[2][0];
    grad_rhov[1]   = grad_sol[2][1];


    PetscReal thcond,cp,rho,difvap,rhoI_vs,fice,fwat,fair;
    ThermalCond(user,ice,sed,&thcond,NULL);
    HeatCap(user,ice,sed,&cp,NULL);
    Density(user,ice,sed,&rho,NULL);
    VaporDiffus(user,tem,&difvap,NULL);
    RhoVS_I(user,tem,&rhoI_vs,NULL);
    Fice(user,ice,sed,&fice,NULL);
    Fwat(user,ice,sed,&fwat,NULL);
    Fair(user,ice,sed,&fair,NULL);


    const PetscReal *N0,(*N1)[2]; 
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
        R_ice += 3.0*mob*eps*(N1[a][0]*grad_ice[0] + N1[a][1]*grad_ice[1]);
        R_ice += N0[a]*mob*3.0/eps/ETA*((Etaw+Etaa)*fice - Etaa*fwat - Etaw*fair);
        R_ice -= N0[a]*alph_sub*ice*ice*air*air*(rhov-rhoI_vs)/rho_ice;        

        R_tem  = rho*cp*N0[a]*tem_t;
        R_tem += xi_T*thcond*(N1[a][0]*grad_tem[0] + N1[a][1]*grad_tem[1]);
        R_tem += xi_T*rho*lat_sub*N0[a]*air_t;

        R_vap  = N0[a]*rhov*air_t;
        if(air>air_lim){
          R_vap += N0[a]*air*rhov_t;
          R_vap += xi_v*difvap*air*(N1[a][0]*grad_rhov[0] + N1[a][1]*grad_rhov[1]);
        } else {
          R_vap += N0[a]*air_lim*rhov_t;
          R_vap += xi_v*difvap*air_lim*(N1[a][0]*grad_rhov[0] + N1[a][1]*grad_rhov[1]);
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

  PetscReal eps = user->eps;
  PetscReal Etai = user->Etai;
  PetscReal Etaw = user->Etaw;
  PetscReal Etaa = user->Etaa;
  PetscReal ETA = Etaa*Etai + Etaa*Etaw + Etaw*Etai; 
  PetscReal alph_sub = user->alph_sub;
  PetscReal rho_ice = user->rho_ice;
  PetscReal lat_sub = user->lat_sub;
  PetscReal air_lim = user->air_lim;
  PetscReal xi_v = user->xi_v;
  PetscReal xi_T = user->xi_T;
  PetscReal mob = user->mob_sub;
  PetscReal rhoSE = rho_ice;

  PetscReal sed = user->Phi_sed[pnt->index + pnt->count*pnt->parent->index];

 if(pnt->atboundary){

 /*   PetscScalar sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, grad_ice[2],modgradice;
    ice          = sol[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];
    modgradice = sqrt(SQ(grad_ice[0])+SQ(grad_ice[1]));
    if(modgradice<1.0e-5) modgradice=1.0e-5;

    PetscScalar air,grad_air[2],modgradair;
    //air          = sol[1];  
    //grad_air[0]  = grad_sol[1][0];
    //grad_air[1]  = grad_sol[1][1];
    //modgradair = sqrt(SQ(grad_air[0])+SQ(grad_air[1]));
    //if(modgradair<1.0e-5) modgradair=1.0e-5;

    PetscReal mob,dmob_ice,dmob_air;

    const PetscReal *N0,(*N1)[2]; 
    IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
    IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);

    PetscInt a,b,nen=pnt->nen;
    PetscScalar (*J)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
    for(a=0; a<nen; a++) {
      for(b=0; b<nen; b++) {

        J[a][0][b][0] += -N0[a]*3.0*eps*dmob_ice*N0[b]*modgradice*costhet;
        J[a][0][b][1] += -N0[a]*3.0*eps*dmob_air*N0[b]*modgradice*costhet;
        J[a][0][b][0] += -N0[a]*3.0*eps*mob*(grad_ice[0]*N1[b][0]+grad_ice[1]*N1[b][1])/modgradice*costhet;
        J[a][1][b][0] += N0[a]*3.0*eps*dmob_ice*N0[b]*modgradair*costhet;
        J[a][1][b][1] += N0[a]*3.0*eps*dmob_air*N0[b]*modgradair*costhet;
        J[a][1][b][1] += N0[a]*3.0*eps*mob*(grad_air[0]*N1[b][0]+grad_air[1]*N1[b][1])/modgradair*costhet;

      }
    }
*/    
  } else {

    PetscScalar sol_t[3],sol[3];
    PetscScalar grad_sol[3][2];
    IGAPointFormValue(pnt,V,&sol_t[0]);
    IGAPointFormValue(pnt,U,&sol[0]);
    IGAPointFormGrad (pnt,U,&grad_sol[0][0]);

    PetscScalar ice, ice_t, grad_ice[2];
    ice          = sol[0]; 
    ice_t        = sol_t[0]; 
    grad_ice[0]  = grad_sol[0][0];
    grad_ice[1]  = grad_sol[0][1];

  //  PetscScalar wat;
  //  WaterXY(user,pnt->point[0],pnt->point[1],&wat);

    PetscScalar air, air_t;
    air          = 1.0-sed-ice;
    air_t        = -ice_t;

    PetscScalar tem, tem_t, grad_tem[2];
    tem          = sol[1];
    tem_t        = sol_t[1];
    grad_tem[0]  = grad_sol[1][0];
    grad_tem[1]  = grad_sol[1][1]; 

    PetscScalar rhov, rhov_t, grad_rhov[2];
    rhov           = sol[2];
    rhov_t         = sol_t[2];
    grad_rhov[0]   = grad_sol[2][0];
    grad_rhov[1]   = grad_sol[2][1];


    PetscReal thcond,dthcond_ice;
    ThermalCond(user,ice,sed,&thcond,&dthcond_ice);
    PetscReal cp,dcp_ice;
    HeatCap(user,ice,sed,&cp,&dcp_ice);
    PetscReal rho,drho_ice;
    Density(user,ice,sed,&rho,&drho_ice);
    PetscReal difvap,d_difvap;
    VaporDiffus(user,tem,&difvap,&d_difvap);
    PetscReal rhoI_vs,drhoI_vs;
    RhoVS_I(user,tem,&rhoI_vs,&drhoI_vs);
    PetscReal fice,fice_ice;
    Fice(user,ice,sed,&fice,&fice_ice);
    PetscReal fwat,fwat_ice;
    Fwat(user,ice,sed,&fwat,&fwat_ice);
    PetscReal fair,fair_ice;
    Fair(user,ice,sed,&fair,&fair_ice);


    const PetscReal *N0,(*N1)[2]; 
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
          J[a][0][b][0] += 3.0*mob*eps*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);

          J[a][0][b][0] += N0[a]*mob*3.0/eps/ETA*((Etaw+Etaa)*fice_ice - Etaa*fwat_ice - Etaw*fair_ice)*N0[b];
          J[a][0][b][0] -= N0[a]*alph_sub*2.0*ice*N0[b]*air*air*(rhov-rhoI_vs)/rho_ice;
          J[a][0][b][0] += N0[a]*alph_sub*ice*ice*2.0*air*N0[b]*(rhov-rhoI_vs)/rho_ice;
          J[a][0][b][1] += N0[a]*alph_sub*ice*ice*air*air*drhoI_vs*N0[b]/rho_ice;
          J[a][0][b][2] -= N0[a]*alph_sub*ice*ice*air*air*N0[b]/rho_ice;


        //temperature
          J[a][1][b][1] += shift*rho*cp*N0[a]*N0[b];
          J[a][1][b][0] += drho_ice*N0[b]*cp*N0[a]*tem_t;
          J[a][1][b][0] += rho*dcp_ice*N0[b]*N0[a]*tem_t;
          J[a][1][b][0] += xi_T*dthcond_ice*N0[b]*(N1[a][0]*grad_tem[0] + N1[a][1]*grad_tem[1]);
          J[a][1][b][1] += xi_T*thcond*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);
          J[a][1][b][0] += xi_T*drho_ice*N0[b]*lat_sub*N0[a]*air_t;
          J[a][1][b][0] -= xi_T*rho*lat_sub*N0[a]*shift*N0[b];

        //vapor density
          J[a][2][b][0] -= N0[a]*rhov*shift*N0[b];
          J[a][2][b][2] += N0[a]*N0[b]*air_t;
          if(air>air_lim){
            J[a][2][b][0] -= N0[a]*N0[b]*rhov_t;
            J[a][2][b][2] += N0[a]*air*shift*N0[b];
            J[a][2][b][0] -= xi_v*difvap*N0[b]*(N1[a][0]*grad_rhov[0] + N1[a][1]*grad_rhov[1]);
            J[a][2][b][1] += xi_v*d_difvap*N0[b]*air*(N1[a][0]*grad_rhov[0] + N1[a][1]*grad_rhov[1]);
            J[a][2][b][2] += xi_v*difvap*air*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);        
          } else {
            J[a][2][b][2] += N0[a]*air_lim*shift*N0[b];
            J[a][2][b][1] += xi_v*d_difvap*N0[b]*air_lim*(N1[a][0]*grad_rhov[0] + N1[a][1]*grad_rhov[1]);
            J[a][2][b][2] += xi_v*difvap*air_lim*(N1[a][0]*N1[b][0] + N1[a][1]*N1[b][1]);
          }
          J[a][2][b][0] += xi_v*N0[a]*rhoSE*shift*N0[b];

        }

      }
    }

  //return 0;
  }
  return 0;
}


PetscErrorCode Integration(IGAPoint pnt, const PetscScalar *U, PetscInt n, 
                            PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  // PetscScalar sol[3];
  PetscScalar sol[3];
  IGAPointFormValue(pnt,U,&sol[0]);

  PetscReal ice     = sol[0]; 
  PetscReal sed     = 0;
  //WaterXY(user,pnt->point[0],pnt->point[1],&wat);
  PetscReal air     = 1.0-sed-ice;
  PetscReal temp    = sol[1];
  PetscReal rhov    = sol[2];
  PetscReal triple = SQ(air)*SQ(sed)*SQ(ice);

  S[0]  = ice;
  S[1]  = triple;
  S[2]  = air;
  S[3]  = temp;
  S[4]  = rhov*air;
  // S[5]  = sed*sed*ice*ice;
  S[5]  = air*air*ice*ice;

  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;


  PetscScalar stats[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
  ierr = IGAComputeScalar(user->iga,U,6,&stats[0],Integration,mctx);CHKERRQ(ierr);
  PetscReal tot_ice   = PetscRealPart(stats[0]);
  PetscReal tot_trip   = PetscRealPart(stats[1]);
  PetscReal tot_air   = PetscRealPart(stats[2]);
  PetscReal tot_temp  = PetscRealPart(stats[3]);
  PetscReal tot_rhov  = PetscRealPart(stats[4]);
  PetscReal sol_interf = PetscRealPart(stats[5]); 

 
//---------------------
  PetscReal dt;
  TSGetTimeStep(ts,&dt);

  if(step==1) user->flag_it0 = 0;

  if(user->flag_tIC==1) if(step==user->nsteps_IC) {
    user->flag_tIC = 0; user->t_IC = t; //user->flag_rtol = 1;
    PetscPrintf(PETSC_COMM_WORLD,"INITIAL_CONDITION!!! \n");
  }
  
  if(step%10==0) PetscPrintf(PETSC_COMM_WORLD,"\nTIME          TIME_STEP     TOT_ICE      TOT_AIR       TEMP      TOT_RHOV     I-W interf   Tot_Tripl \n");
              PetscPrintf(PETSC_COMM_WORLD,"\n(%.0f) %.3e    %.3e   %.3e   %.3e   %.3e   %.3e   %.3e   %.3e \n\n",
                t,t,dt,tot_ice,tot_air,tot_temp,tot_rhov,sol_interf,tot_trip);

  if(step%10==0) {
    char filedata[256];
    const char *env = "folder"; char *dir; dir = getenv(env);

    sprintf(filedata,"%s/Data.dat",dir);
    PetscViewer       view;
    PetscViewerCreate(PETSC_COMM_WORLD,&view);
    PetscViewerSetType(view,PETSCVIEWERASCII);

    if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);

    PetscViewerFileSetName(view,filedata);
    PetscViewerASCIIPrintf(view," %d %e %e %e %e \n",step,t,dt,tot_trip,sol_interf);

    PetscViewerDestroy(&view);
  }

    PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) { //if(step%10==0) {
    char filedata[256];
    const char *env = "folder"; char *dir; dir = getenv(env);

    sprintf(filedata,"%s/SSA_evo.dat",dir);
    PetscViewer       view;
    PetscViewerCreate(PETSC_COMM_WORLD,&view);
    PetscViewerSetType(view,PETSCVIEWERASCII);


    if (step==0) PetscViewerFileSetMode(view,FILE_MODE_WRITE); else PetscViewerFileSetMode(view,FILE_MODE_APPEND);

    PetscViewerFileSetName(view,filedata);
    PetscViewerASCIIPrintf(view,"%e %e %e \n",sol_interf, tot_ice, t);

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
  
  if(step==0) {
    const char *env = "folder"; char *dir; dir = getenv(env);
    char fileiga[256];

    sprintf(fileiga, "%s/igasol.dat", dir);
    ierr = IGAWrite(user->iga,fileiga);CHKERRQ(ierr);
  }

  PetscInt print=0;
  if(user->outp > 0) {
    if(step % user->outp == 0) print=1;
  } else {
    if (t>= user->t_out) print=1;
  }

  if(print==1) {
    user->t_out += user->t_interv;

    const char *env = "folder"; char *dir; dir = getenv(env);
    char filename[256];
    sprintf(filename,"%s/sol%d.dat",dir,step);
    ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode InitialSedGrains(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       nrows, ncols, i, ii, j, jj, n_act;
  FILE           *file;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBegin;

  //----- cluster info
  PetscReal xc=0.0,yc=0.0,rc=0.0, dist=0.0;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  /* Specify the path to your CSV file */
  // PetscStrcpy(filename, "/Users/jacksonbaglino/Desktop/positions.csv");
  PetscStrcpy(filename, "./input/grainReadFile-2.csv");

  /* Open the file for reading */
  file = fopen(filename, "r");
  if (file == NULL) {
    PetscPrintf(PETSC_COMM_WORLD, "Failed to open the file %s\n", filename);
    PetscFinalize();
    return -1;
  }

  /* Count the number of rows and columns in the CSV file */
  nrows = 0;
  ncols = 0;
  char c;
  while ((c = fgetc(file)) != EOF) {
      if (c == ',' && nrows == 0) {
          ncols++;
      } else if (c == '\n') {
          nrows++;
          // ncols++; // Increment ncols by 1 to account for the last column
      }
  }
  // Increment nrows by 1 to account for the last row if it's not empty
  // Same for ncols...
  if (ncols > 0) {
      nrows++;
      ncols++;
  }


  /* Reset the file pointer to the beginning */
  rewind(file);

  /* Allocate memory for the data */
  double **file_data = (double **)malloc(nrows * sizeof(double *));
  if (file_data == NULL) {
      printf("Failed to allocate memory for file_data\n");
      fclose(file);
      return -1;
  }

  for (int i = 0; i < nrows; i++) {
      file_data[i] = (double *)malloc(ncols * sizeof(double));
      if (file_data[i] == NULL) {
          printf("Failed to allocate memory for file_data\n");
          for (int j = 0; j < i; j++) {
              free(file_data[j]);
          }
          free(file_data);
          fclose(file);
          return -1;
      }
  }

  /* Read the data from the CSV file */
  for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
          fscanf(file, "%lf", &file_data[i][j]);
          if (j < ncols - 1)
              fgetc(file);  // Skip the comma
      }
  }

  /* Close the file */
  fclose(file);

  /* Count the number of sediment grains*/
  PetscInt numb_clust = 0;
  for (i = 0; i < nrows; i++) {
    if (file_data[i][3] == 1) {
      numb_clust++;
    }
  }

  /* Initialize pointer vector to show where sed particles are in csv file*/
  PetscInt *indexVector;
  ierr = PetscMalloc1(numb_clust, &indexVector);CHKERRQ(ierr);

  PetscInt numOnes = 0;
  for (i = 0; i < nrows; i++) {
    if (file_data[i][3] == 1){//file_data[i][3] = 1, if i corresponds to sed
      indexVector[numOnes] = i;
      numOnes++;
    }
  }
  
  PetscReal *centX, *centY, *radius;

  ierr = PetscMalloc1(numb_clust, &centX);CHKERRQ(ierr);
  ierr = PetscMalloc1(numb_clust, &centY);CHKERRQ(ierr);
  ierr = PetscMalloc1(numb_clust, &radius);CHKERRQ(ierr);

  PetscInt idx;

  for (PetscInt i = 0; i < numb_clust; i++) {
    idx = indexVector[i];
    centX[i] = file_data[idx][0]/100;
    centY[i] = file_data[idx][1]/100;
    radius[i] = file_data[idx][2]/100;
  }

  PetscPrintf(PETSC_COMM_WORLD, "%d sed particles have been initialized.\n", numb_clust);

  n_act = numb_clust;

  //----- communication  
  ierr = MPI_Bcast(centX,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(centY,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  n_act = numb_clust;
  user->n_actsed = n_act;
  for(jj=0;jj<numb_clust;jj++){
    user->centXsed[jj] = centX[jj];
    user->centYsed[jj] = centY[jj];
    user->radiussed[jj] = radius[jj];
    PetscPrintf(PETSC_COMM_SELF,
          " --- new sed grain %d!!  x %.2e  y %.2e  r %.2e \n",
          jj,centX[jj],centY[jj],radius[jj]);
  }

  // Define number of sed particles
  user->NCsed=numb_clust;

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
          dist=sqrt(SQ(point->point[0]-user->centXsed[aa])+SQ(point->point[1]-user->centYsed[aa]));
          sed += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radiussed[aa]));
        }

        if(sed>1.0) sed=1.0;
        
        user->Phi_sed[ind] = sed;
        ind++;
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }

  /* Free memory */
  for (int i = 0; i < nrows; i++) {
      free(file_data[i]);
  }
  free(file_data);
  ierr = PetscFree(centX);CHKERRQ(ierr);
  ierr = PetscFree(centY);CHKERRQ(ierr);
  ierr = PetscFree(radius);CHKERRQ(ierr);
  ierr = PetscFree(indexVector);CHKERRQ(ierr);


  ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}


PetscErrorCode InitialIceGrains(IGA iga,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       nrows, ncols, i, ii, j, jj, n_act;
  FILE           *file;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscFunctionBegin;

  //----- cluster info
  PetscReal xc=0.0,yc=0.0,rc=0.0, dist=0.0;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  /* Specify the path to your CSV file */
  // PetscStrcpy(filename, "/Users/jacksonbaglino/Desktop/positions.csv");
  PetscStrcpy(filename, "./input/grainReadFile-2.csv");

  /* Open the file for reading */
  file = fopen(filename, "r");
  if (file == NULL) {
    PetscPrintf(PETSC_COMM_WORLD, "Failed to open the file %s\n", filename);
    PetscFinalize();
    return -1;
  }

  /* Count the number of rows and columns in the CSV file */
  nrows = 0;
  ncols = 0;
  char c;
  while ((c = fgetc(file)) != EOF) {
      if (c == ',')
          ncols++;
      else if (c == '\n') {
          nrows++;
          break;
      }
  }
  ncols++;  // Increment ncols by 1 to account for the last column
  while ((c = fgetc(file)) != EOF) {
      if (c == '\n')
          nrows++;
  }

  nrows++; // Increme nrows by 1 to account for the last row

  PetscPrintf(PETSC_COMM_WORLD, "Number of rows: %d, Number of columns: %d\n", nrows, ncols);

  /* Reset the file pointer to the beginning */
  rewind(file);

  /* Allocate memory for the data */
  double **file_data = (double **)malloc(nrows * sizeof(double *));
  if (file_data == NULL) {
      printf("Failed to allocate memory for file_data\n");
      fclose(file);
      return -1;
  }

  for (int i = 0; i < nrows; i++) {
      file_data[i] = (double *)malloc(ncols * sizeof(double));
      if (file_data[i] == NULL) {
          printf("Failed to allocate memory for file_data\n");
          for (int j = 0; j < i; j++) {
              free(file_data[j]);
          }
          free(file_data);
          fclose(file);
          return -1;
      }
  }

  /* Read the data from the CSV file */
  for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
          fscanf(file, "%lf", &file_data[i][j]);
          if (j < ncols - 1)
              fgetc(file);  // Skip the comma
      }
  }

  /* Close the file */
  fclose(file);

  /* Count the number of ice grains*/
  PetscInt numb_clust = 0;
  for (i = 0; i < nrows; i++) {
    if (file_data[i][3] == 0) {
      numb_clust++;
    }
  }

  /* Initialize pointer vector to show where sed particles are in csv file*/
  PetscInt *indexVector;
  ierr = PetscMalloc1(numb_clust, &indexVector);CHKERRQ(ierr);


  PetscInt numZeros = 0;
  for (i = 0; i < nrows; i++) {
    if (file_data[i][3] == 0){//file_data[i][3] = 0, if i corresponds to ice
      indexVector[numZeros] = i;
      numZeros++;
    }
  }
  

  PetscReal *centX, *centY, *radius;

  ierr = PetscMalloc1(numb_clust, &centX);CHKERRQ(ierr);
  ierr = PetscMalloc1(numb_clust, &centY);CHKERRQ(ierr);
  ierr = PetscMalloc1(numb_clust, &radius);CHKERRQ(ierr);

  PetscInt idx;

  for (PetscInt i = 0; i < numb_clust; i++) {
    idx = indexVector[i];
    centX[i] = file_data[idx][0]*(2e-3)/200;
    centY[i] = file_data[idx][1]*(2e-3)/200;
    radius[i] = file_data[idx][2]*sqrt(2*SQ(2e-3))/sqrt(2*SQ(200))*0.975;

    // centX[i] = file_data[idx][0];
    // centY[i] = file_data[idx][1];
    // radius[i] = file_data[idx][2];
  }

  PetscPrintf(PETSC_COMM_WORLD, "%d ice particles have been initialized.\n", numb_clust);

  n_act = numb_clust;

  //----- communication  
  ierr = MPI_Bcast(centX,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(centY,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(radius,numb_clust,MPI_DOUBLE,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n_act,1,MPI_INT,0,PETSC_COMM_WORLD);CHKERRQ(ierr);

  user->n_act = n_act;
  for(jj=0;jj<numb_clust;jj++){
    user->centX[jj] = centX[jj];
    user->centY[jj] = centY[jj];
    user->radius[jj] = radius[jj];
    PetscPrintf(PETSC_COMM_SELF,
        " --- new ice grain %d!!  x %.2e  y %.2e  r %.2e \n",
        jj,centX[jj],centY[jj],radius[jj]);
  }

  // Set number of ice particles
  user->NCice = numb_clust;

  /* Free memory */
  for (int i = 0; i < nrows; i++) {
      free(file_data[i]);
  }
  free(file_data);
  ierr = PetscFree(centX);CHKERRQ(ierr);
  ierr = PetscFree(centY);CHKERRQ(ierr);
  ierr = PetscFree(radius);CHKERRQ(ierr);
  ierr = PetscFree(indexVector);CHKERRQ(ierr);

  // ierr = IGAEndElement(user->iga,&element);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

typedef struct {
  PetscScalar ice,tem,rhov;
} Field;

PetscErrorCode FormInitialCondition(IGA iga, PetscReal t, Vec U,AppCtx *user, 
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
    IGAAxis axisPF0;
    ierr = IGAGetAxis(igaPF,0,&axisPF0);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axisPF0,user->p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axisPF0,user->Nx,0.0,user->Lx,user->C);CHKERRQ(ierr);
    IGAAxis axisPF1;
    ierr = IGAGetAxis(igaPF,1,&axisPF1);CHKERRQ(ierr);
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
    PetscInt i,j;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
        PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx-1) );
        PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );

        //PetscReal ice0 = u[j][i].ice;
        //u[j][i].air = 1.0-ice0;
        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = rho_vs;
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
//        PetscPrintf(PETSC_COMM_SELF,"TEST3 \n");
/*        PetscReal arg1 = sqrt(SQ(x-0.4*user->Lx)+SQ(y-0.2*user->Ly))-0.15*user->Lx;
        PetscReal arg2 = sqrt(SQ(x-0.6*user->Lx)+SQ(y-0.5*user->Ly))-0.18*user->Lx;
        u[j][i].ice = 0.5-0.5*tanh(0.5/user->eps*arg1);
        u[j][i].ice += 0.5-0.5*tanh(0.5/user->eps*arg2);
*/
        PetscReal dist,ice=0.0;
        PetscInt aa;
        for(aa=0;aa<user->n_act;aa++){
          dist=sqrt(SQ(x-user->centX[aa])+SQ(y-user->centY[aa]));
          ice += 0.5-0.5*tanh(0.5/user->eps*(dist-user->radius[aa]));
        }

        if(ice>1.0) ice=1.0;

        u[j][i].ice = ice;    
        u[j][i].tem = user->temp0 + user->grad_temp0[0]*(x-0.5*user->Lx) + user->grad_temp0[1]*(y-0.5*user->Ly);
        
        PetscScalar rho_vs, temp=u[j][i].tem;
        RhoVS_I(user,temp,&rho_vs,NULL);
        u[j][i].rhov = 0.5*rho_vs;
      }
    }

//        PetscPrintf(PETSC_COMM_SELF,"TEST4 \n");
    ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}


typedef struct {
  PetscScalar soil;
} FieldS;

PetscErrorCode FormInitialSoil(IGA igaS,Vec S,AppCtx *user)
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
  PetscInt i,j,kk;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      PetscReal x = user->Lx*(PetscReal)i / ( (PetscReal)(info.mx-1) );
      PetscReal y = user->Ly*(PetscReal)j / ( (PetscReal)(info.my-1) );
      value=0.0;
      for(kk=0;kk<user->n_actsed;kk++){
        dist = sqrt(SQ(x-user->centXsed[kk])+SQ(y-user->centYsed[kk]));
        // value += 0.5-0.5*tanh(100.0*(dist-user->radiussed[kk]));
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

int main(int argc, char *argv[]) {

  // Petsc Initialization rite of passage 
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscLogDouble itim;
  ierr = PetscTime(&itim); CHKERRQ(ierr);

  // Define simulation specific parameters
  AppCtx user;
 
  PetscReal d0_sol = 4.0e-9, d0_sub0 = 1.0e-9, d0_eva0 = 6.0e-8;       //d0   sol 4e-10  sub 9.5e-10  eva 6e-10    
     //note: if minimum ice radius > 0.05mm, d0_sub d0_eva can be increased 2 orders of magnitude (*100) to speed numerics
  PetscReal beta_sol = 125.0, beta_sub0 = 1.4e5, beta_eva0 = 1.53e4;      //bet  sol 12     sub 1.4e5    eva 1.53e4
  PetscReal a1=5.0, a2=0.1581, rho_rhovs = 2.0e5; // at 0C,  rho_rhovs=5e5 at -10C
  PetscReal gamma_iw = 0.033, gamma_iv = 0.109, gamma_wv = 0.076;

  PetscReal d0_sub, d0_eva, beta_sub, beta_eva;
  d0_sub = d0_sub0/rho_rhovs; d0_eva = d0_eva0/rho_rhovs; beta_sub = beta_sub0/rho_rhovs; beta_eva = beta_eva0/rho_rhovs;
  
  gamma_wv=gamma_iw=gamma_iv;
  

  user.flag_it0   = 1;
  user.flag_tIC   = 0;
  user.periodic   = 0;

  user.xi_v       = 1.0e-3; //can be used safely all time
  user.xi_T       = 1.0e-2;//1.0e-3;
  user.flag_xiT   = 1;

  // user.eps        = 5.0e-7;
  user.eps        = 3.0e-6;
  user.Lambd      = 7.0;
  user.air_lim    = 1.0e-6;
  user.phi_L      = 1.0e-8;
  user.nsteps_IC  = 10;

//IMPORTANT: water here denotes the metal/device properties: Steel carbon 1.5%C

  user.lat_sol    = 3.34e5;
  user.lat_sub    = 2.83e6;

  user.thcond_ice = 2.29; //1.27e-6
  user.thcond_wat = 1.30; //36.0; //1.32e-7
  user.thcond_air = 0.02; //1.428e-5

  user.cp_ice     = 1.96e3;
  user.cp_wat     = 45.300; //4.86e2;
  user.cp_air     = 1.044e3;

  user.rho_ice    = 919.0;
  user.rho_wat    = 2203; //7753.0;
  user.rho_air    = 1.341;

  user.dif_vap    = 2.178e-5;
  
  user.T_melt     = 0.0;

  PetscReal lambda_sol, lambda_sub, lambda_eva, tau_sol, tau_sub, tau_eva;
  PetscReal diff_sol = 0.5*(user.thcond_wat/user.rho_wat/user.cp_wat + user.thcond_ice/user.rho_ice/user.cp_ice);
  PetscReal diff_sub = 0.5*(user.thcond_air/user.rho_air/user.cp_air + user.thcond_ice/user.rho_ice/user.cp_ice);
  PetscReal diff_eva = 0.5*(user.thcond_wat/user.rho_wat/user.cp_wat + user.thcond_air/user.rho_air/user.cp_air);
  lambda_sol    = a1*user.eps/d0_sol;
  tau_sol       = user.eps*lambda_sol*(beta_sol/a1 + a2*user.eps/diff_sol );
  lambda_sub    = a1*user.eps/d0_sub;
  tau_sub       = user.eps*lambda_sub*(beta_sub/a1 + a2*user.eps/diff_sub + a2*user.eps/user.dif_vap);
  lambda_eva    = a1*user.eps/d0_eva;
  tau_eva       = user.eps*lambda_eva*(beta_eva/a1 + a2*user.eps/diff_eva + a2*user.eps/user.dif_vap);

  user.mob_sol    = user.eps/3.0/tau_sol;
  user.mob_sub    = user.eps/3.0/tau_sub; 
  user.mob_eva    = user.eps/3.0/tau_eva; 
  user.mav = cbrt(user.mob_sub*user.mob_eva*user.mob_sol);
  user.alph_sol   = lambda_sol/tau_sol; 
  user.alph_sub   = lambda_sub/tau_sub;
  user.alph_eva   = lambda_eva/tau_eva; 
  user.Etai       = gamma_iv + gamma_iw - gamma_wv;
  user.Etaw       = gamma_wv + gamma_iw - gamma_iv;
  user.Etaa       = gamma_iv + gamma_wv - gamma_iw;

  user.mob_sol=user.mob_eva=user.mav=user.mob_sub;

  //user.alph_sol =  user.alph_eva = 0.0;

  PetscPrintf(PETSC_COMM_WORLD,"mav %.4e \n",user.mav);
  PetscPrintf(PETSC_COMM_WORLD,"SOLID: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n",tau_sol,lambda_sol,user.mob_sol,user.alph_sol);
  PetscPrintf(PETSC_COMM_WORLD,"SUBLI: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n",tau_sub,lambda_sub,user.mob_sub,user.alph_sub);
  PetscPrintf(PETSC_COMM_WORLD,"EVAPO: tau %.4e  lambda %.4e  M0 %.4e  alpha %.4e \n",tau_eva,lambda_eva,user.mob_eva,user.alph_eva);


  //initial conditions
  user.temp0      =-0.5;
  user.grad_temp0[0]=0.0/2e-3;     user.grad_temp0[1] = -1.0/1.583e-3;
  user.heat=100.0;
  user.NCsed=0.0; //less than 2000, otherwise update in user
  user.RCsed=0.25e-3;
  user.RCsed_dev=0.4;
  user.NCice=2.0; //less than 2000, otherwise update in user
  user.RCice=0.35e-3;
  user.RCice_dev=0.4;

  //robot leg dimensions, relative to Lx and Ly
  // user.Rx = 0.4;
  // user.Ry = 0.2;

  //domain and mesh characteristics
  PetscReal Lx=2.0e-3, Ly=1.583e-3;
  PetscInt  Nx=400, Ny=400; 
  user.Lx=Lx; user.Ly=Ly; user.Nx=Nx; user.Ny=Ny;
  PetscReal delt_t = 1.0e-4;
  PetscReal t_final =3*7*24*60.0*60.0;
  // PetscReal t_final = 3*60*60;
  // PetscReal t_final = 10*delt_t;
  PetscInt p=1,C=0;
  user.p=p; user.C=C;     // p = order of basis func, C = continuity of basis function

  //output
  user.outp = 1; // if 0 -> output according to t_interv
  user.t_out = 0.0;    user.t_interv = 600.0;

  //boundary conditions
  PetscReal Ttop,Tbot,Tlef,Trig;
  Tlef = user.temp0 - user.grad_temp0[0]*0.5*Lx;
  Trig = user.temp0 + user.grad_temp0[0]*0.5*Lx;  
  Tbot = user.temp0 - user.grad_temp0[1]*0.5*Ly;
  Ttop = user.temp0 + user.grad_temp0[1]*0.5*Ly;

  PetscInt adap = 1;
  PetscInt NRmin = 2, NRmax = 4;
  PetscReal factor = pow(10.0,1.0/8.0);
  PetscReal dtmin = 0.1*delt_t, dtmax = 0.5*user.t_interv;
  if(dtmax>0.5*user.t_interv) PetscPrintf(PETSC_COMM_WORLD,"OUTPUT DATA ERROR: Reduce maximum time step, or increase t_interval \n\n");
  PetscInt max_rej = 10;
  if(adap==1) PetscPrintf(PETSC_COMM_WORLD,"Adapative time stepping scheme: NR_iter %d-%d  factor %.3f  dt0 %.2e  dt_range %.2e-%.2e  \n\n",NRmin,NRmax,factor,delt_t,dtmin,dtmax);

  PetscBool output=PETSC_TRUE,monitor=PETSC_TRUE;
  char initial[PETSC_MAX_PATH_LEN] = {0};
  char PFgeom[PETSC_MAX_PATH_LEN] = {0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Metamorphism Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nx", "number of elements along x dimension", __FILE__, Nx, &Nx, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny", "number of elements along y dimension", __FILE__, Ny, &Ny, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_cond","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial_PFgeom","Load initial ice geometry from file",__FILE__,PFgeom,PFgeom,sizeof(PFgeom),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-meta_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-meta_monitor","Monitor the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"phaseice"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"temperature"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"vap_density"); CHKERRQ(ierr);

  //ierr = IGASetFieldName(iga,1,"phasewat"); CHKERRQ(ierr);
  //ierr = IGASetFieldName(iga,1,"phaseair"); CHKERRQ(ierr);


  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,Nx,0.0,Lx,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  if(user.periodic==1) {ierr = IGAAxisSetPeriodic(axis1,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = IGAAxisSetDegree(axis1,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1,Ny,0.0,Ly,C);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  PetscInt nmb = iga->elem_width[0]*iga->elem_width[1]*(p+1)*(p+1);
  ierr = PetscMalloc(sizeof(PetscReal)*(nmb),&user.Phi_sed);CHKERRQ(ierr);
  ierr = PetscMemzero(user.Phi_sed,sizeof(PetscReal)*(nmb));CHKERRQ(ierr);
  //PetscPrintf(PETSC_COMM_SELF,"nmb %d \n",nmb);

  //Residual and Tangent
  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  //Boundary Condition
  PetscReal rho0_vs;
  RhoVS_I(&user,user.temp0,&rho0_vs,NULL);
  if(user.periodic==0){
    // ierr = IGASetBoundaryValue(iga,0,0,1,Tlef);CHKERRQ(ierr);
    //ierr = IGASetBoundaryForm(iga,0,0,PETSC_TRUE);CHKERRQ(ierr);
    //ierr = IGASetBoundaryValue(iga,0,1,1,Trig);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,1,0,1,Tbot);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,1,1,1,Ttop);CHKERRQ(ierr);
    //ierr = IGASetBoundaryValue(iga,0,0,2,0.4*rho0_vs);CHKERRQ(ierr);
    //ierr = IGASetBoundaryValue(iga,0,0,2,0.5*rho0_vs);CHKERRQ(ierr);
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

  // sed and ice particles (circular)
  ierr = InitialSedGrains(iga,&user);CHKERRQ(ierr);
  ierr = InitialIceGrains(iga,&user);CHKERRQ(ierr);

  //create auxiliary IGA with 1 dof
  IGA igaS;   IGAAxis axis0S;  IGAAxis axis1S;
  ierr = IGACreate(PETSC_COMM_WORLD,&igaS);CHKERRQ(ierr);
  ierr = IGASetDim(igaS,2);CHKERRQ(ierr);
  ierr = IGASetDof(igaS,1);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,0,&axis0S);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0S,Nx,0.0,Lx,C);CHKERRQ(ierr);
  ierr = IGAGetAxis(igaS,1,&axis1S);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis1S,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis1S,Ny,0.0,Ly,C);CHKERRQ(ierr);
  ierr = IGASetFromOptions(igaS);CHKERRQ(ierr);
  ierr = IGASetUp(igaS);CHKERRQ(ierr);

  // Create a vector U and set it to zero
  PetscReal t=0; 
  Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(U);CHKERRQ(ierr);

  // Set the initial condition for U
  ierr = FormInitialCondition(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);

  // Solve the time-stepping problem using the TSSolve function
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  // Destroy the vector U and the TS object ts
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  // Destroy the IGA object iga
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  // Free the memory allocated for the Phi_sed array
  ierr = PetscFree(user.Phi_sed);CHKERRQ(ierr);

  // Measure and print the computation time
  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  // Finalize PETSc
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
  }

  // Create a vector S and set it to zero
  Vec S;
  ierr = IGACreateVec(igaS,&S);CHKERRQ(ierr);
  ierr = VecZeroEntries(S);CHKERRQ(ierr);

  // Set the initial condition for the soil vector S
  ierr = FormInitialSoil(igaS,S,&user);CHKERRQ(ierr);

  // Write the IGA object igaS to a file
  const char *env="folder"; 
  char *dir; 
  dir=getenv(env);
  char filename[256],filevect[256];
  sprintf(filename, "%s/igasoil.dat", dir);
  ierr=IGAWrite(igaS,filename);CHKERRQ(ierr);

  // Write the vector S to a file
  sprintf(filevect, "%s/soil.dat", dir);
  ierr=IGAWriteVec(igaS,S,filevect);CHKERRQ(ierr);

  // Destroy the vector S and the IGA object igaS
  ierr = VecDestroy(&S);CHKERRQ(ierr);
  ierr = IGADestroy(&igaS);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user,initial,PFgeom);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  //ierr = PetscFree(user.gpert);CHKERRQ(ierr);

  ierr = PetscFree(user.Phi_sed);CHKERRQ(ierr);

  PetscLogDouble ltim,tim;
  ierr = PetscTime(&ltim); CHKERRQ(ierr);
  tim = ltim-itim;
  PetscPrintf(PETSC_COMM_WORLD," comp time %e sec  =  %.2f min \n\n",tim,tim/60.0);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
